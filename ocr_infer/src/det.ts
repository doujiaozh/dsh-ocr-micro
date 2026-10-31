/**
 * DSH OCR Micro - DB (Differentiable Binarization) text detection.
 *
 * Mirrors PaddleOCR's DBPostProcess pipeline in pure JS:
 *   probability map -> threshold at detThresh -> connected components ->
 *   rotating-calipers min-area rect -> score filter -> unclip expansion.
 *
 * The exported inference models for PP-OCR det expose the *sigmoid* map
 * [1,1,H,W] as their single output, so post-processing needs no extra
 * threshold map branch.
 */

import { getSession } from './session';
import {
  RGBAImage,
  resizeRgbaToNCHW, resizeFloatMap, minAreaRect, unclipRect, quadAABB, readingOrderQuad, clampQuadCopy,
} from './geometry';
import { DetBox, OcrOptions } from './types';

// PaddleOCR DB det expects (x/255 - 0.5)/0.5 (same as rec pipeline), NOT ImageNet.
const HALF255_MEAN = [0.5, 0.5, 0.5];
const HALF255_STD = [0.5, 0.5, 0.5];

export interface DetectOptions {
  modelPath: string;
  detThresh?: number;
  boxScoreThresh?: number;
  unclipRatio?: number;
  detLimitSideLen?: number;
  maxCandidates?: number;
  minArea?: number;
  threads?: number;
}

/** DB det side length: resize so max(h,w) <= limit, then round to multiples of 32. */
export function detResizeDims(w: number, h: number, limit: number): { rw: number; rh: number } {
  const ratio = limit / Math.max(w, h);
  let rh = Math.round(h * ratio);
  let rw = Math.round(w * ratio);
  rh = Math.max(32, Math.round(rh / 32) * 32);
  rw = Math.max(32, Math.round(rw / 32) * 32);
  return { rw, rh };
}

interface Component {
  size: number;
  points: number[][]; // [x, y] pixel coords
}

/**
 * Iterative connected-component labeling on a binary mask. Uses an explicit
 * stack (no recursion) and 8-connectivity. W x H <= 960x960 in practice, so
 * worst-case memory stays well under a few MB on 8GB machines.
 */
export function connectedComponents(mask: Uint8Array, w: number, h: number): Component[] {
  const visited = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  const comps: Component[] = [];
  for (let start = 0; start < mask.length; start++) {
    if (mask[start] === 0 || visited[start] === 1) continue;
    let size = 0;
    let sp = 0;
    stack[sp++] = start;
    visited[start] = 1;
    const points: number[][] = [];
    while (sp > 0) {
      const idx = stack[--sp];
      const y = (idx / w) | 0;
      const x = idx - y * w;
      points.push([x, y]);
      size++;
      const y0 = y > 0 ? -1 : 0;
      const y1 = y < h - 1 ? 1 : 0;
      const x0 = x > 0 ? -1 : 0;
      const x1 = x < w - 1 ? 1 : 0;
      for (let dy = y0; dy <= y1; dy++) {
        for (let dx = x0; dx <= x1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const n = idx + dy * w + dx;
          if (mask[n] === 1 && visited[n] === 0) {
            visited[n] = 1;
            stack[sp++] = n;
          }
        }
      }
    }
    comps.push({ size, points });
  }
  return comps;
}

/** Mean probability over the component pixels (proxy for DB box score). */
function scoreComponent(prob: Float32Array, points: number[][], w: number): number {
  let sum = 0;
  for (const p of points) sum += prob[p[1] * w + p[0]];
  return points.length ? sum / points.length : 0;
}

export async function detectText(img: RGBAImage, opts: DetectOptions): Promise<DetBox[]> {
  const limit = opts.detLimitSideLen ?? 960;
  const detThresh = opts.detThresh ?? 0.3;
  const boxThresh = opts.boxScoreThresh ?? 0.6;
  const unclip = opts.unclipRatio ?? 1.5;
  const maxCandidates = opts.maxCandidates ?? 1000;
  const minArea = opts.minArea ?? 10;

  const { rw, rh } = detResizeDims(img.width, img.height, limit);
  // PaddleOCR DB det expects (x/255 - 0.5)/0.5 (same as rec), NOT ImageNet.
  const input = resizeRgbaToNCHW(img, rw, rh, HALF255_MEAN, HALF255_STD);
  const session = await getSession(opts.modelPath, opts.threads);

  const inputs: Record<string, { data: Float32Array; dims: number[] }> = {};
  inputs[session.inputNames[0]] = { data: input, dims: [1, 3, rh, rw] };
  const outputs = await session.run(inputs);
  const out = outputs[session.outputNames[0]];
  const detDims = out.dims || [1, 1, rh, rw];
  const mapW = detDims[detDims.length - 1];
  const mapH = detDims[detDims.length - 2];
  const detMap = resizeFloatMap(out.data, mapW, mapH, img.width, img.height);

  const mask = new Uint8Array(img.width * img.height);
  for (let i = 0; i < detMap.length; i++) mask[i] = detMap[i] > detThresh ? 1 : 0;

  const comps = connectedComponents(mask, img.width, img.height);
  const boxes: DetBox[] = [];
  for (const comp of comps) {
    if (comp.size < minArea) continue;
    if (boxes.length >= maxCandidates) break;
    const score = scoreComponent(detMap, comp.points, img.width);
    if (score < boxThresh) continue;
    const rect = minAreaRect(comp.points);
    if (rect.width <= 0 || rect.height <= 0) continue;
    // Guard against degenerate thin lines by enforcing a 2px minimum height.
    const effective = rect.height < 2
      ? { ...rect, height: 2, corners: rect.corners }
      : rect;
    // Public box = TIGHT text rect (the minAreaRect core), so layout metrics
    // like IoU vs tight GT are clean. The padded (unclipped) quad is exposed
    // separately as `crop` for the recognizer - padding lifts rec accuracy
    // (ascenders/descenders) without inflating the reported geometry.
    const tight = clampQuadCopy(readingOrderQuad(effective.corners), img.width, img.height);
    const crop = clampQuadCopy(readingOrderQuad(unclipRect(effective, unclip)), img.width, img.height);
    const aabb = quadAABB(crop);
    if (aabb.x1 - aabb.x0 < 2 || aabb.y1 - aabb.y0 < 2) continue;
    // Guard extreme aspect: DB occasionally leaks page-wide strips; cap ratio.
    if ((aabb.x1 - aabb.x0) > 20 * (aabb.y1 - aabb.y0 + 1)) continue;
    boxes.push({
      box: tight.map(p => [p[0], p[1]] as [number, number]),
      crop: crop.map(p => [p[0], p[1]] as [number, number]),
      score,
    });
  }

  // Reading-friendly order: top-to-bottom, then left-to-right per text band.
  boxes.sort((a, b) => {
    const ay = (a.box[0][1] + a.box[2][1]) / 2;
    const by = (b.box[0][1] + b.box[2][1]) / 2;
    const ha = Math.abs(a.box[2][1] - a.box[0][1]) || 1;
    const hb = Math.abs(b.box[2][1] - b.box[0][1]) || 1;
    const tol = Math.max(ha, hb) * 0.6;
    if (Math.abs(ay - by) <= tol) {
      return (a.box[0][0] + a.box[2][0]) / 2 - (b.box[0][0] + b.box[2][0]) / 2;
    }
    return ay - by;
  });

  return boxes;
}

export function detOptionsFrom(options: OcrOptions, modelPath: string): DetectOptions {
  return {
    modelPath,
    detThresh: options.detThresh,
    boxScoreThresh: options.boxScoreThresh,
    unclipRatio: options.unclipRatio,
    detLimitSideLen: options.detLimitSideLen,
    threads: options.threads,
  };
}
