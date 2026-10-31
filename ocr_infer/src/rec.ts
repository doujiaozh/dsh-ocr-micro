/**
 * DSH OCR Micro - CRNN/SVTR text recognition with CTC decoding.
 *
 * Pipeline: perspective-warp each detected quad into a normalized 48xW crop,
 * run the recognizer (output [1, T, C], C = dict length + 1 CTC blank),
 * greedy-decode with duplicate-collapse + blank-removal, and score the line by
 * the mean softmax probability of kept frames.
 */

import * as fs from 'fs';
import { getSession } from './session';
import {
  RGBAImage, IMAGENET_MEAN, IMAGENET_STD,
  perspectiveWarpToNCHW, quadAABB,
} from './geometry';
import { DetBox, OcrLine, OcrOptions, RecResult, Quad } from './types';

export const REC_HEIGHT = 48;

const HALF255_MEAN = [0.5, 0.5, 0.5];
const HALF255_STD = [0.5, 0.5, 0.5];

export interface RecognizeOptions {
  modelPath: string;
  dictPath: string;
  recMaxWidth?: number;
  recNormalize?: 'half255' | 'imagenet';
  threads?: number;
}

const dictCache = new Map<string, string[]>();

/** Load the PaddleOCR-style character dictionary (one char per line). */
export function loadDict(path: string): string[] {
  const hit = dictCache.get(path);
  if (hit) return hit;
  const raw = fs.readFileSync(path, 'utf8');
  const chars: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const ch = line.replace(/\r/g, '');
    if (ch.length === 0 && chars.length === 0) continue;
    chars.push(ch);
  }
  while (chars.length > 0 && chars[chars.length - 1] === '') chars.pop();
  dictCache.set(path, chars);
  return chars;
}

function recDimsForQuad(quad: Quad, maxW: number): { w: number; h: number } {
  // quad corners are clockwise from the top-left: 0=TL 1=TR 2=BR 3=BL.
  const top = Math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]);
  const bottom = Math.hypot(quad[2][0] - quad[3][0], quad[2][1] - quad[3][1]);
  const left = Math.hypot(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1]);
  const right = Math.hypot(quad[2][0] - quad[1][0], quad[2][1] - quad[1][1]);
  const width = (top + bottom) / 2;
  const height = (left + right) / 2;
  const aspect = height > 0 ? width / height : 4;
  // Round to a multiple of 4 so the time-axis (T = W/4) is exact for both
  // CRNN-style and PP-LCNet-style recognizers.
  let w = Math.round(REC_HEIGHT * aspect);
  w = Math.max(8, Math.min(maxW, w));
  w = Math.round(w / 4) * 4;
  w = Math.max(8, Math.min(maxW, w));
  return { w, h: REC_HEIGHT };
}

/**
 * Greedy CTC decode: argmax per frame, collapse consecutive duplicates, drop
 * blanks (index 0), map remaining indices to dict chars.
 */
export function ctcGreedyDecode(
  logits: Float32Array,
  T: number,
  C: number,
  dict: string[]
): RecResult {
  const chars: string[] = [];
  let probSum = 0;
  let prevIdx = -1;
  // PP-OCRv6 rec exports the post-softmax distribution (values in [0,1],
  // frame sums ~1); older CRNN exports ship raw logits. Auto-detect once:
  // a raw-logit tensor necessarily contains far-from-[0,1] values.
  let isProb = true;
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (v < -1e-3 || v > 1.0 + 1e-3) { isProb = false; break; }
  }
  for (let t = 0; t < T; t++) {
    const base = t * C;
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let c = 0; c < C; c++) {
      const v = logits[base + c];
      if (v > bestVal) { bestVal = v; bestIdx = c; }
    }
    if (bestIdx > 0 && bestIdx !== prevIdx) {
      let char = dict[bestIdx - 1];
      if (char === undefined && bestIdx === dict.length + 1 && C === dict.length + 2) {
        // PP-OCRv4/v6 heads emit two classes past the text dict: CTC blank
        // at 0 and a trailing SPACE at dict.length+1 (Paddle appends ' ').
        // Without it the decoded words run together.
        char = ' ';
      }
      if (char !== undefined) {
        chars.push(char);
        // For the probability case softmax(max) is simply bestVal itself.
        if (isProb) {
          probSum += bestVal;
        } else {
          let sum = 0;
          for (let c = 0; c < C; c++) sum += Math.exp(logits[base + c] - bestVal);
          probSum += 1 / sum;
        }
      }
      prevIdx = bestIdx;
    } else {
      prevIdx = bestIdx;
    }
  }
  const text = chars.join('');
  return { text, score: chars.length ? probSum / chars.length : 0 };
}

export async function recognizeLine(
  session: any,
  dict: string[],
  img: RGBAImage,
  quad: Quad,
  opts: RecognizeOptions
): Promise<RecResult> {
  const maxW = opts.recMaxWidth ?? 320;
  const { w, h } = recDimsForQuad(quad, maxW);
  const norm = opts.recNormalize === 'imagenet'
    ? { mean: IMAGENET_MEAN, std: IMAGENET_STD }
    : { mean: HALF255_MEAN, std: HALF255_STD };
  const data = perspectiveWarpToNCHW(img, quad, w, h, norm.mean, norm.std);
  const inputs: Record<string, { data: Float32Array; dims: number[] }> = {};
  inputs[session.inputNames[0]] = { data, dims: [1, 3, h, w] };
  const outputs = await session.run(inputs);
  const out = outputs[session.outputNames[0]];
  const dims: number[] = out.dims || [1, w / 4, dict.length + 1];
  const T = dims[dims.length - 2];
  const C = dims[dims.length - 1];
  return ctcGreedyDecode(out.data, T, C, dict);
}

/** Recognize every detected box. Boxes are processed in detection order. */
export async function recognizeLines(
  img: RGBAImage,
  boxes: DetBox[],
  opts: RecognizeOptions
): Promise<OcrLine[]> {
  const session = await getSession(opts.modelPath, opts.threads);
  const dict = loadDict(opts.dictPath);
  const lines: OcrLine[] = [];
  for (const box of boxes) {
    // The recognizer crops the padded `crop` quad (unclipped) when present;
    // the tight `box` stays the public geometry.
    const res = await recognizeLine(session, dict, img, box.crop || box.box, opts);
    lines.push({ text: res.text, score: res.score, box: box.box });
  }
  return lines;
}

export function recOptionsFrom(options: OcrOptions, det: string, dict: string): RecognizeOptions {
  return {
    modelPath: det,
    dictPath: dict,
    recMaxWidth: options.recMaxWidth,
    recNormalize: options.recNormalize,
    threads: options.threads,
  };
}

/** Unused helper kept for docs: aabb of a quad. */
export function quadBounds(quad: Quad) {
  return quadAABB(quad);
}
