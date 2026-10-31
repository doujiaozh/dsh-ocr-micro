/**
 * Rotation and skew correction.
 *
 * detectSkew uses the classic projection-variance method: a document rotated
 * by theta maximizes the variance of its horizontal projection at theta.
 * The search runs on a downscaled Otsu mask for speed (full-size would be
 * ~40x slower with no accuracy gain).
 */

import { GrayImage, RGBAImage } from './types';
import { toGray } from './gray';
import { otsuThreshold } from './binarize';

export interface SkewOptions {
  minAngle?: number;   // default -10
  maxAngle?: number;   // default 10
  step?: number;       // default 0.5
  downscale?: number;  // search image max side, default 400
}

function clamp8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : (v + 0.5) | 0;
}

/** Nearest-neighbor rotate of a binary mask, in-place mapping semantics. */
function rotateMaskNearest(
  mask: Uint8Array, w: number, h: number, rad: number
): Uint8Array {
  const out = new Uint8Array(w * h);
  const cos = Math.cos(rad), sin = Math.sin(rad);
  const cx = w / 2, cy = h / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // inverse map: rotate dest by -rad to find source
      const dx = x - cx, dy = y - cy;
      const sx = Math.round(cx + dx * cos + dy * sin);
      const sy = Math.round(cy - dx * sin + dy * cos);
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
        out[y * w + x] = mask[sy * w + sx];
      }
    }
  }
  return out;
}

/**
 * Estimate the document skew angle in degrees (positive = rotated CCW).
 * Returns 0 when no dominant text-line direction is found.
 */
export function detectSkew(g: GrayImage, opts: SkewOptions = {}): number {
  const minA = opts.minAngle ?? -10;
  const maxA = opts.maxAngle ?? 10;
  const step = opts.step ?? 0.5;
  const maxSide = opts.downscale ?? 400;

  // downscale by integer-ish factor to keep the projection search cheap
  let scale = 1;
  while (Math.max(g.width, g.height) / scale > maxSide) scale++;
  const sw = Math.max(1, (g.width / scale) | 0);
  const sh = Math.max(1, (g.height / scale) | 0);
  const small = new Uint8Array(sw * sh);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const sy = Math.min(g.height - 1, y * scale);
      const sx = Math.min(g.width - 1, x * scale);
      small[y * sw + x] = g.data[sy * g.width + sx];
    }
  }
  const smallGray: GrayImage = { width: sw, height: sh, data: small };
  const t = otsuThreshold(smallGray);
  const mask = new Uint8Array(sw * sh);
  const n = sw * sh;
  let ink = 0;
  for (let i = 0; i < n; i++) {
    mask[i] = small[i] <= t ? 1 : 0;
    ink += mask[i];
  }
  if (ink === 0 || ink === n) return 0;

  let bestAngle = 0;
  let bestVar = -1;
  const rowSums = new Float64Array(sh);
  for (let a = minA; a <= maxA + 1e-9; a += step) {
    const rad = (a * Math.PI) / 180;
    const rot = rotateMaskNearest(mask, sw, sh, rad);
    let total = 0;
    for (let y = 0; y < sh; y++) {
      let s = 0;
      const base = y * sw;
      for (let x = 0; x < sw; x++) s += rot[base + x];
      rowSums[y] = s;
      total += s;
    }
    const mean = total / sh;
    let v = 0;
    for (let y = 0; y < sh; y++) {
      const d = rowSums[y] - mean;
      v += d * d;
    }
    if (v > bestVar) { bestVar = v; bestAngle = a; }
  }
  return bestAngle;
}

/**
 * Bilinear rotate of an RGBA image around its center, same output size,
 * filling exposed corners with `fill` (default opaque white).
 */
export function rotateRGBA(
  img: RGBAImage, angleDeg: number, fill: [number, number, number, number] = [255, 255, 255, 255]
): RGBAImage {
  const w = img.width, h = img.height, src = img.data;
  const out = new Uint8Array(w * h * 4);
  const rad = -(angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad), sin = Math.sin(rad);
  const cx = w / 2, cy = h / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx, dy = y - cy;
      const sx = cx + dx * cos + dy * sin;
      const sy = cy - dx * sin + dy * cos;
      const j = (y * w + x) * 4;
      if (sx < -0.5 || sy < -0.5 || sx > w - 0.5 || sy > h - 0.5) {
        out[j] = fill[0]; out[j + 1] = fill[1]; out[j + 2] = fill[2]; out[j + 3] = fill[3];
        continue;
      }
      const x0 = Math.min(w - 1, Math.max(0, sx | 0));
      const y0 = Math.min(h - 1, Math.max(0, sy | 0));
      const x1 = Math.min(w - 1, x0 + 1);
      const y1 = Math.min(h - 1, y0 + 1);
      const fx = sx - x0, fy = sy - y0;
      const i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4;
      const i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
      const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
      for (let c = 0; c < 4; c++) {
        const v = src[i00 + c] * w00 + src[i10 + c] * w10 + src[i01 + c] * w01 + src[i11 + c] * w11;
        out[j + c] = clamp8(v);
      }
    }
  }
  return { width: w, height: h, data: out };
}

/** Bilinear rotate of a gray plane, same output size, fill default 255. */
export function rotateGray(g: GrayImage, angleDeg: number, fill = 255): GrayImage {
  const w = g.width, h = g.height, src = g.data;
  const out = new Uint8Array(w * h);
  const rad = -(angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad), sin = Math.sin(rad);
  const cx = w / 2, cy = h / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx, dy = y - cy;
      const sx = cx + dx * cos + dy * sin;
      const sy = cy - dx * sin + dy * cos;
      if (sx < -0.5 || sy < -0.5 || sx > w - 0.5 || sy > h - 0.5) {
        out[y * w + x] = fill;
        continue;
      }
      const x0 = Math.min(w - 1, Math.max(0, sx | 0));
      const y0 = Math.min(h - 1, Math.max(0, sy | 0));
      const x1 = Math.min(w - 1, x0 + 1);
      const y1 = Math.min(h - 1, y0 + 1);
      const fx = sx - x0, fy = sy - y0;
      const v =
        src[y0 * w + x0] * (1 - fx) * (1 - fy) + src[y0 * w + x1] * fx * (1 - fy) +
        src[y1 * w + x0] * (1 - fx) * fy + src[y1 * w + x1] * fx * fy;
      out[y * w + x] = clamp8(v);
    }
  }
  return { width: w, height: h, data: out };
}

export interface DeskewOptions {
  detectFrom?: GrayImage;       // precomputed luma; defaults to toGray(img)
  minAngle?: number;            // skip rotation below this (default 0.3 deg)
  angle?: number;               // explicit angle, skips detection
}

/**
 * Deskew an RGBA image: detect the skew angle (optional), round to 0.1 deg,
 * and counter-rotate if it exceeds the threshold.
 */
export function deskew(img: RGBAImage, opts: DeskewOptions = {}): { out: RGBAImage; angle: number } {
  const angleRaw = opts.angle ?? detectSkew(opts.detectFrom ?? toGray(img));
  const angle = Math.round(angleRaw * 10) / 10;
  if (Math.abs(angle) < (opts.minAngle ?? 0.3)) {
    return { out: img, angle: 0 };
  }
  return { out: rotateRGBA(img, -angle), angle };
}
