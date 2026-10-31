/**
 * Grayscale conversion and integral-image machinery (the shared substrate for
 * adaptive binarization and box blurring).
 */

import { GrayImage, RGBAImage } from './types';

/** BT.601 luma. */
export function toGray(img: RGBAImage): GrayImage {
  const w = img.width, h = img.height, d = img.data;
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0, n = w * h; i < n; i++, j += 4) {
    const v = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
    out[i] = v < 0 ? 0 : v > 255 ? 255 : (v + 0.5) | 0;
  }
  return { width: w, height: h, data: out };
}

/** Summed-area table with a zero first row/column. Size (w+1)*(h+1). */
export function integralImage(g: GrayImage): Float64Array {
  const w = g.width, h = g.height, src = g.data;
  const stride = w + 1;
  const out = new Float64Array(stride * (h + 1));
  for (let y = 0; y < h; y++) {
    const dstRow = (y + 1) * stride;
    const srcRow = y * w;
    const upRow = dstRow - stride;
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      rowSum += src[srcRow + x];
      out[dstRow + x + 1] = out[upRow + x + 1] + rowSum;
    }
  }
  return out;
}

/** Summed-area table of squares, for local variance (Sauvola). */
export function integralSquareImage(g: GrayImage): Float64Array {
  const w = g.width, h = g.height, src = g.data;
  const stride = w + 1;
  const out = new Float64Array(stride * (h + 1));
  for (let y = 0; y < h; y++) {
    const dstRow = (y + 1) * stride;
    const srcRow = y * w;
    const upRow = dstRow - stride;
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      const v = src[srcRow + x];
      rowSum += v * v;
      out[dstRow + x + 1] = out[upRow + x + 1] + rowSum;
    }
  }
  return out;
}

export function integralRect(
  integral: Float64Array, stride: number,
  x0: number, y0: number, x1: number, y1: number
): number {
  return integral[y1 * stride + x1] - integral[y0 * stride + x1] -
    integral[y1 * stride + x0] + integral[y0 * stride + x0];
}

/**
 * Box blur over any single-channel plane using a summed-area table.
 * `radius` is the half-extent in pixels (window = 2*radius+1).
 * Returns the per-pixel local mean as a float plane.
 */
export function boxBlurChannel(
  src: Uint8Array, w: number, h: number, radius: number
): Float32Array {
  const stride = w + 1;
  const integral = new Float64Array(stride * (h + 1));
  for (let y = 0; y < h; y++) {
    const dstRow = (y + 1) * stride;
    const srcRow = y * w;
    const upRow = dstRow - stride;
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      rowSum += src[srcRow + x];
      integral[dstRow + x + 1] = integral[upRow + x + 1] + rowSum;
    }
  }
  const out = new Float32Array(w * h);
  const inv = 1 / ((2 * radius + 1) * (2 * radius + 1));
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(h, y + radius + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(w, x + radius + 1);
      out[y * w + x] = integralRect(integral, stride, x0, y0, x1, y1) * inv;
    }
  }
  return out;
}
