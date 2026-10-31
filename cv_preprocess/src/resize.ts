/**
 * Resolution enhancement: bicubic upsampling + unsharp masking.
 * A trained SR net would cost megabytes we don't have; bicubic + USM is the
 * honest baseline that still net-wins CER on low-resolution captures.
 */

import { RGBAImage } from './types';
import { unsharpMask } from './filters';

function clamp8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : (v + 0.5) | 0;
}

/** Catmull-Rom (a = -0.5) bicubic kernel. */
function cubic(x: number): number {
  const a = -0.5;
  const ax = Math.abs(x);
  if (ax < 1) return (a + 2) * ax * ax * ax - (a + 3) * ax * ax + 1;
  if (ax < 2) return a * ax * ax * ax - 5 * a * ax * ax + 8 * a * ax - 4 * a;
  return 0;
}

/**
 * Bicubic upscale by an integer factor (1..8). Output size = scale * input,
 * guaranteeing exact 2x/4x dimensions for the detector's 32-divisible input.
 */
export function upscaleBicubic(img: RGBAImage, scale: number): RGBAImage {
  if (scale <= 1) return img;
  const sw = img.width, sh = img.height, src = img.data;
  const dw = sw * scale, dh = sh * scale;
  const out = new Uint8Array(dw * dh * 4);
  // Precompute the source x/y coordinates and kernel weights per output line.
  const xs = new Float32Array(dw);
  const xw = new Float32Array(dw * 4);
  const xi = new Int32Array(dw * 4);
  for (let dx = 0; dx < dw; dx++) {
    const sxf = dx / scale;
    const x0 = Math.floor(sxf);
    xs[dx] = sxf - x0;
    for (let k = -1; k <= 2; k++) {
      xi[dx * 4 + k + 1] = Math.min(sw - 1, Math.max(0, x0 + k));
      xw[dx * 4 + k + 1] = cubic(sxf - (x0 + k));
    }
  }
  for (let dy = 0; dy < dh; dy++) {
    const syf = dy / scale;
    const y0 = Math.floor(syf);
    const yw = [cubic(syf - (y0 - 1)), cubic(syf - y0), cubic(syf - (y0 + 1)), cubic(syf - (y0 + 2))];
    const yi = [
      Math.min(sh - 1, Math.max(0, y0 - 1)),
      Math.min(sh - 1, Math.max(0, y0)),
      Math.min(sh - 1, Math.max(0, y0 + 1)),
      Math.min(sh - 1, Math.max(0, y0 + 2)),
    ];
    for (let dx = 0; dx < dw; dx++) {
      const fx = xs[dx];
      const xii = xi.subarray(dx * 4, dx * 4 + 4);
      const xwi = xw.subarray(dx * 4, dx * 4 + 4);
      const j = (dy * dw + dx) * 4;
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let oy = 0; oy < 4; oy++) {
          const wY = yw[oy];
          if (wY === 0) continue;
          const base = yi[oy] * sw;
          for (let ox = 0; ox < 4; ox++) {
            const wX = xwi[ox];
            if (wX === 0) continue;
            v += src[(base + xii[ox]) * 4 + c] * wY * wX;
          }
        }
        out[j + c] = clamp8(v);
      }
      out[j + 3] = 255;
    }
  }
  return { width: dw, height: dh, data: out };
}

/**
 * Super-resolution for OCR: bicubic upscale then unsharp mask. The unsharp
 * amount scales mildly with the upscale factor to avoid halos on stroke edges.
 */
export function superResolve(
  img: RGBAImage, scale: number, unsharpAmount = 0.6
): RGBAImage {
  const up = upscaleBicubic(img, scale);
  if (unsharpAmount > 0) {
    return unsharpMask(up, 1, unsharpAmount);
  }
  return up;
}
