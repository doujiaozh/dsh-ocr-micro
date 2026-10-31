/**
 * Filtering / edge detection: median denoise, unsharp mask, CLAHE, Sobel, Canny.
 */

import { GrayImage, RGBAImage } from './types';
import { boxBlurChannel } from './gray';

/** Median filter (odd kernel size k x k). Salt-and-pepper killer. */
export function medianFilter(g: GrayImage, k = 3): GrayImage {
  const w = g.width, h = g.height, src = g.data;
  const out = new Uint8Array(w * h);
  const half = (k - 1) >> 1;
  const win = new Array<number>(k * k);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let n = 0;
      for (let dy = -half; dy <= half; dy++) {
        const yy = Math.min(h - 1, Math.max(0, y + dy));
        for (let dx = -half; dx <= half; dx++) {
          const xx = Math.min(w - 1, Math.max(0, x + dx));
          win[n++] = src[yy * w + xx];
        }
      }
      win.length = n;
      win.sort((a, b) => a - b);
      out[y * w + x] = win[n >> 1];
    }
  }
  return { width: w, height: h, data: out };
}

function clamp8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : (v + 0.5) | 0;
}

/**
 * Unsharp masking: out = orig + amount * (orig - boxBlur(orig)).
 * Mild defaults (radius=1, amount=0.6) suit post-upsacle OCR sharpening.
 */
export function unsharpMask(img: RGBAImage, radius = 1, amount = 0.6): RGBAImage {
  const w = img.width, h = img.height, src = img.data;
  const out = new Uint8Array(src.length);
  for (let c = 0; c < 3; c++) {
    const plane = new Uint8Array(w * h);
    for (let i = 0, j = c; i < w * h; i++, j += 4) plane[i] = src[j];
    const blur = boxBlurChannel(plane, w, h, radius);
    for (let i = 0, j = c; i < w * h; i++, j += 4) {
      const o = src[j];
      const v = o + amount * (o - blur[i]);
      out[j] = clamp8(v);
    }
  }
  for (let i = 3; i < src.length; i += 4) out[i] = src[i];
  return { width: w, height: h, data: out };
}

/**
 * CLAHE on luma with tile-LUT bilinear blending. tile=8 (grid), clip=2.0.
 * Color is preserved by scaling each channel by lut(luma)/luma.
 */
export function clahe(img: RGBAImage, tile = 8, clip = 2.0): RGBAImage {
  const w = img.width, h = img.height, src = img.data;
  const n = w * h;
  const luma = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    luma[i] = clamp8(0.299 * src[j] + 0.587 * src[j + 1] + 0.114 * src[j + 2]);
  }
  const tilesX = Math.max(1, Math.ceil(w / tile));
  const tilesY = Math.max(1, Math.ceil(h / tile));
  const luts: Float32Array[] = [];
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const hist = new Array<number>(256).fill(0);
      const x0 = tx * tile, x1 = Math.min(w, (tx + 1) * tile);
      const y0 = ty * tile, y1 = Math.min(h, (ty + 1) * tile);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) { hist[luma[y * w + x]]++; count++; }
      }
      // clip histogram and redistribute excess uniformly
      const limit = Math.max(1, (clip * count) / 256);
      let excess = 0;
      for (let t = 0; t < 256; t++) {
        if (hist[t] > limit) { excess += hist[t] - limit; hist[t] = limit; }
      }
      const add = excess / 256;
      const lut = new Float32Array(256);
      let cdf = 0;
      for (let t = 0; t < 256; t++) {
        cdf += hist[t] + add;
        lut[t] = (cdf / (count + excess)) * 255;
      }
      luts.push(lut);
    }
  }
  const out = new Uint8Array(src.length);
  for (let y = 0; y < h; y++) {
    // fractional tile coordinates for bilinear LUT blend
    const fy = (y + 0.5) / tile - 0.5;
    const ty0 = Math.max(0, Math.min(tilesY - 1, Math.floor(fy)));
    const ty1 = Math.min(tilesY - 1, ty0 + 1);
    const wy = Math.max(0, Math.min(1, fy - ty0));
    for (let x = 0; x < w; x++) {
      const fx = (x + 0.5) / tile - 0.5;
      const tx0 = Math.max(0, Math.min(tilesX - 1, Math.floor(fx)));
      const tx1 = Math.min(tilesX - 1, tx0 + 1);
      const wx = Math.max(0, Math.min(1, fx - tx0));
      const l = luma[y * w + x];
      const l00 = luts[ty0 * tilesX + tx0][l];
      const l10 = luts[ty0 * tilesX + tx1][l];
      const l01 = luts[ty1 * tilesX + tx0][l];
      const l11 = luts[ty1 * tilesX + tx1][l];
      const newVal =
        l00 * (1 - wx) * (1 - wy) + l10 * wx * (1 - wy) +
        l01 * (1 - wx) * wy + l11 * wx * wy;
      const gain = l > 0 ? newVal / l : 1;
      const j = (y * w + x) * 4;
      out[j] = clamp8(src[j] * gain);
      out[j + 1] = clamp8(src[j + 1] * gain);
      out[j + 2] = clamp8(src[j + 2] * gain);
      out[j + 3] = src[j + 3];
    }
  }
  return { width: w, height: h, data: out };
}

/** Sobel operators; returns gx, gy and magnitude planes. */
export function sobel(g: GrayImage): { gx: GrayImage; gy: GrayImage; mag: GrayImage } {
  const w = g.width, h = g.height, src = g.data;
  const gxData = new Uint8Array(w * h);
  const gyData = new Uint8Array(w * h);
  const magData = new Uint8Array(w * h);
  const at = (x: number, y: number) => src[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = at(x - 1, y - 1), b = at(x, y - 1), c = at(x + 1, y - 1);
      const d = at(x - 1, y), f = at(x + 1, y);
      const gg = at(x - 1, y + 1), hh = at(x, y + 1), i = at(x + 1, y + 1);
      const gxv = -a + c - 2 * d + 2 * f - gg + i;
      const gyv = -a - 2 * b - c + gg + 2 * hh + i;
      gxData[y * w + x] = clamp8(gxv + 128);
      gyData[y * w + x] = clamp8(gyv + 128);
      magData[y * w + x] = clamp8(Math.sqrt(gxv * gxv + gyv * gyv));
    }
  }
  const mk = (d: Uint8Array): GrayImage => ({ width: w, height: h, data: d });
  return { gx: mk(gxData), gy: mk(gyData), mag: mk(magData) };
}

/** Canny edge detector with NMS + hysteresis. */
export function canny(g: GrayImage, low = 50, high = 120): GrayImage {
  const w = g.width, h = g.height;
  const { mag } = sobel(g);
  // gradient direction quantized into 4 bins; use gx/gy via mag-proxy: recompute
  const src = g.data;
  const nms = new Float32Array(w * h);
  const gxg = new Float32Array(w * h);
  const gyg = new Float32Array(w * h);
  const at = (x: number, y: number) => src[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = at(x - 1, y - 1), b = at(x, y - 1), c = at(x + 1, y - 1);
      const d = at(x - 1, y), f = at(x + 1, y);
      const gg = at(x - 1, y + 1), hh = at(x, y + 1), i = at(x + 1, y + 1);
      gxg[y * w + x] = -a + c - 2 * d + 2 * f - gg + i;
      gyg[y * w + x] = -a - 2 * b - c + gg + 2 * hh + i;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const m = mag.data[idx];
      const gxv = gxg[idx], gyv = gyg[idx];
      // quantize direction to 0/45/90/135
      const ang = Math.atan2(gyv, gxv) * 180 / Math.PI;
      const a = ((ang % 180) + 180) % 180;
      let n1 = 0, n2 = 0;
      if (a < 22.5 || a >= 157.5) { n1 = mag.data[idx - 1] ?? 0; n2 = mag.data[idx + 1] ?? 0; }
      else if (a < 67.5) { n1 = idx - w - 1 >= 0 ? mag.data[idx - w - 1] : 0; n2 = idx + w + 1 < w * h ? mag.data[idx + w + 1] : 0; }
      else if (a < 112.5) { n1 = idx - w >= 0 ? mag.data[idx - w] : 0; n2 = idx + w < w * h ? mag.data[idx + w] : 0; }
      else { n1 = idx - w + 1 >= 0 ? mag.data[idx - w + 1] : 0; n2 = idx + w - 1 < w * h ? mag.data[idx + w - 1] : 0; }
      nms[idx] = (m >= n1 && m >= n2) ? m : 0;
    }
  }
  const out = new Uint8Array(w * h);
  const stack: number[] = [];
  for (let i = 0; i < w * h; i++) {
    if (nms[i] >= high) { out[i] = 255; stack.push(i); }
  }
  // hysteresis: trace connected weak pixels
  while (stack.length > 0) {
    const idx = stack.pop() as number;
    const x = idx % w, y = (idx / w) | 0;
    for (let dy = -1; dy <= 1; dy++) {
      const yy = y + dy;
      if (yy < 0 || yy >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx;
        if (xx < 0 || xx >= w) continue;
        const n = yy * w + xx;
        if (out[n] === 0 && nms[n] >= low) { out[n] = 255; stack.push(n); }
      }
    }
  }
  return { width: w, height: h, data: out };
}
