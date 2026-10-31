/**
 * Binarization family: global Otsu, adaptive mean, and Sauvola (the document
 * workhorse - handles uneven illumination that defeats Otsu).
 */

import { GrayImage } from './types';
import { integralImage, integralRect, integralSquareImage } from './gray';

function toBinary(src: Uint8Array, pass: (i: number, v: number) => boolean): Uint8Array {
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = pass(i, src[i]) ? 255 : 0;
  return out;
}

/** Otsu threshold maximizing between-class variance. Returns the threshold. */
export function otsuThreshold(g: GrayImage): number {
  const hist = new Array<number>(256).fill(0);
  for (let i = 0; i < g.data.length; i++) hist[g.data[i]]++;
  const total = g.data.length;
  let sumAll = 0;
  for (let t = 0; t < 256; t++) sumAll += t * hist[t];
  let sumB = 0;
  let wB = 0;
  let bestT = 0;
  let bestVar = -1;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > bestVar) { bestVar = between; bestT = t; }
  }
  return bestT;
}

/** Global Otsu binarization: text -> 0, background -> 255 (dark-on-light). */
export function otsuBinarize(g: GrayImage): GrayImage {
  const t = otsuThreshold(g);
  const out = toBinary(g.data, (_i, v) => v <= t);
  return { width: g.width, height: g.height, data: out };
}

/**
 * Adaptive mean binarization (OpenCV-ish ADAPTIVE_THRESH_MEAN_C):
 * pixel > localMean - C ? 255 : 0.
 */
export function adaptiveMeanBinarize(
  g: GrayImage, blockSize = 31, C = 5
): GrayImage {
  const w = g.width, h = g.height;
  const stride = w + 1;
  const integral = integralImage(g);
  const half = (blockSize - 1) >> 1;
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - half);
    const y1 = Math.min(h, y + half + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - half);
      const x1 = Math.min(w, x + half + 1);
      const n = (x1 - x0) * (y1 - y0);
      const mean = integralRect(integral, stride, x0, y0, x1, y1) / n;
      out[y * w + x] = g.data[y * w + x] > mean - C ? 255 : 0;
    }
  }
  return { width: w, height: h, data: out };
}

/**
 * Sauvola binarization: T = m * (1 + k*(s/R - 1)).
 * Defaults: window=25, k=0.2, R=128. Robust to shadows and vignetting.
 */
export function sauvolaBinarize(
  g: GrayImage, window = 25, k = 0.2, r = 128
): GrayImage {
  const w = g.width, h = g.height;
  const stride = w + 1;
  const integral = integralImage(g);
  const integralSq = integralSquareImage(g);
  const half = (window - 1) >> 1;
  const out = new Uint8Array(w * h);
  const src = g.data;
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - half);
    const y1 = Math.min(h, y + half + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - half);
      const x1 = Math.min(w, x + half + 1);
      const n = (x1 - x0) * (y1 - y0);
      const sum = integralRect(integral, stride, x0, y0, x1, y1);
      const sumSq = integralRect(integralSq, stride, x0, y0, x1, y1);
      const mean = sum / n;
      const variance = Math.max(0, sumSq / n - mean * mean);
      const std = Math.sqrt(variance);
      const t = mean * (1 + k * (std / r - 1));
      out[y * w + x] = src[y * w + x] > t ? 255 : 0;
    }
  }
  return { width: w, height: h, data: out };
}
