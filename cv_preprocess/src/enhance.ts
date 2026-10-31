/**
 * One-call enhancement orchestration for the OCR pipeline.
 *
 * Design decision: DB (the detector) is *trained on natural images*, so we do
 * NOT force-binarize by default. Instead we fix resolution (power-of-two
 * bicubic upscale), geometry (deskew), and acutance (unsharp). Binarization is
 * available as an opt-in for degraded scans.
 */

import { GrayImage, RGBAImage } from './types';
import { toGray } from './gray';
import { otsuBinarize, sauvolaBinarize } from './binarize';
import { clahe, unsharpMask } from './filters';
import { deskew } from './rotate';
import { upscaleBicubic } from './resize';

export interface EnhanceOptions {
  /** Auto-upscale until the short side reaches this (default 320). */
  minShortSide?: number;
  /** Geometry correction, default true. */
  deskew?: boolean;
  /** 'otsu' | 'sauvola' | false, default false. */
  binarize?: false | 'otsu' | 'sauvola';
  /** Contrast-limited histogram equalization, default false. */
  clahe?: boolean;
  /** Unsharp amount (radius 1), default 0.6. 0 disables. */
  unsharp?: number;
  /** Upscale cap, default 4. */
  maxScale?: number;
}

export interface EnhanceMeta {
  angle: number;
  scale: number;
  binarized: boolean;
}

export interface EnhanceResult {
  out: RGBAImage;
  meta: EnhanceMeta;
  gray?: GrayImage;
  binary?: GrayImage;
}

export function enhance(img: RGBAImage, opts: EnhanceOptions = {}): EnhanceResult {
  const minSide = opts.minShortSide ?? 320;
  const maxScale = opts.maxScale ?? 4;

  // power-of-two upscale (2x/4x) until the short side clears the threshold
  let scale = 1;
  const short0 = Math.min(img.width, img.height);
  while (scale < maxScale && short0 * scale < minSide) scale *= 2;

  let out: RGBAImage = scale > 1 ? upscaleBicubic(img, scale) : img;

  let angle = 0;
  if (opts.deskew !== false) {
    const r = deskew(out);
    out = r.out;
    angle = r.angle;
  }

  if (opts.clahe) out = clahe(out);

  const us = opts.unsharp ?? 0.6;
  if (us > 0) out = unsharpMask(out, 1, us);

  let binary: GrayImage | undefined;
  let binarized = false;
  if (opts.binarize === 'otsu') {
    binary = otsuBinarize(toGray(out));
    binarized = true;
  } else if (opts.binarize === 'sauvola') {
    binary = sauvolaBinarize(toGray(out));
    binarized = true;
  }

  return { out, meta: { angle, scale, binarized }, binary };
}
