/**
 * cv_preprocess smoke test: PNG round-trip, skew detection, bicubic scale,
 * Sauvola output domain, deskew, and enhance orchestration.
 */

import { decodeImage, encodePng } from './decode';
import { toGray } from './gray';
import { sauvolaBinarize } from './binarize';
import { detectSkew, deskew, rotateRGBA } from './rotate';
import { upscaleBicubic } from './resize';
import { enhance } from './enhance';
import { RGBAImage } from './types';

const passed: string[] = [];
const failed: string[] = [];

function check(name: string, ok: boolean, extra?: string): void {
  (ok ? passed : failed).push(name);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? ' :: ' + extra : ''}`);
}

function whiteImage(w: number, h: number): RGBAImage {
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const j = i * 4;
    data[j] = 255; data[j + 1] = 255; data[j + 2] = 255; data[j + 3] = 255;
  }
  return { width: w, height: h, data };
}

/** Horizontal stripe text-lines: strong, unambiguous projection signal. */
function stripesImage(w: number, h: number): RGBAImage {
  const img = whiteImage(w, h);
  for (let y = 24; y < h - 12; y += 12) {
    for (let yy = 0; yy < 4 && y + yy < h; yy++) {
      for (let x = 12; x < w - 12; x++) {
        const j = ((y + yy) * w + x) * 4;
        img.data[j] = 20; img.data[j + 1] = 20; img.data[j + 2] = 20; img.data[j + 3] = 255;
      }
    }
  }
  return img;
}

function main(): void {
  /* 1. PNG round trip */
  const src = stripesImage(240, 240);
  const png = encodePng(src);
  const png2 = encodePng(src);
  check('encodePng deterministic', png.equals(png2), `${png.length}B`);
  const back = decodeImage(png);
  check('PNG round trip dims', back.width === src.width && back.height === src.height, `${back.width}x${back.height}`);
  let pixelsOk = true;
  for (let i = 0; i < src.data.length; i++) {
    if (src.data[i] !== back.data[i]) { pixelsOk = false; break; }
  }
  check('PNG round trip pixels', pixelsOk);

  /* 2. skew detection on a 6-degree-rotated page */
  const rot = rotateRGBA(src, 6);
  const angle = detectSkew(toGray(rot));
  check('detectSkew(6deg page)', Math.abs(angle - 6) < 0.6, `detected=${angle.toFixed(2)} expected~6`);

  /* 3. bicubic output size = scale * input */
  const up = upscaleBicubic(src, 2);
  check('bicubic 2x dims', up.width === 480 && up.height === 480, `${up.width}x${up.height}`);

  /* 4. Sauvola output domain is {0, 255} */
  const sau = sauvolaBinarize(toGray(rot), 25, 0.2, 128);
  let domainOk = true;
  for (let i = 0; i < sau.data.length; i++) {
    if (sau.data[i] !== 0 && sau.data[i] !== 255) { domainOk = false; break; }
  }
  check('sauvolaBinarize domain {0,255}', domainOk);
  check('sauvola produces some ink', sau.data.includes(0), 'ink present');

  /* 5. deskew counter-rotates */
  const dk = deskew(rot);
  check('deskew detects angle', Math.abs(dk.angle - 6) <= 0.6, `angle=${dk.angle}`);
  check('deskew actually rotates', dk.out !== rot, 'new buffer');

  /* 6. enhance auto-scale + meta */
  const tiny = stripesImage(120, 120);
  const r = enhance(tiny, { minShortSide: 320 });
  check('enhance upscales tiny image', r.meta.scale === 4 && r.out.width === 480, `scale=${r.meta.scale}`);
  const big = stripesImage(500, 500);
  const r2 = enhance(big, { minShortSide: 320 });
  check('enhance leaves large image unscaled', r2.meta.scale === 1 && r2.out.width === 500, `scale=${r2.meta.scale}`);
  check('enhance default no binarization', r2.meta.binarized === false, `binarized=${r2.meta.binarized}`);

  console.log(`\n${passed.length} passed, ${failed.length} failed`);
  process.exit(failed.length > 0 ? 1 : 0);
}

main();
