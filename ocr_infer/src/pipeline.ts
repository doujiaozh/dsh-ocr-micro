/**
 * DSH OCR Micro - end-to-end pipeline: detect -> reading-order sort ->
 * recognize -> post-correct.
 */

import * as fs from 'fs';
import { decodeImage, enhance, RGBAImage } from '../../cv_preprocess/src';
import { detectText, detOptionsFrom } from './det';
import { recognizeLines, recOptionsFrom } from './rec';
import { postProcessLine } from './postprocess';
import {
  getEngineName, modelBytes, resolveModelPaths,
} from './session';
import {
  OcrLine, OcrOptions, OcrResult, OCR_VERSION,
} from './types';

const nowMs = (): number => Number(process.hrtime.bigint()) / 1e6;

function isBuffer(x: unknown): x is Buffer {
  return typeof x === 'object' && x !== null && (x as Buffer).length !== undefined && typeof (x as Buffer).copy === 'function';
}

export async function runOcr(
  input: string | Buffer | RGBAImage,
  options: OcrOptions = {}
): Promise<OcrResult> {
  const t0 = nowMs();
  const paths = resolveModelPaths(options.modelDir);

  /* ---------------- preprocess: decode + optional CV enhancement ------- */
  const tPre = nowMs();
  let img: RGBAImage;
  if (typeof input === 'string') {
    img = await decodeImage(fs.readFileSync(input));
  } else if (isBuffer(input)) {
    img = await decodeImage(input);
  } else {
    img = input as RGBAImage;
  }
  const enhanceMode = options.enhance ?? 'auto';
  if (enhanceMode === true ||
      (enhanceMode === 'auto' && Math.min(img.width, img.height) < 320)) {
    const r = enhance(img, { minShortSide: 320 });
    img = r.out;
    if (options.verbose) {
      console.error(`[dsh-ocr] cv enhance: angle=${r.meta.angle.toFixed(1)} scale=${r.meta.scale}`);
    }
  }
  const preprocessMs = nowMs() - tPre;

  /* ---------------- detection ------------------------------------------ */
  const tDet = nowMs();
  let boxes = await detectText(img, detOptionsFrom(options, paths.det));
  if (options.maxBoxes && options.maxBoxes > 0) boxes = boxes.slice(0, options.maxBoxes);
  const detMs = nowMs() - tDet;

  /* ---------------- recognition ---------------------------------------- */
  const tRec = nowMs();
  const lines: OcrLine[] = await recognizeLines(
    img, boxes, recOptionsFrom(options, paths.rec, paths.dict)
  );
  const recMs = nowMs() - tRec;

  /* ---------------- post-processing ------------------------------------ */
  const tPost = nowMs();
  const dropScore = options.dropScore ?? 0.5;
  const results = lines
    .filter((l) => l.score >= dropScore)
    .map((l) => ({ text: postProcessLine(l.text, options), score: l.score, box: l.box }))
    .filter((l) => l.text.length > 0);
  const text = results.map((l) => l.text).join('\n');
  const postMs = nowMs() - tPost;

  const sizes = modelBytes(paths.det, paths.rec);
  return {
    ok: true,
    version: OCR_VERSION,
    text,
    results,
    timing: { preprocessMs, detMs, recMs, postMs, totalMs: nowMs() - t0 },
    models: { detBytes: sizes.detBytes, recBytes: sizes.recBytes, engine: await getEngineName() },
  };
}
