/**
 * DSH OCR Micro - shared types for the inference core.
 * Contract stability: the CLI --json output is derived from these types and the
 * benchmark module parses them, so do not rename fields without coordination.
 */

export interface Point {
  x: number;
  y: number;
}

/** Four corner points, clockwise starting from top-left. */
export type Quad = [number, number][];

export interface DetBox {
  /** 4x[x, y] corner points in original-image coordinates (tight text rect). */
  box: Quad;
  /** mean DB probability over the text region (0..1). */
  score: number;
  /**
   * Padded quad actually cropped for recognition (unclipRatio expansion of
   * `box`). Absent when it equals `box`. Public box stays tight so layout
   * metrics (IoU vs GT) are unaffected; padding is a rec-internal detail.
   */
  crop?: Quad;
}

export interface RecResult {
  text: string;
  /** mean CTC softmax probability of kept frames (0..1). */
  score: number;
}

export interface OcrLine extends RecResult {
  box: Quad;
}

export interface OcrTiming {
  preprocessMs: number;
  detMs: number;
  recMs: number;
  postMs: number;
  totalMs: number;
}

export interface OcrModelsInfo {
  detBytes: number;
  recBytes: number;
  engine: string;
}

export interface OcrResult {
  ok: boolean;
  version: string;
  text: string;
  results: OcrLine[];
  timing: OcrTiming;
  models: OcrModelsInfo;
}

export interface OcrOptions {
  /** run cv_preprocess.enhance() before detection (auto when image is tiny). */
  enhance?: boolean | 'auto';
  /** det: binarization threshold for the DB probability map. */
  detThresh?: number;
  /** det: drop boxes whose mean probability is below this. */
  boxScoreThresh?: number;
  /** det: unclip expansion ratio. */
  unclipRatio?: number;
  /** det: max side length fed to the detector. */
  detLimitSideLen?: number;
  /** rec: max crop width fed to the recognizer. */
  recMaxWidth?: number;
  /** rec: normalization scheme, 'half255' = (x/255 - 0.5)/0.5 (Paddle PP-OCR). */
  recNormalize?: 'half255' | 'imagenet';
  /** rec: drop recognized lines with score below this. */
  dropScore?: number;
  /** post: fix digit/letter confusions inside numeric runs (0/O, l/1 ...). */
  fixDigitLetter?: boolean;
  /** post: experimental CJK look-alike replacement (opt-in). */
  cnConfusions?: boolean;
  /** max number of detected boxes forwarded to recognition (0 = unlimited). */
  maxBoxes?: number;
  /** ONNX session thread count. */
  threads?: number;
  /** override model directory (default: <repo>/models). */
  modelDir?: string;
  /** detailed console output (never mixed into --json stdout). */
  verbose?: boolean;
}

export const OCR_VERSION = '1.0.0';
