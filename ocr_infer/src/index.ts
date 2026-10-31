/**
 * DSH OCR Micro - public entry point.
 *
 *   import { runOcr } from 'dsh-ocr-micro/ocr_infer';
 *   const result = await runOcr('receipt.jpg', { enhance: 'auto' });
 */

export { runOcr } from './pipeline';
export {
  detectText, detResizeDims, connectedComponents, DetectOptions,
} from './det';
export {
  recognizeLines, recognizeLine, loadDict, ctcGreedyDecode,
  RecognizeOptions, REC_HEIGHT,
} from './rec';
export { postProcessLine, fixNumericRuns, fixCjkPunctuation, fixCnConfusions } from './postprocess';
export { getSession, resolveModelPaths, defaultModelDir, getEngineName, modelBytes } from './session';
export { OCR_VERSION } from './types';
export * from './types';
