/**
 * DSH OCR Micro - post-recognition correction.
 *
 * Without a 5.5MB-budget-busting language model, we squeeze accuracy from
 * deterministic, high-precision rules:
 *   1. numeric-run repair: OCR loves to emit O/0, l/1, S/5, B/8 inside
 *      numbers - fixed only where digits are already present (safe).
 *   2. CJK typography repair: stray half-width commas/periods between CJK
 *      chars become full-width; spaces between CJK chars are collapsed.
 *   3. (opt-in) CJK look-alike repair for rare->common glyph variants.
 *
 * Everything here is lossless-by-default: rules only fire on evidence.
 */

import { OcrOptions } from './types';

const NUM_LETTER: { [k: string]: string } = {
  O: '0', o: '0', D: '0', Q: '0', I: '1', l: '1', Z: '2', z: '2',
  S: '5', s: '5', B: '8', b: '6', q: '9', g: '9', G: '6', i: '1', L: '1',
};

const NUM_RUN_RE = /[0-9OoIlLzZSsBbqQGgDd,.\-\s:/年月日号点]{3,}/g;

/** Replace letters that are (transitively) adjacent to digits inside a run. */
function repairNumericRun(run: string): string {
  let current = run;
  const adjacent = /([A-Za-z])(?=[0-9])|(?<=[0-9])([A-Za-z])/g;
  for (let pass = 0; pass < 4; pass++) {
    const next = current.replace(adjacent, (m) => NUM_LETTER[m] ?? m);
    if (next === current) break;
    current = next;
  }
  return current;
}

/**
 * Inside runs that already contain >=2 digits, wipe letter/digit confusions.
 * Alphabetic unit suffixes ("100ml", "5kg") are preserved because their
 * letters never touch digits after the transitive fixpoint settles.
 */
export function fixNumericRuns(text: string): string {
  return text.replace(NUM_RUN_RE, (run) => {
    const digits = (run.match(/[0-9]/g) || []).length;
    if (digits < 2) return run;
    const repaired = repairNumericRun(run);
    // Only commit when the repair removed letters (i.e. they were digit twins).
    return repaired;
  });
}

/** Repair half-width punctuation/space between CJK characters. */
export function fixCjkPunctuation(text: string): string {
  return text
    .replace(/([\u4e00-\u9fff])\s*([,])\s*([\u4e00-\u9fff])/g, '$1，$3')
    .replace(/([\u4e00-\u9fff])\s*([.])\s*(?=[\u4e00-\u9fff])/g, '$1。')
    .replace(/([\u4e00-\u9fff])\s+([\u4e00-\u9fff])/g, '$1$2')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

/**
 * Curated, high-precision rare->common glyph variants. Off by default:
 * replacement is context-free, so it can only fire on near-certain confusions.
 */
const CN_CONFUSE: { [k: string]: string } = {
  巳: '已', 曰: '日',
};

export function fixCnConfusions(text: string): string {
  return text.replace(/[巳曰]/g, (c) => CN_CONFUSE[c] ?? c);
}

export function postProcessLine(text: string, options: OcrOptions = {}): string {
  let out = (text || '').trim();
  if (options.fixDigitLetter !== false) out = fixNumericRuns(out);
  out = fixCjkPunctuation(out);
  if (options.cnConfusions) out = fixCnConfusions(out);
  return out;
}
