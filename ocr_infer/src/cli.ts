#!/usr/bin/env node
/**
 * DSH OCR Micro CLI.
 *
 * Single-line JSON contract (consumed by benchmark/):
 *   node dist/ocr_infer/src/cli.js <image> --json
 *   => {"ok":true,"version":..,"text":"..","results":[{"text","score","box"}],
 *       "timing":{..},"models":{..}}
 * Errors are reported on stderr with a nonzero exit code.
 */

import * as fs from 'fs';
import { runOcr } from './pipeline';
import { OcrOptions, OcrResult } from './types';

interface CliArgs {
  image: string | null;
  options: OcrOptions;
  json: boolean;
  out: string | null;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = { image: null, options: {}, json: false, out: null };
  const opts = args.options;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = (): string | null => (i + 1 < argv.length ? argv[++i] : null);
    switch (a) {
      case '--json': args.json = true; break;
      case '--out': args.out = next(); break;
      case '--model-dir': if (next() !== null) (opts as any).modelDir = argv[i]; break;
      case '--threads': { const v = next() !== null ? parseInt(argv[i], 10) : NaN; if (v > 0) opts.threads = v; break; }
      case '--det-thresh': { const v = next() !== null ? parseFloat(argv[i]) : NaN; if (!isNaN(v)) opts.detThresh = v; break; }
      case '--box-thresh': { const v = next() !== null ? parseFloat(argv[i]) : NaN; if (!isNaN(v)) opts.boxScoreThresh = v; break; }
      case '--unclip': { const v = next() !== null ? parseFloat(argv[i]) : NaN; if (!isNaN(v)) opts.unclipRatio = v; break; }
      case '--drop-score': { const v = next() !== null ? parseFloat(argv[i]) : NaN; if (!isNaN(v)) opts.dropScore = v; break; }
      case '--rec-max-width': { const v = next() !== null ? parseInt(argv[i], 10) : NaN; if (v > 0) opts.recMaxWidth = v; break; }
      case '--max-boxes': { const v = next() !== null ? parseInt(argv[i], 10) : NaN; if (v > 0) opts.maxBoxes = v; break; }
      case '--enhance': opts.enhance = true; break;
      case '--no-enhance': opts.enhance = false; break;
      case '--cn-confusions': opts.cnConfusions = true; break;
      case '-v': case '--verbose': opts.verbose = true; break;
      case '--version': console.log(require('../package.json').version || '1.0.0'); process.exit(0);
      case '-h': case '--help':
        console.log(USAGE); process.exit(0);
      default:
        if (a.startsWith('--')) { console.error(`unknown flag: ${a}\n${USAGE}`); process.exit(2); }
        if (!args.image) args.image = a;
        break;
    }
  }
  return args;
}

const USAGE = `DSH OCR Micro - world-class OCR under 5.5MB (pure Node.js + ONNX)
usage: node dist/ocr_infer/src/cli.js <image> [options]

options:
  --json              emit single-line JSON on stdout (benchmark contract)
  --out <file>        also write pretty JSON to a file
  --model-dir <dir>   override models directory (default: <repo>/models)
  --threads <n>       ONNX intra-op threads (default: min(4, cpus))
  --det-thresh <f>    DB binarization threshold (default 0.3)
  --box-thresh <f>    DB box score threshold (default 0.6)
  --unclip <f>        box expansion ratio (default 1.5)
  --drop-score <f>    drop recognized lines below score (default 0.5)
  --rec-max-width <n> recognizer crop width cap (default 320)
  --max-boxes <n>     cap number of boxes recognized
  --enhance|--no-enhance  force cv_preprocess.enhance on/off (default auto)
  --cn-confusions     enable experimental CJK look-alike repair
  -v, --verbose       progress on stderr`;

function prettyPrint(r: OcrResult): void {
  const kb = (b: number) => (b / 1024).toFixed(1) + 'KB';
  console.log(`DSH OCR Micro v${r.version} | engine=${r.models.engine} | ` +
    `det=${kb(r.models.detBytes)} rec=${kb(r.models.recBytes)} total=${kb(r.models.detBytes + r.models.recBytes)}`);
  console.log(`timing: total=${r.timing.totalMs.toFixed(0)}ms ` +
    `[pre=${r.timing.preprocessMs.toFixed(0)} det=${r.timing.detMs.toFixed(0)} ` +
    `rec=${r.timing.recMs.toFixed(0)} post=${r.timing.postMs.toFixed(0)}]`);
  for (const line of r.results) {
    console.log(`  [${(line.score * 100).toFixed(0)}%] ${line.text}`);
  }
  if (r.results.length === 0) console.log('  (no text found)');
  console.log('---');
  console.log(r.text);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (!args.image) { console.error(USAGE); process.exit(2); }
  try {
    const result = await runOcr(args.image, args.options);
    if (args.out) fs.writeFileSync(args.out, JSON.stringify(result, null, 2), 'utf8');
    if (args.json) {
      process.stdout.write(JSON.stringify(result) + '\n');
    } else {
      prettyPrint(result);
    }
    if (result.results.length === 0) process.exit(3);
  } catch (e) {
    const msg = (e && (e as Error).message) ? (e as Error).message : String(e);
    console.error(`[dsh-ocr] failed: ${msg}`);
    process.exit(1);
  }
}

main();
