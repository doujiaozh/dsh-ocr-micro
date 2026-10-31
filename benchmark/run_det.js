'use strict';
/**
 * benchmark/run_det.js — run the OCR CLI over a dataset, one subprocess per image.
 *
 *   node run_det.js --dataset <name> [--limit N] [--out out/<name>.pred.json]
 *                          [--per-image-ms 60000]
 *
 * Invokes:  node <repoRoot>/dist/ocr_infer/src/cli.js <img> --json --drop-score 0
 * (cwd = repo root; dist is produced by the lead's `npm run build`.)
 * --drop-score 0 keeps every detected box in the output so the DETECTION
 * metric measures the detector, not the recognizer's confidence filter.
 * (The rec accuracy table shares this pred.json, so it too reports
 * without drop filtering - a stricter setting for the tiny model.)
 * CLI contract: stdout is ONE JSON line:
 *   {"ok":true,"version":"1.0.0","text":"...","results":[{"text","score","box":[[x,y]x4]}],
 *    "timing":{"preprocessMs","detMs","recMs","postMs","totalMs"},
 *    "models":{"detBytes":N,"recBytes":N}}
 * Errors: nonzero exit code + message on stderr; recorded per-image, never fatal.
 */

const fs = require('fs');
const path = require('path');
const cp = require('child_process');
const U = require('./lib/util');
const { loadDataset } = require('./datasets');

/** Run the CLI on one image. Returns a per-image record. */
function runOne(imgPath, { timeoutMs }) {
  const t0 = process.hrtime.bigint();
  const r = cp.spawnSync(process.execPath, [U.CLI_PATH, imgPath, '--json', '--drop-score', '0'], {
    cwd: U.REPO_ROOT,
    encoding: 'utf8',
    timeout: timeoutMs,
    maxBuffer: 64 * 1024 * 1024,
    windowsHide: true,
  });
  const wallMs = Number(process.hrtime.bigint() - t0) / 1e6;
  const rec = { imgPath: U.rel(imgPath), ok: false, wallMs: Number(wallMs.toFixed(1)) };
  if (r.error) { rec.error = String(r.error.message || r.error); rec.exitCode = null; return rec; }
  rec.exitCode = r.status;
  if (r.status !== 0) {
    rec.error = String(r.stderr || '').split('\n').filter(l => l.trim()).slice(-2).join(' ') || `exit ${r.status}`;
    return rec;
  }
  const stdout = String(r.stdout || '');
  // Contract: one JSON line; be tolerant of stray non-JSON lines.
  const line = stdout.split(/\r?\n/).find(l => l.trim().startsWith('{'));
  if (!line) { rec.error = 'no JSON on stdout'; return rec; }
  try {
    const j = JSON.parse(line);
    rec.ok = j.ok !== false;
    rec.version = j.version;
    rec.text = j.text;
    rec.results = (j.results || []).map(r2 => ({
      text: r2.text,
      score: Number.isFinite(r2.score) ? r2.score : null,
      box: r2.box,
    }));
    rec.timing = j.timing || {};
    rec.models = j.models || {};
  } catch (e) {
    rec.error = 'bad JSON on stdout: ' + String(e.message);
  }
  return rec;
}

async function main() {
  const args = U.parseArgs(process.argv.slice(2));
  const dataset = args.flags.dataset || 'synth';
  const limit = U.toInt(args.flags.limit, 0);
  const timeoutMs = U.toInt(args.flags['per-image-ms'], 60000);
  const out = args.flags.out || path.join(U.OUT_DIR, dataset + '.pred.json');

  if (!U.exists(U.CLI_PATH)) {
    U.fail('det', `CLI not built yet: ${U.rel(U.CLI_PATH)} (run 'npm run build' at repo root)`);
    process.exit(3);
  }
  let samples = [];
  try { samples = loadDataset(dataset, { limit }); } catch (e) { U.fail('det', String(e.message)); process.exit(2); }
  if (!samples.length) { U.fail('det', `dataset '${dataset}' has no image+gt samples on disk`); process.exit(2); }

  U.log('det', `${dataset}: ${samples.length} image(s) via ${U.rel(U.CLI_PATH)}`);
  const records = [];
  let okN = 0, failN = 0;
  let cliModels = null, cliVersion = null;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    const rec = runOne(s.imgPath, { timeoutMs });
    records.push(rec);
    if (rec.ok) { okN++; if (rec.models && !cliModels) { cliModels = rec.models; cliVersion = rec.version; } }
    else failN++;
    const done = i + 1;
    if (done % 10 === 0 || done === samples.length) {
      process.stdout.write(`  [${done}/${samples.length}] ok=${okN} fail=${failN}\r`);
    }
  }
  process.stdout.write('\n');

  const summary = {
    dataset,
    cliPath: U.rel(U.CLI_PATH),
    cliVersion: cliVersion || null,
    nImages: records.length,
    nOk: okN,
    nFail: failN,
    models: cliModels,
    generatedAt: new Date().toISOString(),
  };
  U.writeJson(out, { summary, records });
  U.log('det', `${dataset}: ok=${okN} fail=${failN} -> ${U.rel(out)}`);
  process.exit(failN === 0 ? 0 : failN === records.length ? 4 : 0);
}

if (require.main === module) {
  main().catch(e => { U.fail('det', String(e && e.stack || e)); process.exit(1); });
}

module.exports = { runOne };
