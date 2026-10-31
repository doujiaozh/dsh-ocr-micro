'use strict';
/**
 * benchmark/bench_speed.js — latency benchmark against the 5.5MB model budget.
 *
 *   node bench_speed.js [--dataset synth] [--limit 5] [--warm 30] [--cold 5]
 *                       [--image PATH] [--out out/speed_report.json]
 *
 * Warm  : W sequential CLI invocations on one image (OS page cache warm).
 * Cold  : C fresh-subprocess runs, each on a distinct image, measuring the full
 *         process wall time vs the in-process totalMs (runtime init overhead).
 * Phase timings (preprocess/det/rec/post/total) come from the CLI JSON.
 * Reports p50/p95, cold-start overhead, model bytes vs the 5.5MB budget badge.
 */

const path = require('path');
const cp = require('child_process');
const fs = require('fs');
const U = require('./lib/util');
const { loadDataset } = require('./datasets');
const { runOne } = require('./run_det');

const PHASES = ['preprocessMs', 'detMs', 'recMs', 'postMs'];

function phaseStats(warmRecords) {
  const out = {};
  for (const ph of PHASES.concat('totalMs')) {
    const vals = warmRecords.map(r => r.timing && r.timing[ph]).filter(Number.isFinite);
    out[ph] = {
      n: vals.length,
      p50: U.quantile(vals, 0.5),
      p95: U.quantile(vals, 0.95),
      mean: U.mean(vals),
      min: vals.length ? Math.min(...vals) : NaN,
      max: vals.length ? Math.max(...vals) : NaN,
    };
  }
  return out;
}

async function main() {
  const args = U.parseArgs(process.argv.slice(2));
  const dataset = args.flags.dataset || 'synth';
  const nWarm = U.toInt(args.flags.warm, 30);
  const nCold = U.toInt(args.flags.cold, 5);
  const out = args.flags.out || path.join(U.OUT_DIR, 'speed_report.json');

  if (!U.exists(U.CLI_PATH)) { U.fail('speed', `CLI not built: ${U.rel(U.CLI_PATH)} (run 'npm run build')`); process.exit(3); }

  let images = [];
  if (args.flags.image && U.exists(args.flags.image)) images = [args.flags.image];
  else {
    const samples = loadDataset(dataset, { limit: nCold + 1 });
    images = samples.map(s => s.imgPath);
  }
  if (!images.length) { U.fail('speed', `no images for dataset '${dataset}'`); process.exit(2); }
  const warmImg = images[0];
  const coldImgs = images.slice(1, 1 + nCold);

  // ---- warm runs (same image, back-to-back subprocesses)
  U.log('speed', `warm: ${nWarm} x ${path.basename(warmImg)} (${dataset})`);
  const warm = [];
  for (let i = 0; i < nWarm; i++) {
    const rec = runOne(warmImg, { timeoutMs: 120000 });
    warm.push(rec);
    process.stdout.write(`  warm ${(i + 1).toString().padStart(2)}/${nWarm}  ${U.fmtMs((rec.timing && rec.timing.totalMs) || NaN)} ms\r`);
  }
  process.stdout.write('\n');
  const warmOk = warm.filter(r => r.ok);
  if (!warmOk.length) { U.fail('speed', 'all warm runs failed; cannot benchmark (check CLI output above)'); process.exit(4); }

  // ---- cold runs (fresh process, distinct images)
  U.log('speed', `cold: ${nCold} fresh subprocess run(s) on unseen image(s)`);
  const cold = [];
  if (coldImgs.length) {
    for (let i = 0; i < coldImgs.length; i++) {
      const rec = runOne(coldImgs[i], { timeoutMs: 120000 });
      cold.push(rec);
      U.log('speed', `  cold ${i + 1}/${coldImgs.length}  wall=${U.fmtMs(rec.wallMs)} ms  inProc=${U.fmtMs(rec.timing && rec.timing.totalMs)} ms`);
    }
  } else {
    U.warn('speed', `dataset '${dataset}' has only ${images.length} image(s); cold runs need ${nCold + 1} — skipping cold`);
  }
  const coldOk = cold.filter(r => r.ok);

  // ---- model size vs budget
  const models = warmOk[0].models || {};
  const detBytes = models.detBytes || 0, recBytes = models.recBytes || 0;
  const totalBytes = detBytes + recBytes;
  const totalMB = totalBytes / 1048576;
  const badge = totalBytes <= U.BUDGET_MB * 1048576
    ? `OK under ${U.BUDGET_MB} MB`
    : (totalBytes <= U.BUDGET_MB * 1.05 * 1048576 ? `WARN ~${U.BUDGET_MB} MB (within 5%)` : `OVER ${U.BUDGET_MB} MB`);
  const budgetOk = totalBytes <= U.BUDGET_MB * 1048576;

  const stats = phaseStats(warmOk);
  const t = stats.totalMs;
  const coldWall = coldOk.map(r => r.wallMs);
  const coldIn = coldOk.map(r => r.timing && r.timing.totalMs).filter(Number.isFinite);
  const report = {
    meta: {
      dataset,
      image: U.rel(warmImg),
      cliPath: U.rel(U.CLI_PATH),
      cliVersion: warmOk[0].version || null,
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      nWarm, nCold: coldOk.length,
      generatedAt: new Date().toISOString(),
    },
    warm: stats,
    cold: {
      n: coldOk.length,
      wallMs: { mean: U.mean(coldWall), p50: U.quantile(coldWall, 0.5), p95: U.quantile(coldWall, 0.95) },
      inProcMs: { mean: U.mean(coldIn), p50: U.quantile(coldIn, 0.5) },
      processOverheadMs: Number.isFinite(U.mean(coldWall)) && Number.isFinite(U.mean(coldIn))
        ? Number((U.mean(coldWall) - U.mean(coldIn)).toFixed(1)) : NaN,
    },
    models: {
      detBytes, recBytes, totalBytes, totalMB: Number(totalMB.toFixed(3)),
      budgetMB: U.BUDGET_MB, badge, withinBudget: budgetOk,
      note: detBytes || recBytes ? '' : 'models.*Bytes missing from CLI JSON',
    },
    failedRuns: warm.filter(r => !r.ok).length + cold.filter(r => !r.ok).length,
  };

  U.writeJson(out, report);

  const rows = [
    ['total p50 (ms)', U.fmtMs(t.p50)], ['total p95 (ms)', U.fmtMs(t.p95)],
    ['preprocess p50 (ms)', U.fmtMs(stats.preprocessMs.p50)],
    ['det p50 (ms)', U.fmtMs(stats.detMs.p50)], ['rec p50 (ms)', U.fmtMs(stats.recMs.p50)],
    ['post p50 (ms)', U.fmtMs(stats.postMs.p50)],
    ['cold wall mean (ms)', coldOk.length ? U.fmtMs(report.cold.wallMs.mean) : 'n/a'],
    ['cold process overhead (ms)', coldOk.length ? U.fmtMs(report.cold.processOverheadMs) : 'n/a'],
    ['model det+rec', `${U.fmtMB(totalMB)} (det ${(detBytes / 1048576).toFixed(2)} + rec ${(recBytes / 1048576).toFixed(2)})`],
    ['5.5MB budget', badge],
  ];
  const md = U.markdownTable(rows.map(([k, v]) => ({ k, v })), ['k', 'v'], ['latency / size', dataset]);
  console.log('\n' + md + '\n');
  U.log('speed', `speed report -> ${U.rel(out)}`);
  process.exit(budgetOk ? 0 : 0); // badge is informational, not a failure
}

if (require.main === module) {
  main().catch(e => { U.fail('speed', String(e && e.stack || e)); process.exit(1); });
}

module.exports = { phaseStats };
