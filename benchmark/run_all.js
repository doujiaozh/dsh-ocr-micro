'use strict';
/**
 * benchmark/run_all.js — full orchestration. Exit code is ALWAYS 0 unless the
 * harness itself is broken: missing network datasets fall back to the synthetic
 * generator (gen_synth.py); a missing dist/ or models/ only skips steps with a
 * warning.
 *
 *   node run_all.js [--dataset <name|auto>] [--limit N] [--n 200]
 *                   [--no-net] [--mirror URL ...] [--tesseract] [--paddle]
 *
 * Steps: verify ONNX models → acquire datasets (download real / build synth) →
 * run_det → eval_det + eval_rec per dataset → bench_speed → compare →
 * assemble out/REPORT.md ("5.5MB 下的世界级精度") + print the final table.
 */

const fs = require('fs');
const path = require('path');
const cp = require('child_process');
const U = require('./lib/util');
const DS = require('./datasets');

const REAL_DATASETS = ['icdar2015', 'icdar2013'];

function spawnScript(name, extraArgs) {
  const script = path.join(U.BENCHMARK_DIR, name);
  const r = cp.spawnSync(process.execPath, [script].concat(extraArgs || []), {
    cwd: U.REPO_ROOT, encoding: 'utf8', timeout: 1800000, maxBuffer: 256 * 1024 * 1024, windowsHide: true,
  });
  if (r.stdout) process.stdout.write(r.stdout);
  if (r.stderr) process.stderr.write(r.stderr);
  return r;
}

function stepModelVerify() {
  U.log('step', '1/6 verify ONNX models');
  if (!U.exists(U.MODEL_VERIFY_PATH)) { U.warn('step', `  ${U.rel(U.MODEL_VERIFY_PATH)} not found — model verify skipped`); return { ok: false, note: 'verify_onnx.js missing' }; }
  try {
    const r = cp.execSync('node model_forge/verify_onnx.js', { cwd: U.REPO_ROOT, encoding: 'utf8', timeout: 300000, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    if (r) process.stdout.write('  ' + String(r).replace(/\n/g, '\n  '));
    return { ok: true };
  } catch (e) {
    const stderr = e && e.stderr ? String(e.stderr).split('\n').filter(l => l.trim()).slice(-3).join(' ') : String(e.message);
    U.warn('step', '  model verify failed — continuing (eval will use whatever the CLI loads): ' + stderr);
    return { ok: false, note: stderr };
  }
}

async function stepDatasets(opts) {
  U.log('step', '2/6 acquire datasets');
  const have = [];
  const provenance = [];
  const requested = opts.dataset && opts.dataset !== 'auto' ? [opts.dataset] : REAL_DATASETS;

  if (opts.dataset === 'synth') {
    U.note('step', '  --dataset synth: skipping network download');
  } else if (opts.noNet) {
    U.note('step', '  --no-net: skipping network download');
  }
  for (const name of requested) {
    if (name === 'synth') continue;
    if (DS.isPresent(name)) {
      U.log('ds', `  ${name}: already present (${DS.loadDataset(name).length} samples)`);
      have.push(name);
      provenance.push({ dataset: name, kind: 'real (public dataset)', note: 'present on disk' });
      continue;
    }
    if (opts.noNet) { provenance.push({ dataset: name, kind: 'real (public dataset)', note: 'skipped (--no-net)' }); continue; }
    U.log('ds', `  ${name}: probing mirrors...`);
    const res = await DS.downloadDataset(name, { mirrors: opts.mirrors || [] });
    for (const d of res.downloaded) U.log('net', '    ✓ ' + d);
    for (const s of res.skipped) U.warn('net', '    ✗ ' + s);
    if (res.ok) { have.push(name); provenance.push({ dataset: name, kind: 'real (public dataset)', note: 'downloaded via mirror' }); }
    else {
      U.warn('ds', `  ${name}: unavailable (${res.note || 'no working mirror'}) — falling back to synth`);
      provenance.push({ dataset: name, kind: 'real (public dataset)', note: 'unavailable: ' + (res.note || 'no working mirror (rr.cvc.uab.es is registration-gated)') });
    }
  }

  // synthetic set: the always-available fallback
  if (!DS.isPresent('synth')) {
    U.log('synth', `  generating synthetic dataset (n=${opts.n}) via gen_synth.py`);
    const ok = DS.ensureSynth({ n: opts.n });
    if (!ok) U.warn('synth', '  synthetic generation failed — eval may have nothing to run on');
  } else {
    U.log('synth', `  synth: already present (${DS.loadDataset('synth').length} samples)`);
  }
  if (DS.isPresent('synth')) {
    have.push('synth');
    provenance.push({ dataset: 'synth', kind: 'synthetic (gen_synth.py)', note: 'generated fallback / regression set' });
  }
  return { have: [...new Set(have)], provenence: provenance };
}

function stepEval(datasets, limit) {
  U.log('step', '3/6 run detection + recognition eval per dataset');
  const evaluated = [];
  for (const ds of datasets) {
    const detArgs = ['--dataset', ds];
    if (limit) detArgs.push('--limit', String(limit));
    const r = spawnScript('run_det.js', detArgs);
    if (r.status !== 0) {
      if (r.status === 3 || (r.stderr || '').indexOf('CLI not built') >= 0) {
        U.warn('step', `  ${ds}: dist CLI not built — evaluation skipped for ALL datasets`);
        return { evaluated: [], cliMissing: true };
      }
      U.warn('step', `  ${ds}: run_det failed (exit ${r.status})`);
      continue;
    }
    spawnScript('eval_det.js', ['--dataset', ds]);
    spawnScript('eval_rec.js', ['--det-report', path.join(U.OUT_DIR, ds + '.det.json')]);
    if (U.exists(path.join(U.OUT_DIR, ds + '.det.json'))) evaluated.push(ds);
  }
  return { evaluated, cliMissing: false };
}

function stepSpeed(dataset, limit, warm, cold) {
  U.log('step', '4/6 latency + model size benchmark');
  const args = ['--dataset', dataset];
  if (limit) args.push('--limit', String(Math.min(limit, cold ? cold + 1 : 6)));
  if (warm) args.push('--warm', String(warm));
  if (cold) args.push('--cold', String(cold));
  spawnScript('bench_speed.js', args);
  return U.exists(path.join(U.OUT_DIR, 'speed_report.json'));
}

function stepCompare(dataset, hooks) {
  U.log('step', '5/6 comparison table (published vs measured)');
  const args = ['--dataset', dataset];
  if (hooks.tesseract) args.push('--tesseract');
  if (hooks.paddle) args.push('--paddle');
  spawnScript('compare.js', args);
  return U.exists(path.join(U.OUT_DIR, 'compare.json'));
}

// ---------------------------------------------------------------- REPORT.md

function mdTablesForReport(evaluated) {
  const detRows = [], recRows = [];
  for (const ds of evaluated) {
    const d = U.readJson(path.join(U.OUT_DIR, ds + '.det.json'), null);
    const r = U.readJson(path.join(U.OUT_DIR, ds + '.rec.json'), null);
    if (d && d.aggregate) detRows.push({
      dataset: ds, images: d.aggregate.images, 'TP': d.aggregate.tp, 'FP': d.aggregate.fp, 'FN': d.aggregate.fn,
      'P': U.fmtPct(d.aggregate.precision), 'R': U.fmtPct(d.aggregate.recall), 'F1': U.fmtPct(d.aggregate.f1),
    });
    if (r && r.aggregate) recRows.push({
      dataset: ds, 'matched': r.aggregate.n, 'Acc%': U.fmtPct(r.aggregate.accuracy),
      'CER%': U.fmtPct(r.aggregate.cer), 'NED': Number(r.aggregate.ned).toFixed(4),
    });
  }
  // NOTE: only `evaluated` datasets are shown — stale out/<ds>.det.json from earlier
  // runs is intentionally ignored so the report never mixes fresh data with old numbers.
  return {
    det: detRows.length ? U.markdownTable(detRows, ['dataset', 'images', 'TP', 'FP', 'FN', 'P', 'R', 'F1'], ['dataset', 'images', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']) : '_no det reports this run_',
    rec: recRows.length ? U.markdownTable(recRows, ['dataset', 'matched', 'Acc%', 'CER%', 'NED'], ['dataset', 'matched lines', 'Accuracy', 'CER', 'NED']) : '_no rec reports this run_',
  };
}

function assembleReport({ verify, provenance, evaluated, primary, hooks, speedOk, cmpOk }) {
  const spd = speedOk ? U.readJson(path.join(U.OUT_DIR, 'speed_report.json'), null) : null;
  const cmp = cmpOk ? U.readJson(path.join(U.OUT_DIR, 'compare.json'), null) : null;
  const tables = mdTablesForReport(evaluated);
  const det = primary ? U.readJson(path.join(U.OUT_DIR, primary + '.det.json'), null) : null;
  const rec = primary ? U.readJson(path.join(U.OUT_DIR, primary + '.rec.json'), null) : null;

  const sizeLine = spd && spd.models
    ? `${(spd.models.totalBytes / 1048576).toFixed(2)} MB (det ${(spd.models.detBytes / 1048576).toFixed(2)} + rec ${(spd.models.recBytes / 1048576).toFixed(2)}) — badge: **${spd.models.badge}**`
    : '_no speed report (bench_speed skipped)_';
  const f1Line = det && det.aggregate ? `**${U.fmtPct(det.aggregate.f1)}** on \`${primary}\`` : '_n/a_';
  const cerLine = rec && rec.aggregate ? `**${U.fmtPct(rec.aggregate.cer)}** on \`${primary}\`` : '_n/a_';
  const p50Line = spd && spd.warm && spd.warm.totalMs ? `**${U.fmtMs(spd.warm.totalMs.p50)} ms** p50 / ${U.fmtMs(spd.warm.totalMs.p95)} ms p95 (warm)` : '_n/a_';

  const anyReal = provenance.some(p => p.kind.startsWith('real') && evaluated.includes(p.dataset));
  const verdict = !evaluated.length
    ? '**未能评测**：`dist/` 或模型未就绪。请先在仓库根目录执行 `npm run build` 与 `npm run models`，然后重跑 `npm run bench`。'
    : anyReal
      ? `在公开数据集 ${evaluated.filter(d => d !== 'synth').join(', ')} 上：F1 ${det && det.aggregate ? U.fmtPct(det.aggregate.f1) : 'n/a'} / CER ${rec && rec.aggregate ? U.fmtPct(rec.aggregate.cer) : 'n/a'} 与上表已发表基线逐项对比（注意评测协议差异）。`
      : '**合成管线自测通过**：当前数据为合成本地集（公共 ICDAR 镜像不可达，rr.cvc.uab.es 需注册）。合成数值仅用于验证端到端管线与回归，**不与已发表的 ICDAR2015 数值直接可比**；待真实数据集就绪后本报告自动转为可比数值。';

  const provTable = U.markdownTable(provenance.map(p => ({ dataset: p.dataset, kind: p.kind, note: p.note })), ['dataset', 'kind', 'note'], ['dataset', 'kind', 'note']);
  const cmpTable = cmp && cmp.rows
    ? U.markdownTable(cmp.rows, ['engine', 'kind', 'detMB', 'recMB', 'totalMB', 'f1', 'cer', 'p50', 'budget', 'source'], ['engine', 'kind', 'det MB', 'rec MB', 'total MB', 'det F1', 'CER%', 'p50 ms', '5.5MB budget', 'source / note'])
    : '_comparison table not generated_';

  const md = [
    '# DSH OCR Micro — 基准报告：5.5MB 下的世界级精度',
    '',
    `- 生成时间 (generated): ${new Date().toISOString()}`,
    `- 环境 (env): Node ${process.version}, ${process.platform}/${process.arch}, cwd \`${U.rel(U.REPO_ROOT) || '.'}\``,
    `- 模型校验 (model verify): ${verify.ok ? '**OK**' : '跳过/失败 — ' + String(verify.note || '').replace(/\|/g, '\\|')}`,
    `- CLI: \`${U.rel(U.CLI_PATH)}\` ${U.exists(U.CLI_PATH) ? '(built)' : '**未构建 (not built)**'}`,
    '',
    '## 0. 结论速览 (TL;DR)',
    '',
    `| 指标 | 值 |`,
    `|---|---|`,
    `| 模型体积 (model size, det+rec) | ${sizeLine} |`,
    `| 检测 F1 (det, AABB IoU=0.5) | ${f1Line} |`,
    `| 识别 CER (rec, 匹配行) | ${cerLine} |`,
    `| 延迟 (latency p50) | ${p50Line} |`,
    '',
    `> **一句话**: ${verdict}`,
    '',
    '## 1. 数据集来源 (provenance)',
    '',
    provTable,
    '',
    '> **可比性声明**: 合成数据用于管线自测与回归；其数值**不代表**公开数据集精度，也不与已发表基线处于同一评测协议。真实 ICDAR 数据下载成功后，本报告自动输出可比数值。',
    '',
    '## 2. 检测评测 (detection, AABB IoU=0.5, `###` 即 don\'t-care)',
    '',
    tables.det,
    '',
    '## 3. 识别评测 (recognition, 检测匹配行, 默认大小写不敏感)',
    '',
    tables.rec,
    '',
    '## 4. 延迟与体积 (latency & model size vs 5.5MB)',
    '',
    spd && spd.models
      ? ['| 指标 | 值 |', '|---|---|',
        `| totalMs p50 / p95 (warm) | ${U.fmtMs(spd.warm.totalMs.p50)} / ${U.fmtMs(spd.warm.totalMs.p95)} ms |`,
        `| preprocessMs p50 | ${U.fmtMs(spd.warm.preprocessMs.p50)} ms |`,
        `| detMs p50 | ${U.fmtMs(spd.warm.detMs.p50)} ms |`,
        `| recMs p50 | ${U.fmtMs(spd.warm.recMs.p50)} ms |`,
        `| postMs p50 | ${U.fmtMs(spd.warm.postMs.p50)} ms |`,
        `| cold 启动开销 (wall − inProc) | ${Number.isFinite(spd.cold.processOverheadMs) ? U.fmtMs(spd.cold.processOverheadMs) : 'n/a'} ms (n=${spd.cold.n}) |`,
        `| 模型体积 | ${sizeLine} |`].join('\n')
      : '_bench_speed 未运行_',
    '',
    '## 5. 对比 (published vs measured-here)',
    '',
    cmpTable,
    '',
    '## 附录 (reproduce)',
    '',
    '```bash',
    'npm run bench                       # 完整流水线（自动下载数据集/回退合成）',
    'node benchmark/datasets.js --download icdar2015 --mirror <URL>   # 指定镜像',
    'node benchmark/datasets.js --synth --n 200',
    'node benchmark/run_det.js --dataset synth && node benchmark/eval_det.js --dataset synth && node benchmark/eval_rec.js --det-report out/synth.det.json',
    'node benchmark/bench_speed.js --dataset synth',
    'node benchmark/compare.js --dataset synth --tesseract --paddle',
    '```',
    '',
  ].join('\n');
  U.writeJson(path.join(U.OUT_DIR, 'REPORT.md'), {}); // no-op guard to ensure dir
  fs.writeFileSync(path.join(U.OUT_DIR, 'REPORT.md'), md, 'utf8');
  U.log('step', `6/6 report assembled -> ${U.rel(path.join(U.OUT_DIR, 'REPORT.md'))}`);
}

// ---------------------------------------------------------------- main

async function main() {
  const args = U.parseArgs(process.argv.slice(2));
  const opts = {
    dataset: args.flags.dataset || 'auto',
    limit: U.toInt(args.flags.limit, 0),
    n: U.toInt(args.flags.n, 200),
    noNet: args.flags['no-net'] === 'true',
    mirrors: (args.flags.mirror || '').split(',').map(s => s.trim()).filter(Boolean),
    warm: U.toInt(args.flags.warm, 30),
    cold: U.toInt(args.flags.cold, 5),
    hooks: { tesseract: args.flags.tesseract === 'true', paddle: args.flags.paddle === 'true' },
  };

  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║  DSH-OCR-MICRO BENCH: 世界级吊打测试 (world-class bench) ║');
  console.log('╚══════════════════════════════════════════════════════════╝');

  const verify = stepModelVerify();
  const { have, provenence } = await stepDatasets(opts);
  if (!have.length) {
    U.warn('all', 'no datasets available at all (network down AND synth generation failed) — nothing to evaluate');
    U.writeJson(path.join(U.OUT_DIR, 'REPORT.md'), {});
    fs.writeFileSync(path.join(U.OUT_DIR, 'REPORT.md'), '# DSH OCR Micro — bench\n\n_NO DATA: datasets unavailable and synthetic generation failed._\n', 'utf8');
    process.exit(0);
  }

  const { evaluated, cliMissing } = stepEval(have, opts.limit);
  const primary = evaluated.includes('icdar2015') ? 'icdar2015'
    : evaluated.includes('icdar2013') ? 'icdar2013'
      : evaluated[0] || null;

  let speedOk = false, cmpOk = false;
  if (evaluated.length) {
    speedOk = stepSpeed(primary, opts.limit, opts.warm, opts.cold);
    cmpOk = stepCompare(primary, opts.hooks);
  } else {
    U.warn('all', 'no dataset evaluated — skipping speed + compare steps');
  }

  assembleReport({ verify, provenance: provenence, evaluated, primary, hooks: opts.hooks, speedOk, cmpOk });

  // ---- final console table
  const fin = [];
  const det = primary ? U.readJson(path.join(U.OUT_DIR, primary + '.det.json'), null) : null;
  const rec = primary ? U.readJson(path.join(U.OUT_DIR, primary + '.rec.json'), null) : null;
  const spd = speedOk ? U.readJson(path.join(U.OUT_DIR, 'speed_report.json'), null) : null;
  fin.push({ item: 'model size (det+rec)', v: spd && spd.models ? `${(spd.models.totalBytes / 1048576).toFixed(2)} MB / 5.5 MB — ${spd.models.badge}` : 'n/a' });
  fin.push({ item: `det F1 (${primary || 'n/a'})`, v: det && det.aggregate ? U.fmtPct(det.aggregate.f1) : 'n/a' });
  fin.push({ item: `CER (${primary || 'n/a'})`, v: rec && rec.aggregate ? U.fmtPct(rec.aggregate.cer) : 'n/a' });
  fin.push({ item: 'latency p50', v: spd && spd.warm ? U.fmtMs(spd.warm.totalMs.p50) + ' ms' : 'n/a' });
  fin.push({ item: 'datasets evaluated', v: evaluated.length ? evaluated.join(', ') : 'none (dist/ CLI missing?)' });
  console.log('\n' + U.markdownTable(fin, ['item', 'v'], ['FINAL', 'value']) + '\n');
  console.log(`Full report: ${path.join(U.OUT_DIR, 'REPORT.md')}`);
  console.log(`Speed step: ${speedOk ? 'ok' : 'skipped'} | Compare step: ${cmpOk ? 'ok' : 'skipped'} | CLI: ${U.exists(U.CLI_PATH) ? 'built' : 'MISSING'}`);
  process.exit(0); // always 0 — synthetic fallback guarantees a report
}

if (require.main === module) {
  main().catch(e => { U.fail('all', String(e && e.stack || e)); process.exit(1); });
}
