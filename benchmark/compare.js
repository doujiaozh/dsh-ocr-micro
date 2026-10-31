'use strict';
/**
 * benchmark/compare.js — comparison table: us (measured) vs published baselines.
 *
 *   node compare.js [--dataset synth] [--limit 20] [--out out/compare.json]
 *                   [--tesseract] [--paddle]
 *
 * Hardcoded PUBLISHED baseline rows are clearly labeled "published" and carry a
 * source URL; cells the upstream project never published are "n/a (not published)"
 * rather than invented. Rows with kind "measured-here" are produced by this repo's
 * own pipeline (out/<dataset>.det.json / .rec.json / speed_report.json).
 *
 * Optional local hooks (run the same IoU matching, same images, same machine):
 *   --tesseract : uses the local `tesseract` binary (tsv output) if installed
 *   --paddle    : uses the local `paddleocr` pip package if importable
 * Both hooks degrade to a skipped row with a note when the engine is absent.
 */

const path = require('path');
const cp = require('child_process');
const fs = require('fs');
const U = require('./lib/util');
const { loadDataset } = require('./datasets');
const { evaluateDataset } = require('./eval_det');
const { recMetrics } = require('./eval_rec');

const BUDGET_MB = U.BUDGET_MB;

/** Published baseline rows. Sizes from official model tables / package metadata. */
const PUBLISHED = [
  {
    engine: 'DSH-OCR-Micro (this repo)', kind: 'measured-here', detMB: null, recMB: null,
    f1: null, cer: null, p50: null, dataset: null, source: 'benchmark/out (this run)',
  },
  {
    engine: 'Tesseract 5.3 LSTM (chi_sim+eng)', kind: 'published',
    detMB: null, recMB: null, totalMB: 64.69,
    f1: 'n/a (not published)', cer: 'n/a (not published)', p50: 'n/a',
    source: 'tessdata repo: eng 22.38MB + chi_sim 42.31MB (github.com/tesseract-ocr/tessdata; no ICDAR2015 metrics published)',
  },
  {
    engine: 'PaddleOCR PP-OCRv4 server (det+rec)', kind: 'published',
    detMB: 110, recMB: 88, totalMB: 198,
    f1: 'det Hmean 69.2%*', cer: 'rec acc 85.19%*', p50: 'n/a',
    source: 'models_list.md (ch_PP-OCRv4_server_det 110M / server_rec 88M); *PaddleX 3.2 benchmark set, NOT ICDAR2015 (paddlepaddle.github.io/PaddleX/3.2)',
  },
  {
    engine: 'PaddleOCR PP-OCRv4 mobile (det+rec)', kind: 'published',
    detMB: 4.7, recMB: 10.0, totalMB: 14.7,
    f1: 'det Hmean 63.8%*', cer: 'rec acc 78.74%*', p50: 'n/a',
    source: 'models_list.md (ch_PP-OCRv4_det 4.70M / rec 10M); *PaddleX 3.2 benchmark set, NOT ICDAR2015',
  },
  {
    engine: 'RapidOCR (rapidocr-onnxruntime 1.4.4)', kind: 'published',
    detMB: null, recMB: null, totalMB: 14.9,
    f1: 'uses PP-OCRv4 mobile models', cer: 'n/a (not published)', p50: 'n/a',
    source: 'PyPI wheel size 14.9MB, models bundled (pypi.org/project/rapidocr-onnxruntime)',
  },
  {
    engine: 'EasyOCR (CRAFT + CRNN, ch_sim+en)', kind: 'published',
    detMB: 79.2, recMB: 14.7, totalMB: 108.6,
    f1: 'n/a (not published)', cer: 'n/a (not published)', p50: 'n/a',
    source: 'detector 79.2MB + recognizer 14.7MB per language (huggingface.co/qualcomm/EasyOCR)',
  },
];

// ---------------------------------------------------------------- local hooks

function tesseractAvailable() {
  try { const r = cp.spawnSync('tesseract', ['--version'], { encoding: 'utf8', timeout: 20000, windowsHide: true }); return r.status === 0 || /tesseract/i.test(String(r.stdout || '')); }
  catch (e) { return false; }
}

/** Run local tesseract over images; return prediction records like run_det.js. */
function runTesseract(images) {
  return images.map(img => {
    const rec = { imgPath: U.rel(img), ok: false, wallMs: 0 };
    const t0 = process.hrtime.bigint();
    const r = cp.spawnSync('tesseract', [img, 'stdout', 'tsv'], { encoding: 'utf8', timeout: 90000, maxBuffer: 64 * 1024 * 1024, windowsHide: true });
    rec.wallMs = Number((process.hrtime.bigint() - t0) / 1e6).toFixed(1);
    if (r.status !== 0) { rec.error = String(r.stderr || '').split('\n')[0]; return rec; }
    rec.ok = true;
    rec.results = [];
    for (const line of String(r.stdout || '').split(/\r?\n/).slice(1)) {
      const f = line.split('\t');
      if (f.length < 12 || f[10] == null) continue;
      if (f[0] !== '5') continue; // word level
      const text = f[11];
      const conf = parseFloat(f[10]);
      if (!text || text.trim() === '' || conf < 20) continue;
      const l = parseFloat(f[6]), t = parseFloat(f[7]), w = parseFloat(f[8]), h = parseFloat(f[9]);
      if (!Number.isFinite(l) || !Number.isFinite(t) || !Number.isFinite(w) || !Number.isFinite(h)) continue;
      rec.results.push({ text, score: Number.isFinite(conf) ? conf / 100 : null, box: [[l, t], [l + w, t], [l + w, t + h], [l, t + h]] });
    }
    return rec;
  });
}

const PADDLE_SCRIPT = [
  'import sys, json',
  'img = sys.argv[1]',
  'out = []',
  'try:',
  '  from paddleocr import PaddleOCR',
  '  ocr = PaddleOCR(use_angle_cls=True, lang="ch") if "use_angle_cls" in PaddleOCR.__init__.__code__.co_varnames else PaddleOCR(lang="ch")',
  '  res = ocr.ocr(img, cls=True) if hasattr(ocr, "ocr") else None',
  '  if res is None:',
  '    r = ocr.predict(img)',
  '    for item in (r if isinstance(r, list) else [r]):',
  '      polys = getattr(item, "dt_polys", None) or item.get("dt_polys") if isinstance(item, dict) else getattr(item, "dt_polys", None)',
  '      texts = getattr(item, "rec_texts", None) or item.get("rec_texts") if isinstance(item, dict) else getattr(item, "rec_texts", None)',
  '      scores = getattr(item, "rec_scores", None) or item.get("rec_scores") if isinstance(item, dict) else getattr(item, "rec_scores", None)',
  '      for i in range(len(texts or [])):',
  '        box = [[float(x) for x in p] for p in (polys[i] if i < len(polys) else [])]',
  '        out.append({"text": str(texts[i]), "box": box})',
  '  else:',
  '    for page in (res[0] if res and res[0] else []):',
  '      box, (text, score) = page[0], page[1]',
  '      out.append({"text": text, "box": [[float(x) for x in p] for p in box]})',
  'except Exception as e:',
  '  print(json.dumps({"error": str(e)})); sys.exit(0)',
  'print(json.dumps(out, ensure_ascii=False))',
].join('\n');

function paddleAvailable() {
  const py = U.findPython();
  if (!py) return false;
  const r = cp.spawnSync(py, ['-c', 'import paddleocr'], { encoding: 'utf8', timeout: 60000, windowsHide: true });
  return r.status === 0;
}

function runPaddle(images) {
  const py = U.findPython();
  return images.map(img => {
    const rec = { imgPath: U.rel(img), ok: false, wallMs: 0 };
    const t0 = process.hrtime.bigint();
    const r = cp.spawnSync(py, ['-c', PADDLE_SCRIPT, img], { encoding: 'utf8', timeout: 180000, maxBuffer: 64 * 1024 * 1024, windowsHide: true });
    rec.wallMs = Number((process.hrtime.bigint() - t0) / 1e6).toFixed(1);
    if (r.status !== 0) { rec.error = String(r.stderr || '').split('\n').slice(-2).join(' '); return rec; }
    const line = String(r.stdout || '').split(/\r?\n/).pop();
    try {
      const j = JSON.parse(line);
      if (j && j.error) { rec.error = j.error; return rec; }
      rec.ok = true;
      rec.results = (Array.isArray(j) ? j : []).filter(p => p && p.text && p.box && p.box.length === 4)
        .map(p => ({ text: p.text, score: null, box: p.box }));
    } catch (e) { rec.error = 'bad paddle json: ' + String(e.message); }
    return rec;
  });
}

/** Evaluate a local engine the exact same way we evaluate ourselves. */
function evalLocalEngine(engine, images, gtSamples) {
  const records = engine === 'tesseract' ? runTesseract(images) : runPaddle(images);
  const ev = evaluateDataset(gtSamples, records, 0.5, false);
  const a = ev.aggregate;
  const metrics = recMetrics(ev.matches, false);
  return { records, aggregate: a, rec: metrics };
}

// ---------------------------------------------------------------- main

async function main() {
  const args = U.parseArgs(process.argv.slice(2));
  const dataset = args.flags.dataset || 'synth';
  const limit = U.toInt(args.flags.limit, 20);
  const out = args.flags.out || path.join(U.OUT_DIR, 'compare.json');
  const rows = [];
  const notes = [];

  // --- our measured row
  const detDoc = U.readJson(path.join(U.OUT_DIR, dataset + '.det.json'), null);
  const recDoc = U.readJson(path.join(U.OUT_DIR, dataset + '.rec.json'), null);
  const spdDoc = U.readJson(path.join(U.OUT_DIR, 'speed_report.json'), null);
  const ours = {
    engine: 'DSH-OCR-Micro (this repo)', kind: 'measured-here',
    detMB: spdDoc && spdDoc.models ? (spdDoc.models.detBytes / 1048576).toFixed(2) : 'n/a',
    recMB: spdDoc && spdDoc.models ? (spdDoc.models.recBytes / 1048576).toFixed(2) : 'n/a',
    totalMB: spdDoc && spdDoc.models ? (spdDoc.models.totalBytes / 1048576).toFixed(2) : 'n/a',
    f1: detDoc && detDoc.aggregate ? U.fmtPct(detDoc.aggregate.f1) + ` (${dataset})` : 'n/a (run eval first)',
    cer: recDoc && recDoc.aggregate ? U.fmtPct(recDoc.aggregate.cer) + ` (${dataset})` : 'n/a',
    p50: spdDoc && spdDoc.warm && spdDoc.warm.totalMs ? U.fmtMs(spdDoc.warm.totalMs.p50) : 'n/a',
    budget: spdDoc && spdDoc.models ? spdDoc.models.badge : 'n/a',
    source: 'benchmark/out/*.json from this run (IoU=0.5, AABB matching)',
  };
  rows.push(ours);
  if (!detDoc) notes.push('no det report for ' + dataset + ' → our F1 marked n/a (node run_all.js)');
  if (!spdDoc) notes.push('no speed report → our p50/size n/a');

  // --- published rows (sizes solid, metrics labelled with benchmark provenance)
  for (const p of PUBLISHED.slice(1)) {
    rows.push({
      engine: p.engine, kind: 'published',
      detMB: p.detMB != null ? String(p.detMB) : 'bundled',
      recMB: p.recMB != null ? String(p.recMB) : 'bundled',
      totalMB: String(p.totalMB), f1: p.f1, cer: p.cer, p50: p.p50,
      budget: p.totalMB <= BUDGET_MB ? `fits in ${BUDGET_MB}MB` : `${(p.totalMB / BUDGET_MB).toFixed(1)}x over budget`,
      source: p.source,
    });
  }

  // --- local engine hooks
  const images = loadDataset(dataset, { limit }).map(s => s.imgPath);
  const gtSamples = loadDataset(dataset, { limit });
  if (args.flags.tesseract === 'true') {
    if (!images.length) notes.push('--tesseract: no images for dataset ' + dataset);
    else if (!tesseractAvailable()) notes.push('--tesseract: local tesseract not found (install tesseract to enable this row)');
    else {
      U.log('cmp', 'running local tesseract on ' + images.length + ' image(s)...');
      const r = evalLocalEngine('tesseract', images, gtSamples);
      rows.splice(1, 0, {
        engine: 'Tesseract (local binary)', kind: 'measured-here (local hook)',
        detMB: 'n/a', recMB: 'n/a', totalMB: 'n/a',
        f1: U.fmtPct(r.aggregate.f1) + ` (${dataset})`, cer: U.fmtPct(r.rec.cer) + ` (${dataset})`,
        p50: 'n/a', budget: 'see tessdata sizes above',
        source: 'measured on this machine, same images + matching (limit ' + images.length + ')',
      });
    }
  } else notes.push('--tesseract hook available (not requested)');
  if (args.flags.paddle === 'true') {
    if (!images.length) notes.push('--paddle: no images for dataset ' + dataset);
    else if (!paddleAvailable()) notes.push('--paddle: local paddleocr pip package not found (pip install paddleocr to enable this row)');
    else {
      U.log('cmp', 'running local paddleocr on ' + images.length + ' image(s)...');
      const r = evalLocalEngine('paddle', images, gtSamples);
      rows.splice(args.flags.tesseract === 'true' ? 2 : 1, 0, {
        engine: 'PaddleOCR (local pip)', kind: 'measured-here (local hook)',
        detMB: 'n/a', recMB: 'n/a', totalMB: 'n/a',
        f1: U.fmtPct(r.aggregate.f1) + ` (${dataset})`, cer: U.fmtPct(r.rec.cer) + ` (${dataset})`,
        p50: 'n/a', budget: 'see PP-OCRv4 sizes above',
        source: 'measured on this machine, same images + matching (limit ' + images.length + ')',
      });
    }
  } else notes.push('--paddle hook available (not requested)');

  const table = U.markdownTable(rows, ['engine', 'kind', 'detMB', 'recMB', 'totalMB', 'f1', 'cer', 'p50', 'budget', 'source'],
    ['engine', 'kind', 'det MB', 'rec MB', 'total MB', 'det F1', 'CER%', 'p50 ms', '5.5MB budget', 'source / note']);
  console.log('\n' + table + '\n');
  for (const n of notes) U.note('cmp', n);

  U.writeJson(out, { generatedAt: new Date().toISOString(), dataset, limit, rows, notes });
  U.log('cmp', 'comparison -> ' + U.rel(out));
  process.exit(0);
}

if (require.main === module) {
  main().catch(e => { U.fail('cmp', String(e && e.stack || e)); process.exit(1); });
}

module.exports = { PUBLISHED };
