'use strict';
/**
 * benchmark/eval_rec.js — recognition evaluation over detection-matched lines.
 *
 *   node eval_rec.js --det-report out/<name>.det.json [--case-sensitive] [--out out/<name>.rec.json]
 *   node eval_rec.js --dataset <name> --pred out/<name>.pred.json   (recomputes matching first)
 *
 * Per matched line (det match from eval_det.js):
 *   CER%       = editdistance(gt, hyp) / len(gt)           (aggregate: sum dist / sum len)
 *   Accuracy%  = exact match (case-insensitive by default, whitespace-normalized)
 *   NED        = editdistance / max(1, max(len(gt), len(hyp)))
 */

const path = require('path');
const U = require('./lib/util');
const { loadDataset } = require('./datasets');
const { matchDetection } = require('./eval_det');

function recMetrics(pairs, caseSensitive) {
  let n = 0, exact = 0, sumDist = 0, sumGtLen = 0, sumNed = 0, sumLenMax = 0;
  const perPair = [];
  for (const p of pairs) {
    const gt = U.normText(p.gtText, caseSensitive);
    const hyp = U.normText(p.predText, caseSensitive);
    if (!gt) continue; // empty GT (shouldn't happen for non-ignore matches)
    const dist = U.levenshtein(gt, hyp);
    const lineCer = dist / gt.length;
    const lineNed = dist / Math.max(1, Math.max(gt.length, hyp.length));
    n++;
    sumDist += dist; sumGtLen += gt.length; sumNed += lineNed; sumLenMax += Math.max(1, Math.max(gt.length, hyp.length));
    if (gt === hyp) { exact++; perPair.push({ img: p.img, gt: p.gtText, hyp: p.predText, ok: true }); }
    else perPair.push({ img: p.img, gt: p.gtText, hyp: p.predText, ok: false, cer: Number(lineCer.toFixed(4)), ned: Number(lineNed.toFixed(4)) });
  }
  return {
    n, exact,
    accuracy: n > 0 ? exact / n : 0,
    cer: sumGtLen > 0 ? sumDist / sumGtLen : 0,
    ned: n > 0 ? sumNed / n : 0,
    charLevel: { sumDist, sumGtLen, sumLenMax },
    perPair,
  };
}

async function main() {
  const args = U.parseArgs(process.argv.slice(2));
  const caseSensitive = args.flags['case-sensitive'] === 'true';
  let dataset = args.flags.dataset;
  let pairs = null;
  let outPath;

  const detReport = args.flags['det-report'];
  if (detReport) {
    const doc = U.readJson(detReport, null);
    if (!doc || !Array.isArray(doc.matches)) { U.fail('rec', 'bad det report: ' + U.rel(detReport)); process.exit(2); }
    pairs = doc.matches; dataset = doc.aggregate && doc.aggregate.dataset;
    outPath = args.flags.out || path.join(U.OUT_DIR, dataset + '.rec.json');
  } else {
    dataset = dataset || 'synth';
    const predPath = args.flags.pred || path.join(U.OUT_DIR, dataset + '.pred.json');
    const detPath = args.flags['det-file'] || path.join(U.OUT_DIR, dataset + '.det.json');
    const gtSamples = loadDataset(dataset);
    const predDoc = U.readJson(predPath, null);
    if (!predDoc || !Array.isArray(predDoc.records)) { U.fail('rec', 'missing predictions: ' + U.rel(predPath)); process.exit(2); }
    const byImg = new Map(predDoc.records.map(r => [U.rel(r.imgPath), r]));
    const iouThr = parseFloat(args.flags.iou) || 0.5;
    pairs = [];
    for (const s of gtSamples) {
      const rec = byImg.get(U.rel(s.imgPath));
      const preds = rec && rec.ok ? (rec.results || []) : [];
      const m = matchDetection(s.gtWords.map(g => ({ text: g.text, box: g.box, ignore: g.ignore })), preds.map(p => p), iouThr, caseSensitive);
      for (const mm of m.matches) pairs.push({ img: path.basename(s.imgPath), gtText: mm.gtText, predText: mm.predText, iou: mm.iou });
    }
    outPath = args.flags.out || path.join(U.OUT_DIR, dataset + '.rec.json');
  }

  const metrics = recMetrics(pairs, caseSensitive);
  const report = {
    aggregate: Object.assign({ dataset, caseSensitive, generatedAt: new Date().toISOString() }, metrics),
  };
  U.writeJson(outPath, report);
  const a = report.aggregate;
  const table = U.markdownTable([
    { metric: 'matched lines', v: a.n },
    { metric: 'exact-match Accuracy', v: U.fmtPct(a.accuracy) },
    { metric: 'CER%', v: U.fmtPct(a.cer) },
    { metric: 'NED (mean per line)', v: a.ned.toFixed(4) },
  ], ['metric', 'v'], ['metric', dataset]);
  console.log('\n' + table + '\n');
  U.log('rec', `rec report (${dataset}) -> ${U.rel(outPath)}`);
  process.exit(0);
}

if (require.main === module) {
  main().catch(e => { U.fail('rec', String(e && e.stack || e)); process.exit(1); });
}

module.exports = { recMetrics };
