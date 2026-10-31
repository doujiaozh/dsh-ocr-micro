'use strict';
/**
 * benchmark/eval_det.js — detection evaluation with IoU matching (AABB IoU).
 *
 *   node eval_det.js --dataset <name> [--pred out/<name>.pred.json]
 *                     [--iou 0.5] [--case-sensitive] [--out out/<name>.det.json]
 *
 * Both the GT quad and the predicted 4-point box are reduced to their
 * axis-aligned bounding boxes, then matched greedily by IoU.
 * ICDAR don't-care rule: GT words equal to "###" are "ignore" regions —
 *   - they never count as FN when unmatched,
 *   - a prediction that matches an ignore region is neutral (not TP, not FP).
 *
 * Writes <out> and prints a console table; det report also carries the matched
 * (gtText, predText) pairs consumed by eval_rec.js.
 */

const path = require('path');
const U = require('./lib/util');
const { loadDataset } = require('./datasets');

function iouQuadToQuad(a, b) {
  if (!a || !b) return 0;
  return U.iouAABB(U.aabbFromQuad(a), U.aabbFromQuad(b));
}

/**
 * Greedy highest-IoU-first matching.
 * @param gt      [{text, box, ignore}]
 * @param preds   [{text, box, score}]
 * @param iouThr  IoU threshold (default 0.5)
 * @param caseSensitive bool
 * @returns {matches:[{gtIdx,predIdx,iou,gtText,predText}], tp, fp, fn, neutralIgnore, unmatchedGt, unmatchedPred}
 */
function matchDetection(gt, preds, iouThr, caseSensitive) {
  const cand = [];
  for (let gi = 0; gi < gt.length; gi++) {
    if (!gt[gi].box) continue; // word-per-line GT with no geometry: skip in det matching
    for (let pi = 0; pi < preds.length; pi++) {
      if (!preds[pi] || !preds[pi].box) continue;
      const iou = iouQuadToQuad(gt[gi].box, preds[pi].box);
      if (iou >= iouThr) cand.push({ gi, pi, iou });
    }
  }
  cand.sort((a, b) => b.iou - a.iou || a.gi - b.gi || a.pi - b.pi);

  const usedGt = new Set(), usedPred = new Set();
  const matches = [];
  let neutralIgnore = 0, tp = 0;
  for (const c of cand) {
    if (usedGt.has(c.gi) || usedPred.has(c.pi)) continue;
    usedGt.add(c.gi); usedPred.add(c.pi);
    const g = gt[c.gi];
    if (g.ignore) { neutralIgnore++; continue; }  // pred swallowed by don't-care region
    tp++;
    matches.push({ gtIdx: c.gi, predIdx: c.pi, iou: Number(c.iou.toFixed(4)), gtText: g.text, predText: preds[c.pi].text });
  }
  let fn = 0;
  const unmatchedGt = [];
  for (let gi = 0; gi < gt.length; gi++) {
    if (usedGt.has(gi) || gt[gi].ignore || !gt[gi].box) continue;
    fn++; unmatchedGt.push(gi);
  }
  let fp = 0;
  const unmatchedPred = [];
  for (let pi = 0; pi < preds.length; pi++) {
    if (usedPred.has(pi)) continue;
    fp++; unmatchedPred.push(pi);
  }
  return {
    matches, tp, fp, fn, neutralIgnore,
    nIgnoreGt: gt.filter(g => g.ignore).length,
    nGeomGt: gt.filter(g => g.box).length,
    unmatchedGt, unmatchedPred,
  };
}

function prf(tp, fp, fn) {
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
  return { precision, recall, f1 };
}

function evaluateDataset(gtSamples, predRecords, iouThr, caseSensitive) {
  let totTp = 0, totFp = 0, totFn = 0, totNeutral = 0, totIgnore = 0, totGt = 0, totPred = 0;
  const perImage = [];
  const byImg = new Map(predRecords.map(r => [U.rel(r.imgPath), r]));
  for (const s of gtSamples) {
    const key = U.rel(s.imgPath);
    const rec = byImg.get(key);
    const gt = s.gtWords.map(g => ({ text: g.text, box: g.box, ignore: g.ignore }));
    const preds = rec && rec.ok ? (rec.results || []).map(p => ({ text: p.text, box: p.box, score: p.score })) : [];
    const m = matchDetection(gt, preds, iouThr, caseSensitive);
    totTp += m.tp; totFp += m.fp; totFn += m.fn; totNeutral += m.neutralIgnore;
    totIgnore += m.nIgnoreGt; totGt += m.nGeomGt; totPred += preds.length;
    const r = prf(m.tp, m.fp, m.fn);
    perImage.push({ img: path.basename(s.imgPath), tp: m.tp, fp: m.fp, fn: m.fn, precision: r.precision, recall: r.recall, f1: r.f1, matched: m.matches.length });
  }
  const agg = prf(totTp, totFp, totFn);
  return {
    aggregate: {
      dataset: null, images: gtSamples.length,
      tp: totTp, fp: totFp, fn: totFn,
      ignoreGt: totIgnore, neutralizedPreds: totNeutral,
      geomGt: totGt, pred: totPred,
      precision: agg.precision, recall: agg.recall, f1: agg.f1, hmean: agg.f1,
      iouThreshold: iouThr,
    },
    perImage,
    matches: [], // filled by caller across images for eval_rec
  };
}

async function main() {
  const args = U.parseArgs(process.argv.slice(2));
  const dataset = args.flags.dataset || 'synth';
  const predPath = args.flags.pred || path.join(U.OUT_DIR, dataset + '.pred.json');
  const iouThr = parseFloat(args.flags.iou) || 0.5;
  const caseSensitive = args.flags['case-sensitive'] === 'true';
  const out = args.flags.out || path.join(U.OUT_DIR, dataset + '.det.json');

  const gtAll = loadDataset(dataset);
  const predDoc = U.readJson(predPath, null);
  if (!predDoc || !Array.isArray(predDoc.records)) { U.fail('eval', 'missing predictions: ' + U.rel(predPath) + ' (run run_det.js first)'); process.exit(2); }

  // Evaluate only images that were actually predicted (run_det --limit N samples the
  // first N images). Images with no prediction record are reported, not counted as FN.
  const predictedKeys = new Set(predDoc.records.map(r => U.rel(r.imgPath)));
  const gtSamples = gtAll.filter(s => predictedKeys.has(U.rel(s.imgPath)));

  const eval_ = evaluateDataset(gtSamples, predDoc.records, iouThr, caseSensitive);
  // cross-image match list for eval_rec
  {
    const byImg = new Map(predDoc.records.map(r => [U.rel(r.imgPath), r]));
    for (const s of gtSamples) {
      const rec = byImg.get(U.rel(s.imgPath));
      const preds = rec && rec.ok ? (rec.results || []) : [];
      const m = matchDetection(s.gtWords.map(g => ({ text: g.text, box: g.box, ignore: g.ignore })), preds.map(p => p), iouThr, caseSensitive);
      for (const mm of m.matches) eval_.matches.push({ img: path.basename(s.imgPath), gtText: mm.gtText, predText: mm.predText, iou: mm.iou });
    }
  }
  eval_.aggregate.dataset = dataset;
  eval_.aggregate.caseSensitive = caseSensitive;
  eval_.aggregate.gtImagesTotal = gtAll.length;
  eval_.aggregate.gtImagesWithoutPred = gtAll.length - gtSamples.length;
  eval_.meta = { predFile: U.rel(predPath), gtDataset: dataset, generatedAt: new Date().toISOString() };

  U.writeJson(out, eval_);
  const a = eval_.aggregate;
  const table = U.markdownTable([
    { metric: 'TP', v: a.tp }, { metric: 'FP', v: a.fp }, { metric: 'FN', v: a.fn },
    { metric: 'Precision', v: U.fmtPct(a.precision) }, { metric: 'Recall', v: U.fmtPct(a.recall) },
    { metric: 'F1 (Hmean)', v: U.fmtPct(a.f1) },
    { metric: 'ignore-GT (###)', v: a.ignoreGt }, { metric: 'neutralized preds', v: a.neutralizedPreds },
  ], ['metric', 'v'], ['metric', dataset]);
  console.log('\n' + table + '\n');
  U.log('eval', `det report (${dataset}) -> ${U.rel(out)}`);
  process.exit(0);
}

if (require.main === module) {
  main().catch(e => { U.fail('eval', String(e && e.stack || e)); process.exit(1); });
}

module.exports = { matchDetection, prf, evaluateDataset };
