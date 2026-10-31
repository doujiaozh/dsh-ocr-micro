'use strict';
/**
 * benchmark/datasets.js — dataset registry + downloader + ICDAR ground-truth loaders.
 *
 * Real datasets (registration-gated at rr.cvc.uab.es, so mirrors are probed):
 *   - icdar2015 : ICDAR 2015 Incidental Scene Text (Challenge 4, 1000 train images)
 *                 GT format per line: "x1,y1,x2,y2,x3,y3,x4,y4,word"  ("###" = don't-care)
 *   - icdar2013 : ICDAR 2013 Focused Scene Text (Challenge 2, train split)
 *                 GT format per line: "x1 y1 x2 y2 word" (axis-aligned) or bare word-per-line
 *   - synth     : local synthetic dataset from gen_synth.py (fallback, always available)
 *
 * Usage:
 *   node datasets.js --list
 *   node datasets.js --download icdar2015 [--mirror URL] [--mirror URL2 ...] [--no-net]
 *   node datasets.js --load icdar2015 [--limit 5]
 *   node datasets.js --synth [--n 200]
 *
 * Mirror URLs can also be supplied via env DSH_BENCH_MIRRORS="url1|url2".
 * Every mirror is probed with HEAD; failures are skipped gracefully.
 */

const fs = require('fs');
const path = require('path');
const cp = require('child_process');
const U = require('./lib/util');

const IMG_EXT = /\.(jpg|jpeg|png|gif|bmp|tif|tiff)$/i;

// ---------------------------------------------------------------- registry

const REGISTRY = {
  icdar2015: {
    label: 'ICDAR 2015 Incidental Scene Text (Challenge 4, train)',
    dir: path.join(U.DATA_DIR, 'icdar2015'),
    note: 'Official site rr.cvc.uab.es requires registration; public mirrors probed below.',
    artifacts: {
      images: { file: 'ch4_training_images.zip', patterns: /ch4_training_images/i, minBytes: 10 * 1024 * 1024 },
      gt: { file: 'ch4_training_localization_transcription_gt.zip', patterns: /ch4_training_localization_transcription_gt|ch4_training_localization/i, minBytes: 1024 * 1024 },
    },
    mirrors: [
      // kind 'direct'  : URL to an archive (probed with HEAD; honoured when 2xx + size/suffix check)
      // kind 'hf-api'  : Hugging Face dataset id; file listing fetched to find archive URLs
      { kind: 'direct', for: 'images', url: 'https://rr.cvc.uab.es/wp-content/uploads/2021/04/ch4_training_images.zip', official: true },
      { kind: 'direct', for: 'gt', url: 'https://rr.cvc.uab.es/wp-content/uploads/2021/04/ch4_training_localization_transcription_gt.zip', official: true },
      { kind: 'hf-api', for: '*', dataset: 'MiXaiLL76/ICDAR2015_OCR' },
      // Community listing (reference only — probed and skipped if it is not an archive):
      { kind: 'direct', for: 'images', url: 'https://raw.githubusercontent.com/HCIILAB/Scene-Text-Detection-and-Recognition-Datasets/master/README.md', listingOnly: true },
    ],
  },
  icdar2013: {
    label: 'ICDAR 2013 Focused Scene Text (Challenge 2, train)',
    dir: path.join(U.DATA_DIR, 'icdar2013'),
    note: 'Official site rr.cvc.uab.es requires registration; public mirrors probed below.',
    artifacts: {
      images: { file: 'Challenge2_Training_Task12_Images.zip', patterns: /Challenge2_Training_Task12_Images|icdar2013.*images/i, minBytes: 10 * 1024 * 1024 },
      gt: { file: 'Challenge2_Training_Task1_GT.zip', patterns: /Challenge2_Training_Task1_GT|icdar2013.*gt/i, minBytes: 64 * 1024 },
    },
    mirrors: [
      { kind: 'direct', for: 'images', url: 'https://rr.cvc.uab.es/wp-content/uploads/2021/04/Challenge2_Training_Task12_Images.zip', official: true },
      { kind: 'direct', for: 'gt', url: 'https://rr.cvc.uab.es/wp-content/uploads/2021/04/Challenge2_Training_Task1_GT.zip', official: true },
    ],
  },
  synth: {
    label: 'Synthetic CJK+EN scene text (gen_synth.py; fallback & regression set)',
    dir: path.join(U.DATA_DIR, 'synth'),
    local: true,
  },
};

// ---------------------------------------------------------------- gt parsing

/**
 * Parse one ICDAR GT line. Supports:
 *   - "x1,y1,x2,y2,x3,y3,x4,y4,word"  (ICDAR2015; word may itself contain commas)
 *   - "x1 y1 x2 y2 word"              (ICDAR2013 axis-aligned)
 *   - "word"                          (word-per-line, geometry unknown)
 * Returns {text, box:[[x,y]x4]|null, ignore:boolean} or null for blank lines.
 */
function parseGtLine(line) {
  const raw = String(line).replace(/^\uFEFF/, '').trim();
  if (!raw) return null;
  if (raw.indexOf(',') >= 0) {
    const parts = raw.split(',');
    if (parts.length >= 9 && parts.slice(0, 8).every(p => Number.isFinite(parseFloat(p)))) {
      const c = parts.slice(0, 8).map(parseFloat);
      const text = parts.slice(8).join(',').trim();
      const box = [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]], [c[6], c[7]]];
      return { text, box, ignore: text === '###' };
    }
  }
  const ws = raw.split(/\s+/);
  if (ws.length >= 5 && ws.slice(0, 4).every(p => Number.isFinite(parseFloat(p)))) {
    const [x1, y1, x2, y2] = ws.slice(0, 4).map(parseFloat);
    const text = ws.slice(4).join(' ').trim();
    const box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
    return { text, box, ignore: text === '###' };
  }
  const text = raw;
  return { text, box: null, ignore: text === '###' };
}

/** Parse a whole GT file into gtWords entries. */
function parseGtFile(gtPath) {
  const lines = fs.readFileSync(gtPath, 'utf8').split(/\r?\n/);
  const out = [];
  for (const line of lines) {
    const e = parseGtLine(line);
    if (e) out.push(e);
  }
  return out;
}

// ---------------------------------------------------------------- discovery / loading

/** Recursively collect files under root (bounded, symlink-safe). */
function walkFiles(root, fn, limit = 200000) {
  let n = 0;
  const stack = [root];
  while (stack.length && n < limit) {
    const d = stack.pop();
    let ents;
    try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch (e) { continue; }
    for (const e of ents) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) stack.push(p);
      else if (e.isFile()) { n++; if (fn(p) === false) return; }
    }
  }
}

/** Pair gt txt files with images by stem, normalizing ICDAR prefixes:
 *  gt_123.txt <-> img_123.jpg (2015) or 123.jpg (2013). */
function pairFiles(dsDir) {
  const images = new Map();   // normalized stem -> path
  const gts = new Map();      // normalized stem -> path
  const normStem = s => s.replace(/^img_/i, '').replace(/^gt_/i, '');
  walkFiles(dsDir, p => {
    const b = path.basename(p);
    if (IMG_EXT.test(b)) {
      const stem = normStem(b.replace(IMG_EXT, ''));
      if (!images.has(stem)) images.set(stem, p);
    } else if (/\.txt$/i.test(b)) {
      const stem = normStem(b.replace(/\.txt$/i, ''));
      if (!gts.has(stem)) gts.set(stem, p);
    }
  });
  const pairs = [];
  for (const [stem, gtPath] of [...gts.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const imgPath = images.get(stem);
    if (imgPath) pairs.push({ stem, imgPath, gtPath });
  }
  return { pairs, nImages: images.size, nGt: gts.size };
}

/**
 * Load a dataset into the common sample format:
 *   [{ imgPath, gtWords: [{ text, box:[[x,y]x4]|null, ignore:boolean }] }]
 * @param {string} name     dataset key in REGISTRY
 * @param {object} opts     { limit?: number }
 */
function loadDataset(name, opts = {}) {
  const ds = REGISTRY[name];
  if (!ds) throw new Error('unknown dataset: ' + name);
  if (ds.local) return loadSynth(ds, opts);
  const { pairs } = pairFiles(ds.dir);
  const limit = U.toInt(opts && opts.limit, 0);
  const sel = limit > 0 ? pairs.slice(0, limit) : pairs;
  return sel.map(p => ({
    imgPath: p.imgPath,
    imgName: path.basename(p.imgPath),
    dataset: name,
    gtWords: parseGtFile(p.gtPath),
  }));
}

/** Synthetic loader: data/synth/gt.json + img/*.jpg */
function loadSynth(ds, opts = {}) {
  const gtPath = path.join(ds.dir, 'gt.json');
  if (!U.exists(gtPath)) return [];
  const manifest = U.readJson(gtPath, []);
  const limit = U.toInt(opts && opts.limit, 0);
  const sel = limit > 0 ? manifest.slice(0, limit) : manifest;
  const out = [];
  for (const item of sel) {
    const imgPath = path.join(ds.dir, item.img || item.path || '');
    if (!U.exists(imgPath)) continue;
    const gtWords = (item.words || []).map(w => ({
      text: String(w.text != null ? w.text : w),
      box: Array.isArray(w.box) && w.box.length === 4 && Array.isArray(w.box[0]) ? w.box : null,
      ignore: String(w.text) === '###',
    }));
    out.push({ imgPath, imgName: path.basename(imgPath), dataset: 'synth', gtWords });
  }
  return out;
}

/** Is the dataset present on disk (>=1 image with gt)? */
function isPresent(name) {
  const ds = REGISTRY[name];
  if (!ds) return false;
  if (ds.local) return loadSynth(ds).length > 0;
  const { pairs } = pairFiles(ds.dir);
  return pairs.length > 0;
}

/** Registry listing with status (does not touch the network). */
function listDatasets() {
  return Object.keys(REGISTRY).map(name => {
    const ds = REGISTRY[name];
    let samples = 0;
    try { samples = loadDataset(name).length; } catch (e) { /* absent */ }
    return {
      name,
      label: ds.label,
      present: samples > 0,
      images: samples,
      dir: U.rel(ds.dir),
      note: ds.note || (ds.local ? 'generated by gen_synth.py' : ''),
    };
  });
}

// ---------------------------------------------------------------- download

function extraMirrors() {
  const out = [];
  const push = s => { if (s) (String(s).split(/[|;\s]+/)).forEach(x => x && out.push(x)); };
  push(process.env.DSH_BENCH_MIRROR);
  push(process.env.DSH_BENCH_MIRRORS);
  return out;
}

/** Expand an 'hf-api' mirror into direct archive URLs by listing dataset files. */
async function expandHfDataset(entry, artifact) {
  const apiUrl = 'https://huggingface.co/api/datasets/' + entry.dataset;
  const tmp = path.join(U.DATA_DIR, '.hf-' + entry.dataset.replace(/[\\/]/g, '_') + '.json');
  const r = await U.downloadTo(apiUrl, tmp, 20000);
  if (!r.ok) return null;
  let info;
  try { info = JSON.parse(fs.readFileSync(tmp, 'utf8')); } catch (e) { return null; }
  finally { try { fs.unlinkSync(tmp); } catch (e) { /* ignorable */ } }
  const sib = (info && info.siblings) || [];
  const cands = sib
    .map(s => ({ name: s.rfilename, url: 'https://huggingface.co/datasets/' + entry.dataset + '/resolve/main/' + s.rfilename }))
    .filter(s => /\.(zip|tar(\.gz)?)$/i.test(s.name));
  if (!cands.length) return null;
  // Prefer names matching the artifact pattern, else the largest-looking archive.
  const re = artifact.patterns;
  const hit = cands.find(c => re.test(c.name)) || cands[cands.length - 1];
  return hit.url;
}

/**
 * Try to download one dataset over the network.
 * @returns {Promise<{ok:boolean, downloaded:string[], skipped:string[], note:string}>}
 */
async function downloadDataset(name, opts = {}) {
  const ds = REGISTRY[name];
  if (!ds) throw new Error('unknown dataset: ' + name);
  if (ds.local) return { ok: true, downloaded: [], skipped: [], note: 'local dataset, nothing to download' };
  if (opts.noNet) return { ok: false, downloaded: [], skipped: ['network disabled (--no-net)'], note: 'network disabled' };

  const userMirrors = [].concat(opts.mirrors || [], extraMirrors());
  const result = { ok: false, downloaded: [], skipped: [], note: '' };
  U.ensureDir(ds.dir);

  for (const artKey of Object.keys(ds.artifacts)) {
    const art = ds.artifacts[artKey];
    const dest = path.join(ds.dir, art.file);
    if (U.exists(dest) && fs.statSync(dest).size > art.minBytes / 2) {
      result.downloaded.push(art.file + ' (cached)');
      continue;
    }
    // candidate list: user mirrors first, then registry mirrors for this artifact (or '*')
    const entries = []
      .concat(userMirrors.map(u => ({ kind: 'direct', for: artKey, url: u })))
      .concat(ds.mirrors.filter(m => m.for === artKey || m.for === '*'));

    let done = false;
    for (const e of entries) {
      let url = e.url;
      if (e.kind === 'hf-api') {
        url = await expandHfDataset(e, art);
        if (!url) { result.skipped.push(`hf:${e.dataset} (no archive found)`); continue; }
      }
      const isArchive = /\.(zip|tar(\.gz)?|tgz)$/i.test(url || '');
      U.log('net', `probe  ${url}`);
      const probe = await U.headProbe(url, 15000);
      if (!probe.ok) {
        const why = probe.status ? `HTTP ${probe.status}` : (probe.error || 'unreachable');
        const tag = e.official ? ' (official, registration-gated)' : e.listingOnly ? ' (listing page, not an archive)' : '';
        result.skipped.push(`${url} — ${why}${tag}`);
        continue;
      }
      if (e.listingOnly || !isArchive) {
        result.skipped.push(`${url} — reachable but not a dataset archive`);
        continue;
      }
      if (probe.contentLength && probe.contentLength < art.minBytes / 4) {
        result.skipped.push(`${url} — too small (${(probe.contentLength / 1048576).toFixed(1)} MB)`);
        continue;
      }
      U.log('net', `download ${url}`);
      const dl = await U.downloadTo(url, dest, 300000);
      if (!dl.ok || (dl.bytes || 0) < art.minBytes / 4) {
        result.skipped.push(`${url} — download failed (${dl.error || dl.bytes + ' bytes'})`);
        try { fs.unlinkSync(dest); } catch (err) { /* ignorable */ }
        continue;
      }
      U.log('net', `unpack  ${art.file} -> ${U.rel(ds.dir)}`);
      if (!U.unpack(dest, ds.dir)) {
        result.skipped.push(`${art.file} — unpack failed`);
        continue;
      }
      result.downloaded.push(art.file);
      done = true;
      break;
    }
    if (!done && !result.downloaded.some(d => d.startsWith(art.file))) {
      result.note = `could not obtain ${art.key || artKey} from any mirror`;
    }
  }

  result.ok = isPresent(name);
  if (!result.ok && !result.note) result.note = 'archives fetched but no image+gt pairs found after unpack';
  return result;
}

// ---------------------------------------------------------------- synth gen

function ensureSynth(opts = {}) {
  const n = U.toInt(opts.n, 200);
  const py = U.findPython();
  if (!py) { U.fail('synth', 'python not found; cannot run gen_synth.py'); return false; }
  const script = path.join(U.BENCHMARK_DIR, 'gen_synth.py');
  const outDir = path.join(U.DATA_DIR, 'synth');
  U.log('synth', `python gen_synth.py --n ${n} --out ${U.rel(outDir)}`);
  const r = cp.spawnSync(py, [script, '--n', String(n), '--out', outDir], { encoding: 'utf8', timeout: 600000, cwd: U.BENCHMARK_DIR });
  if (r.stdout) process.stdout.write(r.stdout);
  if (r.stderr) process.stderr.write(r.stderr);
  if (r.status !== 0) { U.fail('synth', `gen_synth.py exited ${r.status}`); return false; }
  return isPresent('synth');
}

// ---------------------------------------------------------------- CLI

async function main() {
  const args = U.parseArgs(process.argv.slice(2));
  const cmd = args.positional[0] || (args.flags.list ? 'list' : 'help');

  if (cmd === 'help') {
    process.stdout.write(__doc__());
    return;
  }
  if (cmd === 'list') {
    const rows = listDatasets();
    console.log(U.markdownTable(rows, ['name', 'label', 'present', 'images', 'dir'], ['dataset', 'description', 'present', 'samples', 'dir']));
    return;
  }
  if (cmd === 'synth') {
    const ok = ensureSynth({ n: args.flags.n });
    process.exit(ok ? 0 : 1);
  }
  if (cmd === 'download') {
    const name = args.positional[1] || 'icdar2015';
    const mirrors = (args.flags.mirror || '').split(',').map(s => s.trim()).filter(Boolean);
    const res = await downloadDataset(name, { mirrors, noNet: !!args.flags.noNet });
    for (const d of res.downloaded) U.log('net', '  ✓ ' + d);
    for (const s of res.skipped) U.warn('net', '  ✗ ' + s);
    U[res.ok ? 'log' : 'warn']('net', `${name}: ${res.ok ? 'READY' : 'unavailable'} — ${res.note || (res.downloaded.length + ' artifact(s) in place')}`);
    process.exit(res.ok ? 0 : 2); // 2 = dataset unavailable (not an error for run_all)
  }
  if (cmd === 'load') {
    const name = args.positional[1] || 'icdar2015';
    const samples = loadDataset(name, { limit: U.toInt(args.flags.limit, 3) });
    console.log(JSON.stringify(samples, null, 2));
    return;
  }
  U.fail('ds', 'unknown command: ' + cmd);
  process.stdout.write(__doc__());
  process.exit(1);
}

function __doc__() {
  return [
    'Usage: node datasets.js <command> [options]',
    '  --list                          list datasets + on-disk status',
    '  --download <name> [--mirror URL] [--no-net]   probe mirrors and download',
    '  --load <name> [--limit N]       print loaded samples (JSON)',
    '  --synth [--n 200]               generate the synthetic dataset via gen_synth.py',
    '',
    'Env: DSH_BENCH_MIRRORS="url1|url2"  extra mirrors tried first',
  ].join('\n');
}

if (require.main === module) {
  main().catch(e => { U.fail('ds', String(e && e.stack || e)); process.exit(1); });
}

module.exports = { REGISTRY, listDatasets, loadDataset, downloadDataset, isPresent, ensureSynth, parseGtLine, parseGtFile };
