'use strict';
/**
 * benchmark/test/stub_cli.js — CONTRACT STUB for self-testing the benchmark suite.
 *
 * Not part of the shipped pipeline. It mimics the real CLI contract:
 *     node stub_cli.js <image> --json
 * and emits ONE JSON line exactly like dist/ocr_infer/src/cli.js should:
 *   {"ok":true,"version":"1.0.0","text":"...","results":[{"text","score","box"}],
 *    "timing":{...},"models":{"detBytes","recBytes"}}
 *
 * To produce plausible predictions it reads a GT sidecar manifest
 * (env DSH_STUB_GT=<path to gt.json>, same schema as gen_synth.py output):
 *   - DSH_STUB_MODE=perfect  (default): emit every GT box/text verbatim
 *   - DSH_STUB_MODE=degrade : jitter boxes (~0.95 IoU), drop 10% of words,
 *                             corrupt a few chars → realistic non-saturating metrics
 *
 * Use it to verify the full bench wiring before the real build exists:
 *   set DSH_STUB_GT=D:\Code\dsh-ocr-micro\benchmark\data\synth\gt.json
 *   node benchmark/test/stub_cli.js benchmark/data/synth/img/img_0001.jpg --json
 */

const fs = require('fs');
const path = require('path');

function hashSeed(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) / 4294967296; }

function jitterBox(box, imgW, imgH) {
  // shrink slightly around the box center -> IoU vs original stays ~0.9+
  const xs = box.map(p => p[0]), ys = box.map(p => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const r = 0.94 + hashSeed(String(cx) + cy) * 0.04;
  return box.map(p => {
    const nx = cx + (p[0] - cx) * r;
    const ny = cy + (p[1] - cy) * r;
    return [Number(nx.toFixed(1)), Number(ny.toFixed(1))];
  });
}

function corruptText(text, seed) {
  const chars = [...text];
  const n = Math.max(1, Math.floor(chars.length * 0.15));
  for (let i = 0; i < n && chars.length; i++) {
    const idx = Math.floor(hashSeed(seed + i) * chars.length);
    chars[idx] = hashSeed(seed + text + i) > 0.5 ? '?' : 'X';
  }
  return chars.join('');
}

function main() {
  const args = process.argv.slice(2);
  const img = args.find(a => !a.startsWith('--'));
  const mode = process.env.DSH_STUB_MODE || 'perfect';
  const gtPath = process.env.DSH_STUB_GT;
  const t0 = process.hrtime.bigint();

  if (!img) { console.error('stub_cli: missing <image> arg'); process.exit(2); }
  let results = [];
  if (gtPath && fs.existsSync(gtPath)) {
    const manifest = JSON.parse(fs.readFileSync(gtPath, 'utf8'));
    const base = path.basename(img);
    const item = manifest.find(m => m.img === 'img/' + base || path.basename(String(m.img)) === base);
    if (item) {
      results = (item.words || []).map((w, i) => {
        const seed = hashSeed(base + ':' + i);
        if (mode === 'degrade' && seed < 0.12) return null; // dropped line → FN
        const text = mode === 'degrade' && seed > 0.50
          ? corruptText(String(w.text), seed)
          : String(w.text);
        const box = mode === 'degrade' ? jitterBox(w.box, item.w, item.h) : w.box;
        return { text, score: Number((0.88 + seed * 0.11).toFixed(2)), box };
      }).filter(Boolean);
    }
  } else if (!gtPath) {
    console.error('stub_cli: set DSH_STUB_GT=<gt.json> for realistic predictions; falling back to a dummy row');
  }

  const wallMs = Number(process.hrtime.bigint() - t0) / 1e6;
  const pre = 1 + wallMs * 0.15;
  const det = 1 + wallMs * 0.45;
  const rec = 1 + wallMs * 0.30 + (results.length * 1.5);
  const post = 0.2 + wallMs * 0.05;
  const out = {
    ok: true,
    version: '1.0.0-stub',
    text: results.map(r => r.text).join('\n'),
    results,
    timing: {
      preprocessMs: Number(pre.toFixed(1)),
      detMs: Number(det.toFixed(1)),
      recMs: Number(rec.toFixed(1)),
      postMs: Number(post.toFixed(2)),
      totalMs: Number((pre + det + rec + post).toFixed(1)),
    },
    models: {
      detBytes: parseInt(process.env.DSH_STUB_DET_BYTES || '2700000', 10),
      recBytes: parseInt(process.env.DSH_STUB_REC_BYTES || '2600000', 10),
    },
  };
  process.stdout.write(JSON.stringify(out) + '\n');
}

main();
