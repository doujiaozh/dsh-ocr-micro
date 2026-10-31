#!/usr/bin/env node
'use strict';
/**
 * model_forge/verify_onnx.js — final gate for the 5.5 MB model budget.
 *
 * **Engine**: onnxruntime-web (WASM) — the only ONNX Runtime that loads on
 * the Windows 7 target (onnxruntime-node's native DLL needs Win8+ API sets;
 * verified across versions 1.1-1.31). Pattern validated on Node 26 + Win7:
 * dynamic import + `ort.env.wasm.wasmPaths` as a file:// URL + numThreads=1.
 *
 * Loads models/det.onnx + models/rec.onnx, prints both sizes, runs a dummy
 * forward (det 1x3x640x640; rec 1x3x48x320 — adapts to whatever the session
 * declares), prints output names + dims, then asserts det + rec < 5.5 MB.
 * Exit 0 = pass, 1 = fail.
 *
 * Pure Node >= 14 (no fetch). Usage: node model_forge/verify_onnx.js
 */

const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..');
const MODELS_DIR = path.join(REPO_ROOT, 'models');
const BUDGET = 5_500_000; // bytes — same strict cap as quantize.py ("TOTAL < 5.5MB")

function mb(n) {
  return (n / (1024 * 1024)).toFixed(2) + ' MB';
}
function kb(n) {
  return (n / 1024).toFixed(1) + ' KB';
}

async function loadOrt() {
  let ort = null;
  let engine = null;
  // ---- primary: onnxruntime-web (WASM, the only ORT that loads on Win7) ----
  // Try CJS require first (picks the node build), then dynamic import (ESM build).
  try {
    ort = require('onnxruntime-web');
    engine = 'onnxruntime-web (node build, WASM)';
  } catch (e) {
    try {
      const mod = await import('onnxruntime-web');
      ort = mod && mod.default && mod.default.env ? mod.default : mod;
      engine = 'onnxruntime-web (esm build, WASM)';
    } catch (e2) {
      throw new Error(
        'onnxruntime-web unavailable: require failed (' + (e && e.message) + ') and import failed (' + (e2 && e2.message) + '). ' +
        'Run `npm install` in the repo root first.'
      );
    }
  }
  if (!ort || !ort.env || !ort.env.wasm) {
    throw new Error('onnxruntime-web loaded but ort.env.wasm is missing — unexpected package shape');
  }
  // point the wasm loader at the package's shipped binaries via file:// URL
  const pkgDir = findPackageDir('onnxruntime-web', __dirname);
  if (pkgDir) {
    const wasmDir = findWasmDir(pkgDir);
    if (wasmDir) {
      ort.env.wasm.wasmPaths = 'file:///' + wasmDir.replace(/\\/g, '/') + '/';
    }
  }
  ort.env.wasm.numThreads = 1; // Win7 safety: no threaded wasm workers
  return ort;
}

function findPackageDir(name, fromDir) {
  // classic node_modules lookup (exports-map independent, Node>=14 safe)
  let cur = fromDir;
  while (true) {
    const cand = path.join(cur, 'node_modules', name);
    if (fs.existsSync(path.join(cand, 'package.json'))) return cand;
    const parent = path.dirname(cur);
    if (parent === cur) return null;
    cur = parent;
  }
}

function findWasmDir(pkgDir) {
  // locate the directory holding the shipped .wasm binaries
  const candidates = [path.join(pkgDir, 'dist')];
  for (const d of candidates) {
    if (!fs.existsSync(d)) continue;
    if (fs.readdirSync(d).some((f) => /\.wasm$/i.test(f))) return d;
    // some versions nest wasm under dist/ort-wasm
    for (const sub of fs.readdirSync(d)) {
      const sd = path.join(d, sub);
      if (fs.statSync(sd).isDirectory() && fs.readdirSync(sd).some((f) => /\.wasm$/i.test(f))) return sd;
    }
  }
  return null;
}

function dummyDims(sessionInput, defaultHW) {
  // fill dynamic dims (0/-1/undefined) with spec defaults: batch 1, C 3
  const dims = [];
  for (let k = 0; k < sessionInput.dims.length; k++) {
    let d = sessionInput.dims[k];
    if (typeof d !== 'number' || d <= 0) {
      if (sessionInput.dims.length === 4) d = k === 0 ? 1 : k === 1 ? 3 : defaultHW[k - 2];
      else if (sessionInput.dims.length === 3) d = k === 0 ? 1 : defaultHW[k - 1];
      else d = 1;
    }
    dims.push(d);
  }
  return dims;
}

(async function main() {
  const ort = await loadOrt();
  console.log(`engine: ${await ortVersion(ort)}`);
  // dictionary presence (soft requirement of the rec model)
  const dictPath = path.join(MODELS_DIR, 'ppocrv6_tiny_dict.txt');
  if (fs.existsSync(dictPath)) {
    const lines = fs.readFileSync(dictPath, 'utf8').split(/\r?\n/).filter((l) => l.length > 0);
    console.log(`dictionary: ${path.relative(REPO_ROOT, dictPath)}  classes(lines)=${lines.length}`);
  } else {
    console.log('dictionary: MISSING (models/ppocrv6_tiny_dict.txt) — run node model_forge/download.js');
  }

  let total = 0;
  const rows = [];
  const specs = [
    { file: 'det.onnx', label: 'det', hw: [640, 640] },
    { file: 'rec.onnx', label: 'rec', hw: [48, 320] }
  ];

  for (const s of specs) {
    const p = path.join(MODELS_DIR, s.file);
    if (!fs.existsSync(p)) {
      console.error(`FATAL: missing ${path.relative(REPO_ROOT, p)} — run model_forge/quantize.py first`);
      process.exit(2);
    }
    const size = fs.statSync(p).size;
    total += size;
    console.log(`- ${s.file}  ${kb(size)} (${mb(size)})`);

    let session;
    try {
      session = await ort.InferenceSession.create(fs.readFileSync(p), {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
    } catch (e) {
      console.error(`FATAL: failed to load ${s.file}: ${e.message}`);
      rows.push({ file: s.file, size, ok: false, detail: 'load failed: ' + e.message });
      continue;
    }

    let inputs = {};
    try {
      for (const name of session.inputNames) {
        const si = session.inputMetadata ? session.inputMetadata[name] : null;
        const idims = si && si.dims ? si.dims : [1, 3, s.hw[0], s.hw[1]];
        const dims = dummyDims({ dims: idims }, s.hw);
        const n = dims.reduce((a, b) => a * b, 1);
        const arr = new Float32Array(n); // zeros: dummy forward, we only check shapes
        inputs[name] = new ort.Tensor('float32', arr, dims);
      }
    } catch (e) {
      console.error(`FATAL: failed to build inputs for ${s.file}: ${e.message}`);
      rows.push({ file: s.file, size, ok: false, detail: 'input build failed: ' + e.message });
      continue;
    }

    let detail;
    try {
      const outputs = await session.run(inputs);
      const names = session.outputNames;
      detail = names
        .map((name) => {
          const t = outputs[name];
          const dims = t && t.dims ? Array.from(t.dims) : [];
          return `${name} [${dims.join(' x ')}]`;
        })
        .join(', ');
      console.log(`    inputs : [${session.inputNames.join(', ')}]`);
      console.log(`    outputs: ${detail}`);
      rows.push({ file: s.file, size, ok: true, detail });
    } catch (e) {
      console.error(`FATAL: dummy forward failed for ${s.file}: ${e.message}`);
      rows.push({ file: s.file, size, ok: false, detail: 'run failed: ' + e.message });
    }
  }

  // ---------------- report ----------------
  console.log('');
  console.log('══════════════════════════ MODEL VERIFICATION ═══════════════════════════');
  for (const r of rows) {
    const mark = r.ok ? 'OK ' : 'FAIL';
    console.log(`- [${mark}] ${r.file}  ${kb(r.size)}`);
    console.log(`         ${r.detail}`);
  }
  const dictBytes = fs.existsSync(dictPath) ? fs.statSync(dictPath).size : 0;
  console.log('');
  console.log(`TOTAL (det.onnx + rec.onnx) = ${total} bytes = ${mb(total)}`);
  if (dictBytes) {
    console.log(`dictionary ppocrv6_tiny_dict.txt = ${kb(dictBytes)} (not counted against the model budget)`);
  }
  console.log(`BUDGET = ${BUDGET} bytes = ${mb(BUDGET)}`);
  console.log('');

  const allOk = rows.length === specs.length && rows.every((r) => r.ok);
  if (total >= BUDGET) {
    console.log(`FAIL: over budget by ${total - BUDGET} bytes (${mb(total - BUDGET)}).`);
    console.log('      advice: python model_forge/prune.py --ratio 0.<n>  then re-run quantize.py');
    process.exit(1);
  }
  if (!allOk) {
    console.log('FAIL: at least one model failed to load or run.');
    process.exit(1);
  }
  console.log(`PASS: under budget by ${BUDGET - total} bytes, all models load & run on WASM. ✔`);
  process.exit(0);
})().catch((e) => {
  console.error('VERIFY FAILED:', e && e.stack ? e.stack : e);
  process.exit(1);
});

async function ortVersion(ort) {
  try {
    const v = ort.env.versionInfo;
    return `onnxruntime-web ${v.web || v || '?'} (wasm)`;
  } catch (e) {
    try {
      const pkgDir = findPackageDir('onnxruntime-web', __dirname);
      if (pkgDir) {
        const pj = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8'));
        return `onnxruntime-web ${pj.version} (wasm)`;
      }
    } catch (_) {
      /* fall through */
    }
    return 'onnxruntime-web (version unavailable)';
  }
}
