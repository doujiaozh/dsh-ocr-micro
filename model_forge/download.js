#!/usr/bin/env node
'use strict';
/**
 * model_forge/download.js — 模型获取 (Model Acquisition)
 * --------------------------------------------------------
 * Acquires world-class lightweight CJK+EN OCR weights — **PaddleOCR PP-OCRv6 tiny**
 * (2025 generation, 1.5M-param tier; det Hmean 80.6 / rec acc 73.5) — plus the
 * matching dictionary, and stages them in <repo>/models/ for the forge steps.
 *
 * Sources (tried in order, per artifact; any one success is enough):
 *   1. GitHub release of GreatV/oar-ocr v0.7.0 (release-assets redirect handled)
 *   2. ghproxy.net / gh-proxy.com mirrors of the same release assets
 *   3. ModelScope model repo greatv/oar-ocr (resolve/master)
 *   4. bcebos official PP-OCRv6 ONNX bundles (paddle-model-ecology ...#onnx_infer.tar,
 *      extracted with the pure-JS `tar` package)
 *   5. (last resort, needs conversion) bcebos PP-OCRv6 Paddle inference tars ->
 *      model_forge/paddle2onnx_convert.py
 *
 * Targets:
 *   models/det_fp32.onnx      <- pp-ocrv6_tiny_det.onnx   (fp32, dynamic HxW)
 *   models/rec_fp32.onnx      <- pp-ocrv6_tiny_rec.onnx   (fp32, dynamic batch/seq/width)
 *   models/ppocrv6_tiny_dict.txt <- ppocrv6_tiny_dict.txt (6904 characters)
 *
 * Pure Node >= 14: only https / fs / crypto built-ins, no fetch(), no native deps.
 *
 * Usage: node model_forge/download.js [--force] [--paddle]
 *   --force   re-download even if the target file already looks valid
 *   --paddle  skip ONNX sources and go straight to the PaddlePaddle tar last resort
 */

const https = require('https');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// ----------------------------------------------------------------------------
const FORGE_DIR = __dirname;
const REPO_ROOT = path.resolve(FORGE_DIR, '..');
const MODELS_DIR = path.join(REPO_ROOT, 'models');
const CACHE_DIR = path.join(FORGE_DIR, 'cache');

const OAR_OCR_RELEASE = 'https://github.com/GreatV/oar-ocr/releases/download/v0.7.0';
const GHPROXY_MIRRORS = ['https://ghproxy.net/', 'https://gh-proxy.com/'];
const MODELSCOPE = 'https://www.modelscope.cn/models/greatv/oar-ocr/resolve/master';

const ASSETS = {
  det: {
    file: 'pp-ocrv6_tiny_det.onnx',
    dest: 'det_fp32.onnx',
    expectBytes: 1_780_590,
    minBytes: 512 * 1024,
  },
  rec: {
    file: 'pp-ocrv6_tiny_rec.onnx',
    dest: 'rec_fp32.onnx',
    expectBytes: 4_462_639,
    minBytes: 1024 * 1024,
  },
  dict: {
    file: 'ppocrv6_tiny_dict.txt',
    dest: 'ppocrv6_tiny_dict.txt',
    expectBytes: 27_156,
    minBytes: 5 * 1024,
  },
};

// bcebos official ONNX bundles (fallback for det/rec only)
const BCEBOS_ONNX_TARS = {
  det: 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_det_onnx_infer.tar',
  rec: 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_rec_onnx_infer.tar',
};

// bcebos Paddle inference tars (absolute last resort; needs paddle2onnx_convert.py)
const BCEBOS_PADDLE_TARS = {
  det: 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_det_infer.tar',
  rec: 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_rec_infer.tar',
};

const HTTP_TIMEOUT_MS = 60 * 1000;
const MAX_REDIRECTS = 5;
const ATTEMPTS = 2;

const argv = process.argv.slice(2);
const FORCE = argv.includes('--force');
const FORCE_PADDLE = argv.includes('--paddle');

// ----------------------------------------------------------------------------
// Tiny helpers
// ----------------------------------------------------------------------------
function log(tag, msg) {
  const t = new Date().toISOString().slice(11, 23);
  console.log(`[${t}] ${tag.padEnd(11)} ${msg}`);
}

function fmtBytes(n) {
  if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(2) + ' MB';
  if (n >= 1024) return (n / 1024).toFixed(1) + ' KB';
  return n + ' B';
}

function rmrf(p) {
  try {
    fs.rmSync(p, { recursive: true, force: true });
  } catch (e) {
    try {
      fs.unlinkSync(p);
    } catch (_) {
      /* already gone */
    }
  }
}

function ensureDir(d) {
  fs.mkdirSync(d, { recursive: true });
}

function sha256File(p) {
  return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
}

/**
 * Download `url` to `dest` with redirect following, timeout, retry and sanity
 * checks (HTTP 200, Content-Length match, minimum byte floor, non-HTML body).
 * Returns { bytes, sha256 }.
 */
function download(url, dest, minBytes) {
  return new Promise((resolve, reject) => {
    const attempt = (n) => {
      let redirects = 0;
      const get = (u) => {
        let parsed;
        try {
          parsed = new URL(u);
        } catch (e) {
          return reject(new Error(`bad url: ${u}`));
        }
        const hash = crypto.createHash('sha256');
        let received = 0;
        const req = https.get(
          {
            protocol: parsed.protocol,
            hostname: parsed.hostname,
            port: parsed.port || (parsed.protocol === 'https:' ? 443 : 80),
            path: parsed.pathname + parsed.search,
            method: 'GET',
            timeout: HTTP_TIMEOUT_MS,
            headers: { 'User-Agent': 'dsh-ocr-micro/model-forge' }
          },
          (res) => {
            if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
              res.resume();
              if (++redirects > MAX_REDIRECTS) {
                return reject(new Error(`too many redirects for ${url}`));
              }
              const next = new URL(res.headers.location, parsed).toString();
              log('redirect', `${res.statusCode} -> ${next.slice(0, 96)}`);
              return get(next);
            }
            if (res.statusCode !== 200) {
              res.resume();
              const err = new Error(`HTTP ${res.statusCode} for ${u}`);
              if (n < ATTEMPTS) {
                log('retry', `${err.message} (attempt ${n + 1}/${ATTEMPTS})`);
                return setTimeout(() => attempt(n + 1), 500);
              }
              return reject(err);
            }
            const tmp = dest + '.part';
            const out = fs.createWriteStream(tmp);
            const expected = res.headers['content-length'] ? parseInt(res.headers['content-length'], 10) : null;
            res.on('data', (chunk) => {
              received += chunk.length;
              hash.update(chunk);
            });
            res.pipe(out);
            out.on('finish', () => {
              out.close((closeErr) => {
                if (closeErr) return reject(closeErr);
                if (expected !== null && received !== expected) {
                  rmrf(tmp);
                  const err = new Error(`size mismatch for ${u}: got ${received}, expected ${expected}`);
                  if (n < ATTEMPTS) {
                    log('retry', `${err.message} (attempt ${n + 1}/${ATTEMPTS})`);
                    return setTimeout(() => attempt(n + 1), 500);
                  }
                  return reject(err);
                }
                if (received < minBytes) {
                  rmrf(tmp);
                  return reject(
                    new Error(`file too small for ${u}: ${fmtBytes(received)} < ${fmtBytes(minBytes)} floor (corrupt download or wrong mirror)`)
                  );
                }
                fs.renameSync(tmp, dest);
                resolve({ bytes: received, sha256: hash.digest('hex') });
              });
            });
            out.on('error', (e) => {
              rmrf(tmp);
              if (n < ATTEMPTS) return setTimeout(() => attempt(n + 1), 500);
              reject(e);
            });
          }
        );
        req.on('timeout', () => {
          req.destroy(new Error(`timeout after ${HTTP_TIMEOUT_MS}ms for ${u}`));
        });
        req.on('error', (e) => {
          if (n < ATTEMPTS) {
            log('retry', `${e.message} (attempt ${n + 1}/${ATTEMPTS})`);
            return setTimeout(() => attempt(n + 1), 500);
          }
          reject(e);
        });
      };
      get(url);
    };
    attempt(0);
  });
}

function sniffNotText(dest) {
  // guard: a 200-served HTML error page is not a valid binary/text asset
  const fd = fs.openSync(dest, 'r');
  const buf = Buffer.alloc(32);
  const n = fs.readSync(fd, buf, 0, 32, 0);
  fs.closeSync(fd);
  return !/^\s*<(!DOCTYPE|html|HTML)/.test(buf.slice(0, n).toString('ascii'));
}

function validFile(p, minBytes) {
  try {
    const st = fs.statSync(p);
    return st.isFile() && st.size >= minBytes;
  } catch (e) {
    return false;
  }
}

// Ordered candidate URLs for an oar-ocr release asset.
// NOTE: github.com:443 is unreachable from this box (SSL failure), so the
// China-hosted ModelScope mirror is tried FIRST; github release links stay
// as additional fallbacks (via proxies that do reach it).
function candidateUrls(assetFile) {
  const urls = [];
  urls.push(`${MODELSCOPE}/${assetFile}`);
  for (const proxy of GHPROXY_MIRRORS) {
    urls.push(`${proxy}${OAR_OCR_RELEASE}/${assetFile}`);
  }
  urls.push(`${OAR_OCR_RELEASE}/${assetFile}`);
  return urls;
}

// ----------------------------------------------------------------------------
// tar fallbacks
// ----------------------------------------------------------------------------
function loadTarLib() {
  try {
    return require('tar'); // pure-JS dependency of this workspace
  } catch (e) {
    console.error('');
    console.error('FATAL: the `tar` npm package is not installed (needed to extract bcebos tars).');
    console.error('       run `npm install` in the repo root first (npm workspaces installs it for model_forge).');
    process.exit(3);
  }
}

async function extractBcebosOnnxBundle(kind) {
  // returns path of the extracted *.onnx
  const tarLib = loadTarLib();
  ensureDir(CACHE_DIR);
  const tarPath = path.join(CACHE_DIR, `${kind}_onnx_bundle.tar`);
  log(kind, `bcebos ONNX bundle ${BCEBOS_ONNX_TARS[kind]}`);
  const info = await download(BCEBOS_ONNX_TARS[kind], tarPath, ASSETS[kind].minBytes);
  log(kind, `got bundle ${fmtBytes(info.bytes)} sha256=${info.sha256.slice(0, 12)}...`);
  const extractDir = path.join(CACHE_DIR, `${kind}_onnx`);
  rmrf(extractDir);
  ensureDir(extractDir);
  await tarLib.x({ file: tarPath, cwd: extractDir });
  const onnxFiles = [];
  const walk = (d) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, entry.name);
      if (entry.isDirectory()) walk(p);
      else if (/\.onnx$/i.test(entry.name)) onnxFiles.push(p);
    }
  };
  walk(extractDir);
  if (!onnxFiles.length) throw new Error(`no .onnx found inside ${path.basename(BCEBOS_ONNX_TARS[kind])}`);
  onnxFiles.sort((a, b) => fs.statSync(b).size - fs.statSync(a).size); // biggest first
  return onnxFiles[0];
}

async function downloadBcebosPaddleTar(kind) {
  // returns { model, params } extracted from the Paddle inference tar
  const tarLib = loadTarLib();
  ensureDir(CACHE_DIR);
  const tarPath = path.join(CACHE_DIR, `${kind}_paddle.tar`);
  log(kind, `bcebos Paddle tar ${BCEBOS_PADDLE_TARS[kind]}`);
  const info = await download(BCEBOS_PADDLE_TARS[kind], tarPath, 256 * 1024);
  log(kind, `got paddle tar ${fmtBytes(info.bytes)} sha256=${info.sha256.slice(0, 12)}...`);
  const extractDir = path.join(CACHE_DIR, `${kind}_paddle`);
  rmrf(extractDir);
  ensureDir(extractDir);
  await tarLib.x({ file: tarPath, cwd: extractDir });
  const found = {};
  const walk = (d) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, entry.name);
      if (entry.isDirectory()) walk(p);
      else if (entry.name === 'inference.pdmodel') found.model = p;
      else if (entry.name === 'inference.pdiparams') found.params = p;
    }
  };
  walk(extractDir);
  if (!found.model || !found.params) {
    throw new Error(`inference.pdmodel/pdiparams not found inside ${path.basename(BCEBOS_PADDLE_TARS[kind])}`);
  }
  return found;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
(async function main() {
  ensureDir(MODELS_DIR);
  const summary = []; // { file, bytes, sha256, note }

  for (const key of ['det', 'rec', 'dict']) {
    const asset = ASSETS[key];
    const dest = path.join(MODELS_DIR, asset.dest);

    if (!FORCE && validFile(dest, asset.minBytes)) {
      log(key, `${asset.dest} already present (${fmtBytes(fs.statSync(dest).size)}), keeping it`);
      summary.push({ file: path.relative(REPO_ROOT, dest), bytes: fs.statSync(dest).size, sha256: sha256File(dest), note: 'cached', expect: asset.expectBytes });
      continue;
    }

    if (FORCE_PADDLE && key !== 'dict') {
      // --paddle: straight to the Paddle inference tar last resort
      const files = await downloadBcebosPaddleTar(key);
      log(key, `staged paddle model: ${path.relative(CACHE_DIR, files.model)}`);
      summary.push({ file: path.relative(REPO_ROOT, files.model), bytes: fs.statSync(files.model).size, sha256: sha256File(files.model), note: 'paddle .pdmodel (needs convert)', expect: 0 });
      summary.push({ file: path.relative(REPO_ROOT, files.params), bytes: fs.statSync(files.params).size, sha256: sha256File(files.params), note: 'paddle .pdiparams', expect: 0 });
      continue;
    }

    // ---- ordered mirror chain ----
    let ok = false;
    const urls = candidateUrls(asset.file);
    for (let i = 0; i < urls.length; i++) {
      const url = urls[i];
      try {
        log(key, `mirror[${i}] ${url.slice(0, 100)}`);
        const info = await download(url, dest, asset.minBytes);
        if (key !== 'dict' && !sniffNotText(dest)) {
          rmrf(dest);
          throw new Error('mirror returned an HTML page, not a binary');
        }
        log(key, `saved ${fmtBytes(info.bytes)} sha256=${info.sha256.slice(0, 12)}...`);
        summary.push({ file: path.relative(REPO_ROOT, dest), bytes: info.bytes, sha256: info.sha256, note: url.includes('modelscope') ? 'modelscope' : 'oar-ocr mirror', expect: asset.expectBytes });
        ok = true;
        break;
      } catch (e) {
        log(key, `mirror FAILED: ${e.message}`);
        rmrf(dest);
      }
    }

    // ---- bcebos ONNX bundle fallback (det/rec only) ----
    if (!ok && key !== 'dict') {
      try {
        const onnxPath = await extractBcebosOnnxBundle(key);
        fs.copyFileSync(onnxPath, dest);
        const bytes = fs.statSync(dest).size;
        const sha = sha256File(dest);
        log(key, `extracted ONNX from bcebos bundle -> ${fmtBytes(bytes)} sha256=${sha.slice(0, 12)}...`);
        summary.push({ file: path.relative(REPO_ROOT, dest), bytes, sha256: sha, note: 'bcebos onnx bundle', expect: asset.expectBytes });
        ok = true;
      } catch (e) {
        log(key, `bcebos ONNX bundle FAILED: ${e.message}`);
      }
    }

    // ---- bcebos Paddle inference tar last resort (det/rec only) ----
    if (!ok && key !== 'dict') {
      const files = await downloadBcebosPaddleTar(key);
      log(key, `staged paddle model (last resort): ${path.relative(CACHE_DIR, files.model)}`);
      summary.push({ file: path.relative(REPO_ROOT, files.model), bytes: fs.statSync(files.model).size, sha256: sha256File(files.model), note: 'paddle .pdmodel (needs convert)', expect: 0 });
      summary.push({ file: path.relative(REPO_ROOT, files.params), bytes: fs.statSync(files.params).size, sha256: sha256File(files.params), note: 'paddle .pdiparams', expect: 0 });
      summary.push({ file: path.relative(REPO_ROOT, dest), bytes: 0, sha256: '', note: 'PENDING paddle2onnx_convert.py', expect: asset.expectBytes });
      continue;
    }

    if (!ok) {
      throw new Error(`could not acquire ${asset.file} from any source (github release, ghproxy mirrors, ModelScope)`);
    }
  }

  // ---------------- summary ----------------
  console.log('');
  console.log('════════════════════════════ DOWNLOAD SUMMARY ════════════════════════════');
  const nameWidth = Math.max(...summary.map((s) => s.file.length), 32);
  console.log(`${'artifact'.padEnd(nameWidth)}  ${'size'.padStart(10)}  ${'expected'.padStart(10)}  ${'sha256'.padStart(12)}  source`);
  for (const s of summary) {
    const expect = s.expect > 0 ? (s.bytes === s.expect ? 'match' : `${s.expect}`) : '-';
    console.log(
      `${s.file.padEnd(nameWidth)}  ${(s.bytes ? fmtBytes(s.bytes) : '-').padStart(10)}  ${expect.padStart(10)}  ${(s.sha256 || '').slice(0, 12)}  ${s.note}`
    );
  }
  const totalOnnx = ['det_fp32.onnx', 'rec_fp32.onnx']
    .map((n) => path.join(MODELS_DIR, n))
    .filter((p) => fs.existsSync(p) && fs.statSync(p).size > 0)
    .reduce((a, p) => a + fs.statSync(p).size, 0);
  console.log('');
  if (totalOnnx > 0) {
    console.log(`fp32 ONNX on disk: ${fmtBytes(totalOnnx)}  (budget for the final INT8 set is 5.5 MB)`);
    console.log('next: python model_forge/quantize.py   ->   node model_forge/verify_onnx.js');
  } else {
    console.log('Paddle inference models staged in model_forge/cache/.');
    console.log('next: python model_forge/paddle2onnx_convert.py   ->   python model_forge/quantize.py');
  }
  console.log('══════════════════════════════════════════════════════════════════════════');
})().catch((e) => {
  console.error('');
  console.error('DOWNLOAD FAILED:', e.message || e);
  process.exit(1);
});
