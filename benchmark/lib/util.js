'use strict';
/**
 * Shared helpers for the @dsh-ocr/benchmark suite.
 * Pure CommonJS, zero npm dependencies: Node built-ins + optional python3 child_process only.
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const url = require('url');
const cp = require('child_process');

const BENCHMARK_DIR = path.resolve(__dirname, '..'); // benchmark/ (this file lives in benchmark/lib/)
const REPO_ROOT = path.resolve(BENCHMARK_DIR, '..');
const DATA_DIR = path.join(BENCHMARK_DIR, 'data');
const OUT_DIR = path.join(BENCHMARK_DIR, 'out');
const CLI_PATH = process.env.DSH_BENCH_CLI
  ? path.resolve(process.cwd(), process.env.DSH_BENCH_CLI)
  : path.join(REPO_ROOT, 'dist', 'ocr_infer', 'src', 'cli.js');
const MODEL_VERIFY_PATH = path.join(REPO_ROOT, 'model_forge', 'verify_onnx.js');

/** Model size budget the whole project is graded against. */
const BUDGET_MB = 5.5;

// ---------------------------------------------------------------- args

/** Very small argv parser: --flag, --key value, --key=value, "--" stops parsing. */
function parseArgs(argv) {
  const flags = {};
  const positional = [];
  let i = 0;
  while (i < argv.length) {
    const a = argv[i];
    if (a === '--') { positional.push(...argv.slice(i + 1)); break; }
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq > 0) { flags[a.slice(2, eq)] = a.slice(eq + 1); }
      else if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) {
        flags[a.slice(2)] = argv[++i];
      } else { flags[a.slice(2)] = 'true'; }
    } else positional.push(a);
    i++;
  }
  return { flags, positional };
}

function toInt(v, dflt) {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : dflt;
}

// ---------------------------------------------------------------- fs

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); return p; }

function readJson(p, dflt) {
  try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) { return dflt; }
}

function writeJson(p, obj) {
  ensureDir(path.dirname(p));
  fs.writeFileSync(p, JSON.stringify(obj, null, 2), 'utf8');
}

function exists(p) { try { fs.accessSync(p); return true; } catch (e) { return false; } }

/** Relativize a path for console/log output (nicer than absolute everywhere). */
function rel(p) {
  const abs = path.resolve(p);
  const fromCwd = path.relative(process.cwd(), abs);
  if (fromCwd && !fromCwd.startsWith('..')) return fromCwd;
  return path.relative(REPO_ROOT, abs);
}

// ---------------------------------------------------------------- geo

/** Quad [[x,y],[x,y],[x,y],[x,y]] or [x1,y1,x2,y2,...] flat -> AABB. */
function aabbFromQuad(quad) {
  const pts = Array.isArray(quad[0]) ? quad : [];
  let xs, ys;
  if (pts.length) {
    xs = pts.map(p => p[0]); ys = pts.map(p => p[1]);
  } else {
    // flat number array [x1,y1,x2,y2,x3,y3,x4,y4]
    xs = []; ys = [];
    for (let i = 0; i + 1 < quad.length; i += 2) { xs.push(quad[i]); ys.push(quad[i + 1]); }
  }
  const x1 = Math.min(...xs), x2 = Math.max(...xs);
  const y1 = Math.min(...ys), y2 = Math.max(...ys);
  return { x1, y1, x2, y2 };
}

/** IoU of two AABBs. Degenerate (zero-area) boxes get IoU 0. */
function iouAABB(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const uni = areaA + areaB - inter;
  return uni > 0 ? inter / uni : 0;
}

// ---------------------------------------------------------------- strings

/** Classic DP Levenshtein distance (case-sensitive). */
function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a) return b.length;
  if (!b) return a.length;
  const m = a.length, n = b.length;
  let prev = new Array(n + 1);
  let cur = new Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    cur[0] = i;
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= n; j++) {
      const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    const t = prev; prev = cur; cur = t;
  }
  return prev[n];
}

/**
 * Case-insensitive comparators (OCR ground truth in ICDAR is typically matched
 * case-insensitively; pass compareCase=true to enforce exact case).
 */
function normText(s, compareCase) {
  let t = String(s == null ? '' : s).replace(/\s+/g, ' ').trim();
  if (!compareCase) t = t.toLowerCase();
  return t;
}

// ---------------------------------------------------------------- stats

/** Nearest-rank quantile of a numeric array (q in [0,1]). Array copied, not mutated. */
function quantile(values, q) {
  const a = values.slice().sort((x, y) => x - y);
  if (!a.length) return NaN;
  const idx = Math.ceil(q * a.length) - 1;
  return a[Math.max(0, Math.min(a.length - 1, idx))];
}

function mean(values) {
  if (!values.length) return NaN;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

const fmtMs = v => (Number.isFinite(v) ? v.toFixed(1) : 'n/a');
const fmtPct = v => (Number.isFinite(v) ? (v * 100).toFixed(2) + '%' : 'n/a');
const fmtMB = v => (Number.isFinite(v) ? v.toFixed(2) + ' MB' : 'n/a');

// ---------------------------------------------------------------- python

let _python = undefined;
/** Find a usable python interpreter ('python3' preferred, then 'python'). Null if none. */
function findPython() {
  if (_python !== undefined) return _python;
  for (const cand of ['python3', 'python']) {
    try {
      const r = cp.spawnSync(cand, ['-c', 'import sys;print(sys.version_info[:2])'], { encoding: 'utf8', timeout: 15000 });
      if (r.status === 0 && /3\./.test(r.stdout)) { _python = cand; return cand; }
    } catch (e) { /* keep trying */ }
  }
  _python = null;
  return null;
}

/** Unpack .zip / .tar / .tar.gz via python stdlib (zipfile/tarfile). Returns true on success. */
function unpack(archivePath, destDir) {
  const py = findPython();
  if (!py) { warn('py', 'no python interpreter; cannot unpack ' + rel(archivePath)); return false; }
  const code = [
    'import sys,zipfile,tarfile,os',
    'p=sys.argv[1]; d=sys.argv[2]; os.makedirs(d,exist_ok=True)',
    'if p.endswith((".zip")):',
    '  zipfile.ZipFile(p).extractall(d)',
    'else:',
    '  tarfile.open(p).extractall(d)',
    'print("ok")'
  ].join('\n');
  const r = cp.spawnSync(py, ['-c', code, archivePath, destDir], { encoding: 'utf8', timeout: 600000 });
  if (r.status !== 0) {
    warn('py', 'unpack failed for ' + rel(archivePath) + ': ' + String((r.stderr || '')).split('\n')[0]);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------- http

function _request(method, target, timeoutMs) {
  return new Promise((resolve) => {
    let u;
    try { u = new url.URL(target); } catch (e) { resolve({ status: 0, error: 'bad url' }); return; }
    const lib = u.protocol === 'https:' ? https : u.protocol === 'http:' ? http : null;
    if (!lib) { resolve({ status: 0, error: 'unsupported protocol' }); return; }
    const req = lib.request(u, { method, timeout: timeoutMs, headers: { 'User-Agent': 'dsh-ocr-benchmark/1.0' } }, res => {
      // drain so the socket can be reused / freed
      res.resume();
      resolve({ status: res.statusCode, headers: res.headers, url: target });
    });
    req.on('timeout', () => { req.destroy(); resolve({ status: 0, error: 'timeout' }); });
    req.on('error', e => resolve({ status: 0, error: String(e && e.message || e) }));
    req.end();
  });
}

/** HEAD probe with up to 5 redirects. Resolves {status, ok, finalUrl, contentLength, error?}. */
async function headProbe(target, timeoutMs) {
  let cur = target, hops = 0;
  while (hops++ <= 5) {
    const r = await _request('HEAD', cur, timeoutMs || 10000);
    if (r.status >= 300 && r.status < 400 && r.headers && r.headers.location) {
      cur = new url.URL(r.headers.location, cur).href; continue;
    }
    const len = r.headers && (parseInt(r.headers['content-length'], 10) || 0);
    return { status: r.status, ok: r.status >= 200 && r.status < 300, finalUrl: cur, contentLength: Number.isFinite(len) ? len : 0, error: r.error };
  }
  return { status: 0, ok: false, error: 'too many redirects' };
}

/** GET to a file with redirect following; writes body to destPath. Resolves {status, ok, bytes, error?}. */
function downloadTo(target, destPath, timeoutMs) {
  return new Promise((resolve) => {
    let cur = target, hops = 0;
    const step = () => {
      if (hops++ > 5) { resolve({ status: 0, ok: false, error: 'too many redirects' }); return; }
      let u;
      try { u = new url.URL(cur); } catch (e) { resolve({ status: 0, ok: false, error: 'bad url' }); return; }
      const lib = u.protocol === 'https:' ? https : u.protocol === 'http:' ? http : null;
      if (!lib) { resolve({ status: 0, ok: false, error: 'unsupported protocol' }); return; }
      const req = lib.request(u, { method: 'GET', timeout: timeoutMs || 120000, headers: { 'User-Agent': 'dsh-ocr-benchmark/1.0' } }, res => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume(); cur = new url.URL(res.headers.location, cur).href; step(); return;
        }
        if (res.statusCode < 200 || res.statusCode >= 300) {
          res.resume(); resolve({ status: res.statusCode, ok: false, error: 'HTTP ' + res.statusCode }); return;
        }
        ensureDir(path.dirname(destPath));
        const out = fs.createWriteStream(destPath);
        let bytes = 0;
        res.on('data', c => { bytes += c.length; });
        res.on('end', () => { out.end(() => resolve({ status: res.statusCode, ok: true, bytes })); });
        res.pipe(out);
        out.on('error', e => resolve({ status: 0, ok: false, error: String(e.message) }));
      });
      req.on('timeout', () => { req.destroy(); resolve({ status: 0, ok: false, error: 'timeout' }); });
      req.on('error', e => resolve({ status: 0, ok: false, error: String(e && e.message || e) }));
      req.end();
    };
    step();
  });
}

// ---------------------------------------------------------------- logging

const C = { reset: '\x1b[0m', dim: '\x1b[2m', bold: '\x1b[1m', red: '\x1b[31m', green: '\x1b[32m', yellow: '\x1b[33m', blue: '\x1b[34m', cyan: '\x1b[36m' };

function log(tag, msg) { process.stdout.write(`${C.cyan}[${tag}]${C.reset} ${msg}\n`); }
function warn(tag, msg) { process.stdout.write(`${C.yellow}[${tag}:WARN]${C.reset} ${msg}\n`); }
function note(tag, msg) { process.stdout.write(`${C.blue}[${tag}]${C.reset} ${msg}\n`); }
function fail(tag, msg) { process.stderr.write(`${C.red}[${tag}:ERR]${C.reset} ${msg}\n`); }

// ---------------------------------------------------------------- table

/** Render an array of rows to a markdown table. rows: [{a:.., b:..}], cols: [keys], header labels optional. */
function markdownTable(rows, keys, headers) {
  const hs = headers || keys;
  const esc = v => String(v == null ? '' : v).replace(/\|/g, '\\|').replace(/\n/g, ' ');
  const lines = ['| ' + hs.map(esc).join(' | ') + ' |', '| ' + hs.map(() => '---').join(' | ') + ' |'];
  for (const r of rows) lines.push('| ' + keys.map(k => esc(r[k])).join(' | ') + ' |');
  return lines.join('\n');
}

module.exports = {
  BENCHMARK_DIR, REPO_ROOT, DATA_DIR, OUT_DIR, CLI_PATH, MODEL_VERIFY_PATH, BUDGET_MB,
  parseArgs, toInt,
  ensureDir, readJson, writeJson, exists, rel,
  aabbFromQuad, iouAABB,
  levenshtein, normText,
  quantile, mean, fmtMs, fmtPct, fmtMB,
  findPython, unpack,
  headProbe, downloadTo,
  C, log, warn, note, fail,
  markdownTable,
};
