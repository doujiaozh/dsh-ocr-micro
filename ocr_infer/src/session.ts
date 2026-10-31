/**
 * DSH OCR Micro - ONNX session management (Win7-proof engine layer).
 *
 * Engine strategy (order matters):
 *   1. onnxruntime-web (WASM) - PRIMARY on the Win7 target. Pure JS/WASM, zero
 *      native DLL dependencies, immune to the missing api-ms-win-core-* stubs
 *      that make every onnxruntime-node release fail to load on Win7 SP1.
 *   2. onnxruntime-node - optional fast path when the native binding happens to
 *      load (e.g. dev machines on Win10+). Failure is silent and non-fatal.
 *
 * onnxruntime-web ships ESM-only in recent versions; our compiled output is
 * CommonJS, so we use a Function-wrapped dynamic import to keep a REAL ESM
 * import at runtime (TS would otherwise down-level import() to require()).
 * The WASM binary directory must be given as a file:// URL on Windows.
 *
 * Sessions are created lazily and cached per process. Everything here is
 * CommonJS + Node >= 14 friendly.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { pathToFileURL } from 'url';

/** Plain-data tensor contract used by det/rec - decoupled from any ORT API. */
export interface OrtTensor {
  data: Float32Array;
  dims: number[];
}

export type OrtSession = {
  run(inputs: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  inputNames: string[];
  outputNames: string[];
};

export interface ModelPaths {
  det: string;
  rec: string;
  dict: string;
}

const dynamicImport: (id: string) => Promise<any> = new Function(
  'id',
  'return import(id)',
) as (id: string) => Promise<any>;

function requireSafe(id: string): any {
  const req: (id: string) => any =
    typeof (globalThis as any).require === 'function'
      ? (globalThis as any).require
      : (0, eval)('require');
  return req(id);
}

let ortModule: any | null = null;
let engineName = '';
let ortLoading: Promise<any> | null = null;

function configureWasm(mod: any): void {
  // Locate the wasm binaries that ship with onnxruntime-web.
  let distDir = '';
  try {
    const pkgPath = require.resolve('onnxruntime-web/package.json');
    distDir = path.join(path.dirname(pkgPath), 'dist');
  } catch {
    // Fall back to walking up node_modules from this file.
    let dir: string | undefined = __dirname;
    for (let i = 0; i < 6 && dir; i++) {
      const cand = path.join(dir, 'node_modules', 'onnxruntime-web', 'dist');
      if (fs.existsSync(cand)) {
        distDir = cand;
        break;
      }
      dir = path.dirname(dir);
    }
  }
  if (distDir && fs.existsSync(distDir)) {
    // Windows absolute paths must be converted to file:// URLs for the ESM loader.
    mod.env.wasm.wasmPaths = pathToFileURL(distDir + path.sep).href;
  }
  // Single-threaded: avoids pthread worker requirements in non-browser hosts.
  mod.env.wasm.numThreads = 1;
}

async function loadOrt(): Promise<any> {
  if (ortModule) return ortModule;
  if (!ortLoading) {
    ortLoading = (async () => {
      // Optional native fast path; never fatal.
      try {
        const mod = requireSafe('onnxruntime-node');
        // eslint-disable-next-line no-new-wrappers
        if (mod && mod.InferenceSession) {
          engineName = 'onnxruntime-node';
          return (ortModule = mod);
        }
      } catch {
        // native binding unavailable (typical on Win7) - fall through to WASM
      }
      // Primary path: pure WASM.
      const mod = await dynamicImport('onnxruntime-web');
      if (!mod || !mod.InferenceSession) {
        throw new Error('[dsh-ocr-micro] onnxruntime-web loaded but has no InferenceSession');
      }
      configureWasm(mod);
      engineName = 'onnxruntime-web/wasm';
      return (ortModule = mod);
    })().catch((e) => {
      ortLoading = null;
      throw e;
    });
  }
  return ortLoading;
}

/** Load the ort module first. */
export async function getEngineName(): Promise<string> {
  await loadOrt();
  return engineName;
}

/** Default model location: <repo-root>/models (dist layout: dist/ocr_infer/src). */
export function defaultModelDir(): string {
  return path.resolve(__dirname, '..', '..', '..', 'models');
}

export function resolveModelPaths(modelDir?: string): ModelPaths {
  const dir = modelDir ? path.resolve(modelDir) : defaultModelDir();
  // PP-OCRv6 tiny ships its dict as ppocrv6_tiny_dict.txt; older tiers used
  // ppocr_keys_v1.txt. Accept either so model_forge can swap tiers freely.
  const dict =
    fileExists(path.join(dir, 'ppocr_keys_v1.txt'))
      ? path.join(dir, 'ppocr_keys_v1.txt')
      : path.join(dir, 'ppocrv6_tiny_dict.txt');
  return {
    det: path.join(dir, 'det.onnx'),
    rec: path.join(dir, 'rec.onnx'),
    dict,
  };
}

const sessionCache = new Map<string, Promise<OrtSession>>();
const sizeCache = new Map<string, number>();

function fileBytes(p: string): number {
  if (!sizeCache.has(p)) {
    try {
      sizeCache.set(p, fs.statSync(p).size);
    } catch {
      sizeCache.set(p, 0);
    }
  }
  return sizeCache.get(p) as number;
}

export function modelBytes(det: string, rec: string): { detBytes: number; recBytes: number } {
  return { detBytes: fileBytes(det), recBytes: fileBytes(rec) };
}

function cpuThreads(requested?: number): number {
  if (requested && requested > 0) return requested;
  const cpus = (os && os.cpus && os.cpus().length) || 1;
  // Win7/8GB: keep the thread pool modest; ONNX scales poorly past 4 threads here.
  return Math.max(1, Math.min(4, cpus));
}

function wrapSession(sess: any): OrtSession {
  return {
    inputNames: sess.inputNames,
    outputNames: sess.outputNames,
    async run(inputs: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>> {
      const ort = await loadOrt();
      const feeds: Record<string, any> = {};
      for (const key of Object.keys(inputs)) {
        const t = inputs[key];
        feeds[key] = new ort.Tensor('float32', t.data, t.dims);
      }
      const outputs = await sess.run(feeds);
      const result: Record<string, OrtTensor> = {};
      for (const key of Object.keys(outputs)) {
        const out = outputs[key];
        result[key] = { data: out.data, dims: out.dims };
      }
      return result;
    },
  };
}

export async function getSession(modelPath: string, threads?: number): Promise<OrtSession> {
  const key = `${modelPath}#${threads || 0}`;
  const hit = sessionCache.get(key);
  if (hit) return hit;
  const promise = (async () => {
    const ort = await loadOrt();
    let buf: Buffer;
    try {
      buf = fs.readFileSync(modelPath);
    } catch (e) {
      throw new Error(`[dsh-ocr-micro] model file not found: ${modelPath} (run: node model_forge/download.js)`);
    }
    const opts: any = {
      intraOpNumThreads: cpuThreads(threads),
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
      enableCpuMemArena: false,
      enableMemPattern: false,
    };
    const make = async (o?: any) => (await ort.InferenceSession.create(buf, o)) as any;
    try {
      return wrapSession(await make(opts));
    } catch (e) {
      // Older/smaller onnxruntime builds reject some option keys - retry with defaults.
      return wrapSession(await make(undefined));
    }
  })();
  sessionCache.set(key, promise);
  return promise;
}

export function fileExists(p: string): boolean {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}
