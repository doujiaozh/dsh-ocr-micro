/**
 * DSH OCR Micro smoke test.
 * Part 1 is dependency-free unit testing of the pure-JS kernels (always runs).
 * Part 2 runs the full OCR pipeline if models are present, auto-generating a
 * test image via Pillow when needed.
 */

import * as fs from 'fs';
import * as path from 'path';
import { spawnSync } from 'child_process';
import { solveLinear, solveHomography, convexHull, minAreaRect } from './geometry';
import { ctcGreedyDecode } from './rec';
import { runOcr } from './pipeline';
import { defaultModelDir, fileExists } from './session';

const root = path.resolve(__dirname, '..', '..', '..');
const passed: string[] = [];
const failed: string[] = [];

function check(name: string, ok: boolean, extra?: string): void {
  (ok ? passed : failed).push(name + (extra ? ` (${extra})` : ''));
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? ' :: ' + extra : ''}`);
}

function approx(v: number, target: number, tol = 1e-6): boolean {
  return Math.abs(v - target) <= tol;
}

/* ------------------------- part 1: pure-JS kernels --------------------- */

function unitTests(): void {
  // 3x3 linear solve with a known solution.
  const x = solveLinear([[2, 1], [1, 3]], [5, 10]);
  check('solveLinear 2x2', approx(x[0], 1, 1e-9) && approx(x[1], 3, 1e-9), `x=[${x[0]},${x[1]}]`);

  // Identity homography on the unit square (dst == src).
  const sq: number[][] = [[0, 0], [9, 0], [9, 9], [0, 9]];
  const H = solveHomography(sq, sq);
  check('solveHomography identity',
    approx(H[0], 1) && approx(H[4], 1) &&
    approx(H[1], 0, 1e-9) && approx(H[2], 0, 1e-9) &&
    approx(H[3], 0, 1e-9) && approx(H[5], 0, 1e-9) &&
    approx(H[6], 0, 1e-9) && approx(H[7], 0, 1e-9));

  // Convex hull of a rotated square cloud has exactly 4 extreme points.
  // The exact corners are pushed last so boundary/collinear samples pop and
  // interior samples never reach the hull.
  const pts: number[][] = [];
  const angle = (30 * Math.PI) / 180;
  const cos = Math.cos(angle), sin = Math.sin(angle);
  const uv = (u: number, v: number): number[] => [
    100 + (u * 100) * cos - (v * 40) * sin,
    100 + (u * 100) * sin + (v * 40) * cos,
  ];
  for (let i = 0; i < 400; i++) {
    const u = ((i * 37) % 101) / 100 - 0.5; // deterministic interior params
    const v = ((i * 53) % 97) / 100 - 0.5;
    pts.push(uv(u, v));
  }
  pts.push(uv(-0.5, -0.5), uv(0.5, -0.5), uv(0.5, 0.5), uv(-0.5, 0.5));
  const hull = convexHull(pts);
  check('convexHull size', hull.length === 4, `hull=${hull.length}`);

  // Rotating calipers: area ~ 100*40 and angle ~ 30 deg.
  const rect = minAreaRect(pts);
  check('minAreaRect area', approx(rect.area, 4000, 40), `area=${rect.area.toFixed(1)}`);
  const deg = (rect.angle * 180) / Math.PI;
  const near30 = Math.min(Math.abs(deg - 30), Math.abs(deg + 60), Math.abs(deg - 150)) < 1.5;
  check('minAreaRect angle', near30, `angle=${deg.toFixed(2)}deg`);

  // CTC greedy decode: duplicate collapse + blank handling.
  const dict = ['a', 'b'];
  const C = 3, T = 4;
  const logits = new Float32Array(T * C);
  const set = (t: number, c: number, v: number) => { logits[t * C + c] = v; };
  set(0, 0, 8); set(1, 1, 8); set(2, 1, 8); set(3, 0, 8); // blank, a, a, blank -> "a"
  let r = ctcGreedyDecode(logits, T, C, dict);
  check('CTC collapse repeats', r.text === 'a', JSON.stringify(r));

  const logits2 = new Float32Array(T * C);
  const set2 = (t: number, c: number, v: number) => { logits2[t * C + c] = v; };
  set2(0, 1, 8); set2(1, 0, 8); set2(2, 1, 8); set2(3, 0, 8); // a, blank, a, blank -> "aa"
  r = ctcGreedyDecode(logits2, T, C, dict);
  check('CTC blank separates repeats', r.text === 'aa', JSON.stringify(r));
}

/* ------------------------- part 2: end-to-end --------------------------- */

const PY_GEN = `
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import random, os
random.seed(2026)
W, H = 720, 260
img = Image.new('RGB', (W, H), (248, 248, 248))
d = ImageDraw.Draw(img)
for _ in range(900):
    x, y = random.randint(0, W - 1), random.randint(0, H - 1)
    img.putpixel((x, y), (225, 225, 225))
font = None
for cand in ('msyh.ttc', 'msyhbd.ttc', 'simhei.ttf', 'simsun.ttc', 'Deng.ttf'):
    p = os.path.join('C:/Windows/Fonts', cand)
    if os.path.exists(p):
        font = ImageFont.truetype(p, 40)
        break
if font is None:
    raise SystemExit('no CJK font found')
lines = ['Hello DSH OCR Micro', '你好，世界 2026', '5.5MB 极限轻量精度']
y = 20
for t in lines:
    d.text((40, y), t, fill=(20, 20, 20), font=font)
    y += 70
img = img.rotate(-3.0, fillcolor=(248, 248, 248))
img = img.filter(ImageFilter.GaussianBlur(0.6))
img.save(sys.argv[1] if False else os.environ.get('DSH_TEST_OUT', 'test.png'))
print('ok')
`;

function generateTestImage(target: string): boolean {
  if (!fs.existsSync(path.join(root, 'test'))) fs.mkdirSync(path.join(root, 'test'), { recursive: true });
  const py = PY_GEN.replace("'test.png'", JSON.stringify(target));
  const res = spawnSync('python', ['-c', py], {
    env: { ...process.env, DSH_TEST_OUT: target },
    encoding: 'utf8',
    windowsHide: true,
  });
  return res.status === 0 && fs.existsSync(target);
}

async function e2e(): Promise<void> {
  const modelDir = defaultModelDir();
  const det = path.join(modelDir, 'det.onnx');
  if (!fileExists(det)) {
    console.log('\n[smoke] models not built yet - skipping e2e (run: npm run models && npm run compress)');
    return;
  }
  const imgPath = path.join(root, 'test', 'hello.png');
  if (!fs.existsSync(imgPath)) {
    if (!generateTestImage(imgPath)) {
      console.log('\n[smoke] could not generate test image (Pillow missing?) - skipping e2e');
      return;
    }
  }
  console.log(`\n[smoke] running OCR on ${path.relative(root, imgPath)} ...`);
  const result = await runOcr(imgPath, { verbose: true });
  console.log(`  engine=${result.models.engine} det=${result.models.detBytes}B rec=${result.models.recBytes}B`);
  console.log(`  timing=${JSON.stringify(result.timing)}`);
  for (const line of result.results) {
    console.log(`  [${(line.score * 100).toFixed(0)}%] ${line.text}`);
  }
  check('e2e: detected text', result.text.length > 0, `"${result.text.slice(0, 60)}"`);
  const lower = result.text.toLowerCase();
  check('e2e: latin line recognized', lower.includes('hello') || lower.includes('dsh') || lower.includes('ocr'), lower.slice(0, 40));
  check('e2e: cjk line recognized', /[\u4e00-\u9fff]/.test(result.text), '');
  check('e2e: model budget', result.models.detBytes + result.models.recBytes < 5.5 * 1024 * 1024,
    `${((result.models.detBytes + result.models.recBytes) / 1024 / 1024).toFixed(2)}MB`);
}

async function main(): Promise<void> {
  console.log('DSH OCR Micro - smoke test\n');
  unitTests();
  try {
    await e2e();
  } catch (e) {
    const msg = (e && (e as Error).message) ? (e as Error).message : String(e);
    failed.push('e2e: ' + msg);
    console.log('FAIL  e2e :: ' + msg);
  }
  console.log(`\n${passed.length} passed, ${failed.length} failed`);
  process.exit(failed.length > 0 ? 1 : 0);
}

main();
