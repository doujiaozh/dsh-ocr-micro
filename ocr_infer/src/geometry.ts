/**
 * DSH OCR Micro - pure-JS geometric kernels.
 *
 * No OpenCV, no canvas: everything an OCR engine needs between "bytes" and
 * "tensors" is implemented here with plain typed arrays so it runs anywhere
 * Node runs (Win7 included).
 *
 *  - bilinear RGBA -> NCHW float resize (detector input)
 *  - homography solve + perspective warp (text-line crop for the recognizer)
 *  - bilinear float-map resize (DB probability map back to original size)
 *  - convex hull (monotone chain) + rotating-calipers minimum-area rectangle
 *  - rect unclip expansion (DB box post-processing)
 */

export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

export interface RGBAImage {
  width: number;
  height: number;
  data: Uint8Array; // w*h*4
}

/* --------------------------------------------------------------------------
 * Linear algebra
 * ------------------------------------------------------------------------ */

/** Solve A*x = b with Gaussian elimination + partial pivoting. A is n x n. */
export function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M: number[][] = A.map((row, i) => row.slice().concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    let best = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r][col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-12) throw new Error('solveLinear: singular matrix');
    if (piv !== col) { const tmp = M[piv]; M[piv] = M[col]; M[col] = tmp; }
    const invDiag = 1 / M[col][col];
    for (let r = col + 1; r < n; r++) {
      const factor = M[r][col] * invDiag;
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = s / M[r][r];
  }
  return x;
}

/**
 * Solve the 3x3 homography (last coefficient fixed at 1) that maps `dst`
 * points to `src` points. dst is the output rectangle corner set, src the
 * (possibly rotated) text-line quad in image space. 4 correspondences => 8
 * unknowns, exactly determined.
 */
export function solveHomography(dst: number[][], src: number[][]): number[] {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const x = dst[i][0], y = dst[i][1];
    const u = src[i][0], v = src[i][1];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  return solveLinear(A, b);
}

/* --------------------------------------------------------------------------
 * Pixel resampling
 * ------------------------------------------------------------------------ */

/** Bilinear RGBA -> NCHW float32 (1x3xdhxdw), channel-normalized. */
export function resizeRgbaToNCHW(
  img: RGBAImage,
  dw: number,
  dh: number,
  mean: number[],
  std: number[]
): Float32Array {
  const sw = img.width, sh = img.height, src = img.data;
  const out = new Float32Array(3 * dw * dh);
  const xRatio = sw > 1 && dw > 1 ? (sw - 1) / (dw - 1) : 0;
  const yRatio = sh > 1 && dh > 1 ? (sh - 1) / (dh - 1) : 0;
  const plane = dw * dh;
  for (let dy = 0; dy < dh; dy++) {
    const sy = dy * yRatio;
    const y0 = Math.min(sh - 1, Math.floor(sy));
    const y1 = Math.min(sh - 1, y0 + 1);
    const wy = sy - y0;
    for (let dx = 0; dx < dw; dx++) {
      const sx = dx * xRatio;
      const x0 = Math.min(sw - 1, Math.floor(sx));
      const x1 = Math.min(sw - 1, x0 + 1);
      const wx = sx - x0;
      const i00 = (y0 * sw + x0) << 2, i10 = (y0 * sw + x1) << 2;
      const i01 = (y1 * sw + x0) << 2, i11 = (y1 * sw + x1) << 2;
      const w00 = (1 - wx) * (1 - wy), w10 = wx * (1 - wy);
      const w01 = (1 - wx) * wy, w11 = wx * wy;
      for (let c = 0; c < 3; c++) {
        const v =
          src[i00 + c] * w00 + src[i10 + c] * w10 +
          src[i01 + c] * w01 + src[i11 + c] * w11;
        out[c * plane + dy * dw + dx] = (v / 255 - mean[c]) / std[c];
      }
    }
  }
  return out;
}

function sampleRGBAClamped(
  src: Uint8Array, sw: number, sh: number, x: number, y: number, out: number[]
): void {
  const cx = Math.max(0, Math.min(sw - 1, x));
  const cy = Math.max(0, Math.min(sh - 1, y));
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  const x1 = Math.min(sw - 1, x0 + 1), y1 = Math.min(sh - 1, y0 + 1);
  const wx = cx - x0, wy = cy - y0;
  const i00 = (y0 * sw + x0) << 2, i10 = (y0 * sw + x1) << 2;
  const i01 = (y1 * sw + x0) << 2, i11 = (y1 * sw + x1) << 2;
  for (let c = 0; c < 3; c++) {
    out[c] =
      src[i00 + c] * (1 - wx) * (1 - wy) + src[i10 + c] * wx * (1 - wy) +
      src[i01 + c] * (1 - wx) * wy + src[i11 + c] * wx * wy;
  }
}

/**
 * Perspective-warp an arbitrary rotated text quad out of the image into a
 * normalized (dw x dh) crop, returned as NCHW float32 (1x3xdhxdw) with the
 * given normalization. Out-of-range samples clamp to the border, which is
 * robust for text crops (PaddleOCR uses BORDER_REPLICATE as well).
 */
export function perspectiveWarpToNCHW(
  img: RGBAImage,
  srcQuad: number[][],
  dw: number,
  dh: number,
  mean: number[],
  std: number[]
): Float32Array {
  const dstQuad: number[][] = [[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]];
  const H = solveHomography(dstQuad, srcQuad);
  const h0 = H[0], h1 = H[1], h2 = H[2], h3 = H[3], h4 = H[4], h5 = H[5], h6 = H[6], h7 = H[7];
  const sw = img.width, sh = img.height, src = img.data;
  const out = new Float32Array(3 * dw * dh);
  const plane = dw * dh;
  const rgb = [0, 0, 0];
  for (let dy = 0; dy < dh; dy++) {
    for (let dx = 0; dx < dw; dx++) {
      const w = 1 / (h6 * dx + h7 * dy + 1);
      const sx = (h0 * dx + h1 * dy + h2) * w;
      const sy = (h3 * dx + h4 * dy + h5) * w;
      sampleRGBAClamped(src, sw, sh, sx, sy, rgb);
      for (let c = 0; c < 3; c++) {
        out[c * plane + dy * dw + dx] = (rgb[c] / 255 - mean[c]) / std[c];
      }
    }
  }
  return out;
}

/** Bilinear resize of a single-channel float map (detector output upsampling). */
export function resizeFloatMap(
  src: Float32Array, sw: number, sh: number,
  dw: number, dh: number
): Float32Array {
  const out = new Float32Array(dw * dh);
  const xRatio = sw > 1 && dw > 1 ? (sw - 1) / (dw - 1) : 0;
  const yRatio = sh > 1 && dh > 1 ? (sh - 1) / (dh - 1) : 0;
  for (let dy = 0; dy < dh; dy++) {
    const sy = dy * yRatio;
    const y0 = Math.min(sh - 1, Math.floor(sy));
    const y1 = Math.min(sh - 1, y0 + 1);
    const wy = sy - y0;
    for (let dx = 0; dx < dw; dx++) {
      const sx = dx * xRatio;
      const x0 = Math.min(sw - 1, Math.floor(sx));
      const x1 = Math.min(sw - 1, x0 + 1);
      const wx = sx - x0;
      const i00 = y0 * sw + x0, i10 = y0 * sw + x1, i01 = y1 * sw + x0, i11 = y1 * sw + x1;
      out[dy * dw + dx] =
        src[i00] * (1 - wx) * (1 - wy) + src[i10] * wx * (1 - wy) +
        src[i01] * (1 - wx) * wy + src[i11] * wx * wy;
    }
  }
  return out;
}

/* --------------------------------------------------------------------------
 * Convex hull + rotating calipers minimum-area rectangle
 * ------------------------------------------------------------------------ */

export function convexHull(points: number[][]): number[][] {
  const pts = points.slice().sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const n = pts.length;
  if (n <= 2) return pts;
  const cross = (o: number[], a: number[], b: number[]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  // Pop while not a strict CCW turn; the epsilon swallows floating-point
  // residue on exactly-collinear boundary points (rotated grids produce
  // cross values around 1e-12 instead of a hard 0).
  const EPS = 1e-6;
  const lower: number[][] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= EPS) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: number[][] = [];
  for (let i = n - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= EPS) {
      upper.pop();
    }
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

export interface MinAreaRect {
  corners: number[][]; // 4x[x, y], clockwise from top-left
  center: number[];
  width: number;
  height: number;
  angle: number; // radians of the width-axis w.r.t. +x
  area: number;
}

/**
 * Minimum-area enclosing rectangle of a point set via the classic
 * rotating-calipers technique over the convex hull. Equivalent of
 * cv2.minAreaRect, and the quality gate for our DB box construction.
 */
export function minAreaRect(points: number[][]): MinAreaRect {
  const hull = convexHull(points);
  if (hull.length === 0) {
    return { corners: [[0, 0], [0, 0], [0, 0], [0, 0]], center: [0, 0], width: 0, height: 0, angle: 0, area: 0 };
  }
  if (hull.length === 1) {
    const p = hull[0];
    return { corners: [p, p, p, p], center: p.slice(), width: 1, height: 1, angle: 0, area: 1 };
  }
  if (hull.length === 2) {
    const [a, b] = hull;
    const cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2;
    const w = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    return { corners: [a, b, b, a], center: [cx, cy], width: w, height: 1, angle: Math.atan2(b[1] - a[1], b[0] - a[0]), area: w };
  }
  let best: MinAreaRect | null = null;
  let bestArea = Infinity;
  for (let i = 0; i < hull.length; i++) {
    const p0 = hull[i];
    const p1 = hull[(i + 1) % hull.length];
    let ex = p1[0] - p0[0], ey = p1[1] - p0[1];
    const elen = Math.hypot(ex, ey);
    if (elen < 1e-9) continue;
    ex /= elen; ey /= elen; // unit edge vector
    const nx = -ey, ny = ex; // perpendicular
    let minU = Infinity, maxU = -Infinity, minN = Infinity, maxN = -Infinity;
    for (const p of hull) {
      const du = (p[0] - p0[0]) * ex + (p[1] - p0[1]) * ey;
      const dn = (p[0] - p0[0]) * nx + (p[1] - p0[1]) * ny;
      if (du < minU) minU = du;
      if (du > maxU) maxU = du;
      if (dn < minN) minN = dn;
      if (dn > maxN) maxN = dn;
    }
    const w = maxU - minU, h = maxN - minN;
    const area = w * h;
    if (area < bestArea) {
      bestArea = area;
      const cu = (minU + maxU) / 2, cn = (minN + maxN) / 2;
      const cx = p0[0] + ex * cu + nx * cn;
      const cy = p0[1] + ey * cu + ny * cn;
      // corner offsets in (e, n) basis
      const du = w / 2, dn = h / 2;
      const corners: number[][] = [
        [cx - ex * du - nx * dn, cy - ey * du - ny * dn],
        [cx + ex * du - nx * dn, cy + ey * du - ny * dn],
        [cx + ex * du + nx * dn, cy + ey * du + ny * dn],
        [cx - ex * du + nx * dn, cy - ey * du + ny * dn],
      ];
      best = { corners, center: [cx, cy], width: w, height: h, angle: Math.atan2(ey, ex), area };
    }
  }
  return best as MinAreaRect;
}

/**
 * DB "unclip": expand the minimum-area rectangle along its own axes so the
 * returned quad fully contains the text stroke. Uses the PaddleOCR-style
 * area/perimeter offset ratio, applied as a rectangle expansion.
 */
export function unclipRect(rect: MinAreaRect, ratio: number): number[][] {
  const w = rect.width * ratio;
  const h = rect.height * ratio;
  const cx = rect.center[0], cy = rect.center[1];
  const a = rect.angle;
  const ex = Math.cos(a), ey = Math.sin(a);
  const nx = -ey, ny = ex;
  const du = w / 2, dn = h / 2;
  return [
    [cx - ex * du - nx * dn, cy - ey * du - ny * dn],
    [cx + ex * du - nx * dn, cy + ey * du - ny * dn],
    [cx + ex * du + nx * dn, cy + ey * du + ny * dn],
    [cx - ex * du + nx * dn, cy - ey * du + ny * dn],
  ];
}

export function clampQuad(quad: number[][], w: number, h: number): void {
  for (const p of quad) {
    if (p[0] < 0) p[0] = 0; if (p[0] > w - 1) p[0] = w - 1;
    if (p[1] < 0) p[1] = 0; if (p[1] > h - 1) p[1] = h - 1;
  }
}

/** Non-mutating clamp: returns a fresh quad, leaves the input untouched. */
export function clampQuadCopy(quad: number[][], w: number, h: number): number[][] {
  return quad.map(p => [
    p[0] < 0 ? 0 : (p[0] > w - 1 ? w - 1 : p[0]),
    p[1] < 0 ? 0 : (p[1] > h - 1 ? h - 1 : p[1]),
  ]);
}

/** Axis-aligned bounding box of a quad. */
export function quadAABB(quad: number[][]): { x0: number; y0: number; x1: number; y1: number } {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of quad) {
    if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0];
    if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1];
  }
  return { x0, y0, x1, y1 };
}

/**
 * Reorder 4 quad corners into reading order: [TL, TR, BR, BL].
 *
 * minAreaRect (like cv2) picks its starting corner by the caliper angle, so a
 * box whose width-axis points leftward arrives as TR-at-x0 - and a perspective
 * warp built on that order mirrors the text. This normalization is winding-
 * and convention-agnostic: works for lines rotated by deskew residuals.
 * Exact for any quad whose reading direction is nearer horizontal than
 * vertical (the post-deskew invariant).
 */
export function readingOrderQuad(quad: number[][]): number[][] {
  if (quad.length !== 4) return quad;
  let tl = 0, br = 0, tr = 0, bl = 0;
  for (let i = 1; i < 4; i++) {
    if (quad[i][0] + quad[i][1] < quad[tl][0] + quad[tl][1]) tl = i;
    if (quad[i][0] + quad[i][1] > quad[br][0] + quad[br][1]) br = i;
    if (quad[i][0] - quad[i][1] > quad[tr][0] - quad[tr][1]) tr = i;
    if (quad[i][0] - quad[i][1] < quad[bl][0] - quad[bl][1]) bl = i;
  }
  return [quad[tl], quad[tr], quad[br], quad[bl]];
}
