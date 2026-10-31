/**
 * Image decoding/encoding without native addons.
 * PNG via pngjs (pure JS), JPEG via jpeg-js (pure JS).
 */

import * as png from 'pngjs';
import * as jpeg from 'jpeg-js';
import { RGBAImage } from './types';

const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47];
const JPEG_MAGIC = [0xff, 0xd8, 0xff];

function headMatches(buf: Buffer, magic: number[]): boolean {
  if (buf.length < magic.length) return false;
  for (let i = 0; i < magic.length; i++) {
    if (buf[i] !== magic[i]) return false;
  }
  return true;
}

function asUint8(data: Uint8Array | Buffer): Uint8Array {
  if (data instanceof Uint8Array) return data;
  return new Uint8Array(data);
}

/** Decode a PNG or JPEG buffer into RGBA8. Throw on anything else. */
export function decodeImage(buf: Buffer | Uint8Array): RGBAImage {
  const b: Buffer = Buffer.isBuffer(buf) ? buf : Buffer.from(buf);
  if (headMatches(b, PNG_MAGIC)) {
    const p = png.PNG.sync.read(b);
    return { width: p.width, height: p.height, data: asUint8(p.data) };
  }
  if (headMatches(b, JPEG_MAGIC)) {
    const j: any = jpeg.decode(b, { useTArray: true, maxMemoryUsageInMB: 512 });
    if (!j || !j.width || !j.data) {
      throw new Error('jpeg decode failed');
    }
    return { width: j.width, height: j.height, data: asUint8(j.data as Uint8Array) };
  }
  throw new Error('unsupported image format (PNG/JPEG only): magic=' + b.slice(0, 4).toString('hex'));
}

/** Encode RGBA8 back to a PNG buffer (deterministic output). */
export function encodePng(img: RGBAImage): Buffer {
  const p = new png.PNG({ width: img.width, height: img.height });
  p.data = Buffer.from(img.data); // defensive copy; PNG.sync.write mutates
  return png.PNG.sync.write(p);
}

/** Convenience: read an image file from disk. */
export function decodeFile(path: string): RGBAImage {
  const fs = require('fs');
  return decodeImage(fs.readFileSync(path));
}
