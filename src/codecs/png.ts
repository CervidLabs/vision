/**
 * PNG codec — zero npm dependencies.
 * Uses Node.js built-in `zlib` for DEFLATE/inflate.
 *
 * Read  supports: 8-bit and 16-bit, all color types (0,2,3,4,6).
 *                 Indexed (type 3) is expanded to RGB/RGBA.
 *                 Grayscale+Alpha (type 4) is expanded to RGBA.
 *                 16-bit images are truncated to 8-bit (high byte kept).
 * Write supports: 1-channel (grayscale), 3-channel (RGB), 4-channel (RGBA).
 *                 Uses filter type 0 (None) — simple and portable.
 */

import { inflateSync, deflateSync } from 'node:zlib';
import { promises as fs } from 'node:fs';
import { VisionFrame } from '../core/VisionFrame.js';

// ── CRC-32 ────────────────────────────────────────────────────────────────────

const CRC_TABLE = new Uint32Array(256);
(function () {
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    CRC_TABLE[n] = c;
  }
})();

function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const PNG_SIG = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

function u32be(b: Uint8Array, o: number) {
  return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
}

function wu32be(b: Uint8Array, o: number, v: number) {
  b[o] = (v >>> 24) & 0xff;
  b[o + 1] = (v >>> 16) & 0xff;
  b[o + 2] = (v >>> 8) & 0xff;
  b[o + 3] = v & 0xff;
}

function makeChunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  wu32be(out, 0, data.length);
  for (let i = 0; i < 4; i++) {
    out[4 + i] = type.charCodeAt(i);
  }
  out.set(data, 8);
  wu32be(out, 8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

// ── Paeth predictor ───────────────────────────────────────────────────────────

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a),
    pb = Math.abs(p - b),
    pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

// ── De-filter (applied after inflate) ────────────────────────────────────────

function defilter(raw: Uint8Array, w: number, h: number, bpp: number): Uint8Array {
  const stride = w * bpp;
  const out = new Uint8Array(h * stride);
  let rp = 0,
    op = 0;

  for (let y = 0; y < h; y++) {
    const filter = raw[rp++];
    const row = raw.subarray(rp, rp + stride);
    rp += stride;
    const prev = y === 0 ? null : out.subarray((y - 1) * stride, y * stride);
    const cur = out.subarray(op, op + stride);

    switch (filter) {
      case 0:
        cur.set(row);
        break;
      case 1: // Sub
        for (let x = 0; x < stride; x++) {
          cur[x] = (row[x] + (x >= bpp ? cur[x - bpp] : 0)) & 0xff;
        }
        break;
      case 2: // Up
        for (let x = 0; x < stride; x++) {
          cur[x] = (row[x] + (prev ? prev[x] : 0)) & 0xff;
        }
        break;
      case 3: // Average
        for (let x = 0; x < stride; x++) {
          const a = x >= bpp ? cur[x - bpp] : 0;
          const b = prev ? prev[x] : 0;
          cur[x] = (row[x] + Math.floor((a + b) / 2)) & 0xff;
        }
        break;
      case 4: // Paeth
        for (let x = 0; x < stride; x++) {
          const a = x >= bpp ? cur[x - bpp] : 0;
          const b = prev ? prev[x] : 0;
          const c = x >= bpp && prev ? prev[x - bpp] : 0;
          cur[x] = (row[x] + paeth(a, b, c)) & 0xff;
        }
        break;
      default:
        throw new Error(`PNG: unknown filter type ${filter} at row ${y}`);
    }

    op += stride;
  }
  return out;
}

// ── Reader ────────────────────────────────────────────────────────────────────

export async function readPNG(path: string): Promise<VisionFrame> {
  const file = await fs.readFile(path);
  const buf = new Uint8Array(file.buffer, file.byteOffset, file.byteLength);

  // Verify signature
  for (let i = 0; i < 8; i++) {
    if (buf[i] !== PNG_SIG[i]) {
      throw new Error('PNG: invalid signature');
    }
  }

  let pos = 8;
  let width = 0,
    height = 0,
    bitDepth = 0,
    colorType = 0;
  const idatParts: Uint8Array[] = [];
  let palette: Uint8Array | null = null;
  let trns: Uint8Array | null = null;

  while (pos < buf.length) {
    const len = u32be(buf, pos);
    const type = String.fromCharCode(buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    pos += 12 + len;

    if (type === 'IHDR') {
      width = u32be(data, 0);
      height = u32be(data, 4);
      bitDepth = data[8];
      colorType = data[9];
      if (data[12] !== 0) {
        throw new Error('PNG: interlaced images not supported');
      }
      if (bitDepth !== 8 && bitDepth !== 16) {
        throw new Error(`PNG: bit depth ${bitDepth} not supported`);
      }
    } else if (type === 'PLTE') {
      palette = new Uint8Array(data);
    } else if (type === 'tRNS') {
      trns = new Uint8Array(data);
    } else if (type === 'IDAT') {
      idatParts.push(new Uint8Array(data));
    } else if (type === 'IEND') {
      break;
    }
  }

  if (!width || !height) {
    throw new Error('PNG: missing IHDR');
  }
  if (!idatParts.length) {
    throw new Error('PNG: no IDAT chunks');
  }

  // Concatenate compressed IDAT data
  const compLen = idatParts.reduce((s, c) => s + c.length, 0);
  const compressed = new Uint8Array(compLen);
  let off = 0;
  for (const c of idatParts) {
    compressed.set(c, off);
    off += c.length;
  }

  const decompBuf = inflateSync(Buffer.from(compressed.buffer, compressed.byteOffset, compressed.byteLength));
  const raw = new Uint8Array(decompBuf.buffer, decompBuf.byteOffset, decompBuf.byteLength);

  // channels in the raw data
  const rawCh = [1, 0, 3, 1, 2, 0, 4][colorType];
  if (!rawCh && colorType !== 0) {
    throw new Error(`PNG: unsupported color type ${colorType}`);
  }

  const bps = bitDepth === 16 ? 2 : 1; // bytes per sample
  const bpp = rawCh * bps; // bytes per pixel

  const pixels = defilter(raw, width, height, bpp);
  const n = width * height;

  // Determine output channel count
  let outCh: 1 | 3 | 4;
  switch (colorType) {
    case 0:
      outCh = 1;
      break; // grayscale
    case 2:
      outCh = 3;
      break; // RGB
    case 3:
      outCh = trns ? 4 : 3;
      break; // indexed
    case 4:
      outCh = 4;
      break; // grayscale+alpha → expand to RGBA
    case 6:
      outCh = 4;
      break; // RGBA
    default:
      outCh = 3;
  }

  const frame = new VisionFrame(width, height, outCh);
  const dst = frame.data;

  for (let i = 0; i < n; i++) {
    const sp = i * bpp;

    switch (colorType) {
      case 0: // grayscale (keep high byte for 16-bit)
        dst[i] = pixels[sp];
        break;
      case 2: // RGB
        if (bps === 2) {
          dst[i * 3] = pixels[sp];
          dst[i * 3 + 1] = pixels[sp + 2];
          dst[i * 3 + 2] = pixels[sp + 4];
        } else {
          dst[i * 3] = pixels[sp];
          dst[i * 3 + 1] = pixels[sp + 1];
          dst[i * 3 + 2] = pixels[sp + 2];
        }
        break;
      case 3: {
        // indexed — expand via palette
        const idx = pixels[sp];
        const r = palette![idx * 3],
          g = palette![idx * 3 + 1],
          b = palette![idx * 3 + 2];
        if (outCh === 4) {
          const a = trns && idx < trns.length ? trns[idx] : 255;
          dst[i * 4] = r;
          dst[i * 4 + 1] = g;
          dst[i * 4 + 2] = b;
          dst[i * 4 + 3] = a;
        } else {
          dst[i * 3] = r;
          dst[i * 3 + 1] = g;
          dst[i * 3 + 2] = b;
        }
        break;
      }
      case 4: {
        // grayscale+alpha → RGBA
        const v = pixels[sp];
        const a = bps === 2 ? pixels[sp + 2] : pixels[sp + 1];
        dst[i * 4] = v;
        dst[i * 4 + 1] = v;
        dst[i * 4 + 2] = v;
        dst[i * 4 + 3] = a;
        break;
      }
      case 6: // RGBA
        if (bps === 2) {
          dst[i * 4] = pixels[sp];
          dst[i * 4 + 1] = pixels[sp + 2];
          dst[i * 4 + 2] = pixels[sp + 4];
          dst[i * 4 + 3] = pixels[sp + 6];
        } else {
          dst[i * 4] = pixels[sp];
          dst[i * 4 + 1] = pixels[sp + 1];
          dst[i * 4 + 2] = pixels[sp + 2];
          dst[i * 4 + 3] = pixels[sp + 3];
        }
        break;
    }
  }

  return frame;
}

// ── Writer ────────────────────────────────────────────────────────────────────

export async function writePNG(path: string, frame: VisionFrame): Promise<void> {
  const { width, height, channels } = frame;
  if (channels !== 1 && channels !== 3 && channels !== 4) {
    throw new Error(`writePNG: unsupported channel count ${channels}`);
  }

  const colorType = channels === 1 ? 0 : channels === 3 ? 2 : 6;

  // IHDR
  const ihdr = new Uint8Array(13);
  wu32be(ihdr, 0, width);
  wu32be(ihdr, 4, height);
  ihdr[8] = 8;
  ihdr[9] = colorType; // others = 0

  // Raw scanlines: each row prefixed with filter byte 0 (None)
  const stride = width * channels;
  const rawData = new Uint8Array(height * (stride + 1));
  for (let y = 0; y < height; y++) {
    rawData[y * (stride + 1)] = 0; // filter None
    rawData.set(frame.data.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
  }

  const compressed = deflateSync(Buffer.from(rawData.buffer, rawData.byteOffset, rawData.byteLength), { level: 6 });
  const idat = new Uint8Array(compressed.buffer, compressed.byteOffset, compressed.byteLength);

  const iend = new Uint8Array(0);
  const chunks = [PNG_SIG, makeChunk('IHDR', ihdr), makeChunk('IDAT', idat), makeChunk('IEND', iend)];
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(total);
  let p = 0;
  for (const c of chunks) {
    out.set(c, p);
    p += c.length;
  }

  await fs.writeFile(path, Buffer.from(out.buffer, out.byteOffset, out.byteLength));
}
