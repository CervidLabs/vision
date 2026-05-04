/**
 * PNG codec — zero npm dependencies.
 * Uses Node.js built-in `zlib` for DEFLATE/inflate.
 *
 * Read  supports: 8-bit and 16-bit, all color types (0,2,3,4,6).
 * Write supports: 1, 3, 4-channel.  Sub filter + parallel DEFLATE.
 *
 * Parallel DEFLATE strategy
 * ─────────────────────────
 * zlib's deflate is single-threaded.  For a 30 MP image we split the
 * filtered scanline buffer into (cpus-1) independent chunks and compress
 * each compressed by a real OS thread (Worker) using deflateRawSync. Each
 * non-final chunk ends with a 4-byte sync-flush marker (00 00 FF FF)
 * that is stripped before concatenation.  The raw DEFLATE bitstreams
 * are then joined and wrapped in a standard 2-byte zlib CMF/FLG header
 * + 4-byte Adler-32 trailer, producing a valid zlib stream for PNG IDAT.
 */

import { deflateSync, inflateSync } from 'node:zlib';
import { promises as fs } from 'node:fs';
import { cpus } from 'node:os';
import { VisionFrame } from '../core/VisionFrame.js';
import { getPngCompressPool } from './PngCompressPool.js';

// ── CRC-32 ────────────────────────────────────────────────────────────────────

const CRC_TABLE = new Uint32Array(256);
(function () {
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    CRC_TABLE[n] = c;
  }
})();

function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ── Adler-32 (needed to build the zlib stream wrapper manually) ───────────────

function adler32(data: Uint8Array): number {
  let a = 1, b = 0;
  const NMAX = 5552;  // max iterations before 32-bit overflow
  let i = 0;
  while (i < data.length) {
    const end = Math.min(i + NMAX, data.length);
    while (i < end) { a += data[i++]; b += a; }
    a %= 65521;
    b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const PNG_SIG = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

function u32be(b: Uint8Array, o: number) {
  return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
}

function wu32be(b: Uint8Array, o: number, v: number) {
  b[o] = (v >>> 24) & 0xff; b[o + 1] = (v >>> 16) & 0xff;
  b[o + 2] = (v >>> 8) & 0xff; b[o + 3] = v & 0xff;
}

function makeChunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  wu32be(out, 0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  wu32be(out, 8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

// ── Paeth predictor ───────────────────────────────────────────────────────────

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

// ── De-filter ─────────────────────────────────────────────────────────────────

function defilter(raw: Uint8Array, w: number, h: number, bpp: number): Uint8Array {
  const stride = w * bpp;
  const out = new Uint8Array(h * stride);
  let rp = 0, op = 0;

  for (let y = 0; y < h; y++) {
    const filter = raw[rp++];
    const row = raw.subarray(rp, rp + stride); rp += stride;
    const prev = y === 0 ? null : out.subarray((y - 1) * stride, y * stride);
    const cur = out.subarray(op, op + stride);

    switch (filter) {
      case 0: cur.set(row); break;
      case 1:
        for (let x = 0; x < stride; x++)
          cur[x] = (row[x] + (x >= bpp ? cur[x - bpp] : 0)) & 0xff;
        break;
      case 2:
        for (let x = 0; x < stride; x++)
          cur[x] = (row[x] + (prev ? prev[x] : 0)) & 0xff;
        break;
      case 3:
        for (let x = 0; x < stride; x++) {
          const a = x >= bpp ? cur[x - bpp] : 0;
          const b = prev ? prev[x] : 0;
          cur[x] = (row[x] + ((a + b) >> 1)) & 0xff;
        }
        break;
      case 4:
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

// ── Parallel DEFLATE via persistent worker pool ──────────────────────────────
//
// Each chunk is compressed by a real OS thread (Worker) using deflateRawSync.
// Workers are persistent (PngCompressPool) — no ~30ms startup cost per call.
//
// Non-final chunks use Z_SYNC_FLUSH which appends a valid zero-length stored
// block (5 bytes: 00 00 00 FF FF).  We keep these bytes — they are valid
// DEFLATE that any decoder handles transparently (LEN=0, NLEN=~0 ✓).
//
// Final chunk uses Z_FINISH (BFINAL=1).
//
// Output zlib stream:
//   [0x78 0x9C]                 zlib header
//   [chunk_0][00 00 00 FF FF]   chunk + sync-flush marker (LEN=0 stored block)
//   ...
//   [chunk_N]                   final chunk (BFINAL=1, no sync marker)
//   [Adler-32 big-endian]       checksum of all uncompressed bytes

const NUM_DEFLATE_WORKERS = Math.max(1, Math.min(8, cpus().length - 1));
const ZLIB_HEADER = Buffer.from([0x78, 0x9c]);  // CMF=0x78, FLG=0x9C (% 31 === 0 ✓)

async function deflateParallel(dataSAB: SharedArrayBuffer, byteOffset: number, byteLength: number, level: number): Promise<Buffer> {
  const data = new Uint8Array(dataSAB, byteOffset, byteLength);
  const pool = getPngCompressPool();
  const n = NUM_DEFLATE_WORKERS;
  const chunkSize = Math.ceil(byteLength / n);

  // Dispatch all chunks to worker threads.  Workers start deflateRawSync
  // immediately in their OS threads — truly parallel, no libuv pool limit.
  const jobs: Promise<ArrayBuffer>[] = [];
  for (let i = 0; i < n; i++) {
    const offset = byteOffset + i * chunkSize;
    const length = Math.min(byteLength - i * chunkSize, chunkSize);
    if (length <= 0) break;
    jobs.push(pool.run({
      dataSAB,
      offset,
      length,
      isFinal: (offset + length >= byteOffset + byteLength),
      level,
    }));
  }

  // While workers compress (parallel), compute Adler-32 on the main thread.
  // adler32 takes ~100ms on 30MB — it overlaps with the ~250ms worker time,
  // so it costs nothing in wall-clock terms.
  const checksum = adler32(data);

  const results = await Promise.all(jobs);
  const parts = results.map(ab => Buffer.from(ab));
  const raw = Buffer.concat(parts);
  const trailer = Buffer.allocUnsafe(4);
  trailer.writeUInt32BE(checksum, 0);
  return Buffer.concat([ZLIB_HEADER, raw, trailer]);
}

// ── Reader ────────────────────────────────────────────────────────────────────

export async function readPNG(path: string): Promise<VisionFrame> {
  const file = await fs.readFile(path);
  const buf = new Uint8Array(file.buffer, file.byteOffset, file.byteLength);

  for (let i = 0; i < 8; i++)
    if (buf[i] !== PNG_SIG[i]) throw new Error('PNG: invalid signature');

  let pos = 8;
  let width = 0, height = 0, bitDepth = 0, colorType = 0;
  const idatParts: Uint8Array[] = [];
  let palette: Uint8Array | null = null;
  let trns: Uint8Array | null = null;

  while (pos < buf.length) {
    const len = u32be(buf, pos);
    const type = String.fromCharCode(buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    pos += 12 + len;

    if (type === 'IHDR') {
      width = u32be(data, 0); height = u32be(data, 4);
      bitDepth = data[8]; colorType = data[9];
      if (data[12] !== 0) throw new Error('PNG: interlaced not supported');
      if (bitDepth !== 8 && bitDepth !== 16) throw new Error(`PNG: bit depth ${bitDepth} not supported`);
    }
    else if (type === 'PLTE') { palette = new Uint8Array(data); }
    else if (type === 'tRNS') { trns = new Uint8Array(data); }
    else if (type === 'IDAT') { idatParts.push(new Uint8Array(data)); }
    else if (type === 'IEND') { break; }
  }

  if (!width || !height) throw new Error('PNG: missing IHDR');
  if (!idatParts.length) throw new Error('PNG: no IDAT chunks');

  const compLen = idatParts.reduce((s, c) => s + c.length, 0);
  const compressed = new Uint8Array(compLen);
  let off = 0;
  for (const c of idatParts) { compressed.set(c, off); off += c.length; }

  const decompBuf = inflateSync(Buffer.from(compressed.buffer, compressed.byteOffset, compressed.byteLength));
  const raw = new Uint8Array(decompBuf.buffer, decompBuf.byteOffset, decompBuf.byteLength);

  const rawCh = [1, 0, 3, 1, 2, 0, 4][colorType];
  if (!rawCh && colorType !== 0) throw new Error(`PNG: unsupported color type ${colorType}`);

  const bps = bitDepth === 16 ? 2 : 1;
  const bpp = rawCh * bps;
  const pixels = defilter(raw, width, height, bpp);
  const n = width * height;

  let outCh: 1 | 3 | 4;
  switch (colorType) {
    case 0: outCh = 1; break; case 2: outCh = 3; break;
    case 3: outCh = trns ? 4 : 3; break;
    case 4: case 6: outCh = 4; break; default: outCh = 3;
  }

  const frame = new VisionFrame(width, height, outCh);
  const dst = frame.data;

  for (let i = 0; i < n; i++) {
    const sp = i * bpp;
    switch (colorType) {
      case 0: dst[i] = pixels[sp]; break;
      case 2:
        if (bps === 2) { dst[i * 3] = pixels[sp]; dst[i * 3 + 1] = pixels[sp + 2]; dst[i * 3 + 2] = pixels[sp + 4]; }
        else { dst[i * 3] = pixels[sp]; dst[i * 3 + 1] = pixels[sp + 1]; dst[i * 3 + 2] = pixels[sp + 2]; }
        break;
      case 3: {
        const idx = pixels[sp];
        const r = palette![idx * 3], g = palette![idx * 3 + 1], b = palette![idx * 3 + 2];
        if (outCh === 4) { dst[i * 4] = r; dst[i * 4 + 1] = g; dst[i * 4 + 2] = b; dst[i * 4 + 3] = trns && idx < trns.length ? trns[idx] : 255; }
        else { dst[i * 3] = r; dst[i * 3 + 1] = g; dst[i * 3 + 2] = b; }
        break;
      }
      case 4: {
        const v = pixels[sp], a = bps === 2 ? pixels[sp + 2] : pixels[sp + 1];
        dst[i * 4] = v; dst[i * 4 + 1] = v; dst[i * 4 + 2] = v; dst[i * 4 + 3] = a;
        break;
      }
      case 6:
        if (bps === 2) { dst[i * 4] = pixels[sp]; dst[i * 4 + 1] = pixels[sp + 2]; dst[i * 4 + 2] = pixels[sp + 4]; dst[i * 4 + 3] = pixels[sp + 6]; }
        else { dst[i * 4] = pixels[sp]; dst[i * 4 + 1] = pixels[sp + 1]; dst[i * 4 + 2] = pixels[sp + 2]; dst[i * 4 + 3] = pixels[sp + 3]; }
        break;
    }
  }

  return frame;
}

// ── Writer ────────────────────────────────────────────────────────────────────
export interface PNGWriteOptions {
  level?: number;
  filter?: "none" | "sub";
}

export async function writePNG(
  path: string,
  frame: VisionFrame,
  opts: PNGWriteOptions = {},
): Promise<void> {
  const level = opts.level ?? 1;
  const filter = opts.filter ?? "sub"
  const { width, height, channels } = frame;

  if (channels !== 1 && channels !== 3 && channels !== 4) {
    throw new Error(`writePNG: unsupported channel count ${channels}`);
  }

  const colorType = channels === 1 ? 0 : channels === 3 ? 2 : 6;
  const bpp = channels;
  const stride = width * channels;

  const ihdr = new Uint8Array(13);
  wu32be(ihdr, 0, width);
  wu32be(ihdr, 4, height);
  ihdr[8] = 8;
  ihdr[9] = colorType;

  const rawData = new Uint8Array(height * (stride + 1));
  const src = frame.data;

  for (let y = 0; y < height; y++) {
    const rb = y * (stride + 1);
    const sb = y * stride;

    if (filter === "none") {
      rawData[rb] = 0;
      rawData.set(src.subarray(sb, sb + stride), rb + 1);
      continue;
    }

    rawData[rb] = 1;

    for (let x = 0; x < bpp; x++) {
      rawData[rb + 1 + x] = src[sb + x];
    }

    for (let x = bpp; x < stride; x++) {
      rawData[rb + 1 + x] = (src[sb + x] - src[sb + x - bpp]) & 0xff;
    }
  }

  const compressed = deflateSync(rawData, { level });
  const idat = new Uint8Array(
    compressed.buffer,
    compressed.byteOffset,
    compressed.byteLength
  );

  const iend = new Uint8Array(0);

  const chunks = [
    PNG_SIG,
    makeChunk('IHDR', ihdr),
    makeChunk('IDAT', idat),
    makeChunk('IEND', iend),
  ];

  const total = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(total);

  let p = 0;

  for (const c of chunks) {
    out.set(c, p);
    p += c.length;
  }

  await fs.writeFile(path, Buffer.from(out.buffer, out.byteOffset, out.byteLength));
}