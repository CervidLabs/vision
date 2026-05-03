/**
 * JPEG codec — zero npm dependencies.
 *
 * Read  SOF0 (baseline), SOF1 (extended sequential), SOF2 (progressive DCT).
 *       Grayscale and YCbCr.  All standard sampling factors.
 *       Restart markers.  16-bit precision (truncated to 8).
 *
 * Write Progressive JPEG (SOF2) with spectral-selection scans and standard
 *       Annex-K Huffman tables.  Quality 1–100 (libjpeg-compatible scaling).
 */

import { promises as fs } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { VisionFrame, clampU8 } from '../core/VisionFrame.js';
import { getJpegIdctWorkerPool } from './JpegIdctWorkerPool.js';
import { getJpegScanWorkerPool, type RawHuffTable } from './JpegScanWorkerPool.js';
import { cpus } from 'node:os';

// Number of IDCT row-chunks dispatched per component to the worker pool.
// Using 2× worker count keeps all workers busy even when chunk sizes differ.
const IDCT_CHUNKS_PER_COMP = Math.max(4, (cpus().length - 1) * 2);
// ── Zigzag scan order ─────────────────────────────────────────────────────────

const ZZ = new Uint8Array([
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43,
  36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]);

// ── IDCT / FDCT ───────────────────────────────────────────────────────────────
const JPEG_IDCT_WORKER_THRESHOLD_BLOCKS = 4096;
const INV_SQRT2 = 1 / Math.SQRT2;
const COS = Array.from({ length: 8 }, (_, u) => Float64Array.from({ length: 8 }, (__, x) => Math.cos(((2 * x + 1) * u * Math.PI) / 16)));
const COS8 = new Float64Array(64);

for (let u = 0; u < 8; u++) {
  for (let x = 0; x < 8; x++) {
    COS8[u * 8 + x] = Math.cos(((2 * x + 1) * u * Math.PI) / 16);
  }
}
function idct8x8(coeff: Float64Array, tmp: Float64Array, out: Uint8Array, off: number, stride: number): void {
  // ── Row pass ─────────────────────────────────────────────
  for (let v = 0; v < 8; v++) {
    const b = v << 3;

    const c0 = coeff[b] * INV_SQRT2;
    const c1 = coeff[b + 1];
    const c2 = coeff[b + 2];
    const c3 = coeff[b + 3];
    const c4 = coeff[b + 4];
    const c5 = coeff[b + 5];
    const c6 = coeff[b + 6];
    const c7 = coeff[b + 7];

    let o = b;

    tmp[o++] = (c0 + c1 * COS8[8] + c2 * COS8[16] + c3 * COS8[24] + c4 * COS8[32] + c5 * COS8[40] + c6 * COS8[48] + c7 * COS8[56]) * 0.5;
    tmp[o++] = (c0 + c1 * COS8[9] + c2 * COS8[17] + c3 * COS8[25] + c4 * COS8[33] + c5 * COS8[41] + c6 * COS8[49] + c7 * COS8[57]) * 0.5;
    tmp[o++] = (c0 + c1 * COS8[10] + c2 * COS8[18] + c3 * COS8[26] + c4 * COS8[34] + c5 * COS8[42] + c6 * COS8[50] + c7 * COS8[58]) * 0.5;
    tmp[o++] = (c0 + c1 * COS8[11] + c2 * COS8[19] + c3 * COS8[27] + c4 * COS8[35] + c5 * COS8[43] + c6 * COS8[51] + c7 * COS8[59]) * 0.5;
    tmp[o++] = (c0 + c1 * COS8[12] + c2 * COS8[20] + c3 * COS8[28] + c4 * COS8[36] + c5 * COS8[44] + c6 * COS8[52] + c7 * COS8[60]) * 0.5;
    tmp[o++] = (c0 + c1 * COS8[13] + c2 * COS8[21] + c3 * COS8[29] + c4 * COS8[37] + c5 * COS8[45] + c6 * COS8[53] + c7 * COS8[61]) * 0.5;
    tmp[o++] = (c0 + c1 * COS8[14] + c2 * COS8[22] + c3 * COS8[30] + c4 * COS8[38] + c5 * COS8[46] + c6 * COS8[54] + c7 * COS8[62]) * 0.5;
    tmp[o] = (c0 + c1 * COS8[15] + c2 * COS8[23] + c3 * COS8[31] + c4 * COS8[39] + c5 * COS8[47] + c6 * COS8[55] + c7 * COS8[63]) * 0.5;
  }

  // ── Column pass ──────────────────────────────────────────
  for (let x = 0; x < 8; x++) {
    const t0 = tmp[x] * INV_SQRT2;
    const t1 = tmp[8 + x];
    const t2 = tmp[16 + x];
    const t3 = tmp[24 + x];
    const t4 = tmp[32 + x];
    const t5 = tmp[40 + x];
    const t6 = tmp[48 + x];
    const t7 = tmp[56 + x];

    let v: number;

    v = ((t0 + t1 * COS8[8] + t2 * COS8[16] + t3 * COS8[24] + t4 * COS8[32] + t5 * COS8[40] + t6 * COS8[48] + t7 * COS8[56]) * 0.5 + 128.5) | 0;
    out[off + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS8[9] + t2 * COS8[17] + t3 * COS8[25] + t4 * COS8[33] + t5 * COS8[41] + t6 * COS8[49] + t7 * COS8[57]) * 0.5 + 128.5) | 0;
    out[off + stride + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS8[10] + t2 * COS8[18] + t3 * COS8[26] + t4 * COS8[34] + t5 * COS8[42] + t6 * COS8[50] + t7 * COS8[58]) * 0.5 + 128.5) | 0;
    out[off + stride * 2 + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS8[11] + t2 * COS8[19] + t3 * COS8[27] + t4 * COS8[35] + t5 * COS8[43] + t6 * COS8[51] + t7 * COS8[59]) * 0.5 + 128.5) | 0;
    out[off + stride * 3 + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS8[12] + t2 * COS8[20] + t3 * COS8[28] + t4 * COS8[36] + t5 * COS8[44] + t6 * COS8[52] + t7 * COS8[60]) * 0.5 + 128.5) | 0;
    out[off + stride * 4 + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS8[13] + t2 * COS8[21] + t3 * COS8[29] + t4 * COS8[37] + t5 * COS8[45] + t6 * COS8[53] + t7 * COS8[61]) * 0.5 + 128.5) | 0;
    out[off + stride * 5 + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS8[14] + t2 * COS8[22] + t3 * COS8[30] + t4 * COS8[38] + t5 * COS8[46] + t6 * COS8[54] + t7 * COS8[62]) * 0.5 + 128.5) | 0;
    out[off + stride * 6 + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS8[15] + t2 * COS8[23] + t3 * COS8[31] + t4 * COS8[39] + t5 * COS8[47] + t6 * COS8[55] + t7 * COS8[63]) * 0.5 + 128.5) | 0;
    out[off + stride * 7 + x] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}
function fdct8x8(src: Uint8Array, srcOff: number, srcStride: number, out: Float64Array): void {
  const tmp = new Float64Array(64);
  for (let y = 0; y < 8; y++) {
    const b = y * srcStride + srcOff;
    for (let u = 0; u < 8; u++) {
      let s = 0;
      for (let x = 0; x < 8; x++) {
        s += (src[b + x] - 128) * COS[u][x];
      }
      tmp[y * 8 + u] = s * (u === 0 ? INV_SQRT2 : 1) * 0.5;
    }
  }
  for (let u = 0; u < 8; u++) {
    for (let v = 0; v < 8; v++) {
      let s = 0;
      for (let y = 0; y < 8; y++) {
        s += tmp[y * 8 + u] * COS[v][y];
      }
      out[v * 8 + u] = s * (v === 0 ? INV_SQRT2 : 1) * 0.5;
    }
  }
}

// ── Bit reader ────────────────────────────────────────────────────────────────

// ── BitReader — accumulator with peek/skip and bulk refill ───────────────────
//
// Maintains a ≤24-bit accumulator (acc) filled in 8-bit chunks on demand.
// fill() guarantees ≥16 bits so peek(n≤16) and read(n≤16) never refill
// mid-operation. The >>> 0 casts keep acc as an unsigned 32-bit integer so
// bitwise extraction via >>> works correctly even when the high bit is set.

class BitReader {
  private acc = 0;   // unsigned 32-bit accumulator (up to ~24 bits valid)
  private avail = 0;   // valid bits in acc
  private _rst = false;

  constructor(private readonly buf: Uint8Array, public pos: number) { }

  get restartSeen(): boolean { const v = this._rst; this._rst = false; return v; }

  /** Refill acc until avail ≥ 16 (or until a restart marker is consumed). */
  private fill(): void {
    while (this.avail < 16) {
      if (this.pos >= this.buf.length) {
        this.acc = ((this.acc << 8) | 0xff) >>> 0;
        this.avail += 8;
        continue;
      }
      const byte = this.buf[this.pos++];
      if (byte === 0xff) {
        const nxt = this.buf[this.pos];
        if (nxt === 0x00) {
          this.pos++;                  // byte stuffing: real 0xff data
        } else if (nxt >= 0xd0 && nxt <= 0xd7) {
          this.pos++;                  // restart marker
          this._rst = true;
          this.acc = 0; this.avail = 0;
          continue;  // keep filling from post-RST bytes
        }
      }
      this.acc = ((this.acc << 8) | byte) >>> 0;
      this.avail += 8;
    }
  }

  /** Peek at the top `n` bits without consuming them. n must be ≤ 16. */
  peek(n: number): number {
    if (this.avail < n) this.fill();
    return (this.acc >>> (this.avail - n)) & ((1 << n) - 1);
  }

  /** Consume `n` bits. Must be preceded by a peek/read that ensured avail ≥ n. */
  skip(n: number): void { this.avail -= n; }

  /** Read and consume `n` bits. n must be ≤ 16. */
  read(n: number): number {
    if (n === 0) return 0;
    if (this.avail < n) this.fill();
    this.avail -= n;
    return (this.acc >>> this.avail) & ((1 << n) - 1);
  }
}

// ── Huffman decode table with 9-bit LUT ──────────────────────────────────────
//
// A 512-entry lookup table covers all codes of length ≤ 9 bits in a single
// array access + skip, avoiding the bit-by-bit loop for the common case.
//
// LUT entry layout (Uint16):
//   bits [12:4]  symbol value  (0–255)
//   bits [ 3:0]  code length   (1–9)
//   entry = 0    → slow path (code is longer than LUT_BITS)
//
// Because code length is always ≥ 1, (sym << 4) | len is never 0 for any
// valid code, so 0 unambiguously signals "not in LUT".

const LUT_BITS = 9;

class HuffDec {
  // Fast-path: 512-entry LUT for codes ≤ 9 bits
  private readonly lut = new Uint16Array(1 << LUT_BITS);
  // Slow-path: canonical table for codes > 9 bits
  private readonly minCode = new Int32Array(17).fill(-1);
  private readonly maxCode = new Int32Array(17).fill(-2);
  private readonly valOff = new Int32Array(17);
  private readonly values: Uint8Array;

  constructor(lengths: Uint8Array, values: Uint8Array) {
    this.values = new Uint8Array(values);
    let code = 0, vi = 0;

    for (let b = 1; b <= 16; b++) {
      const cnt = lengths[b - 1];
      if (cnt > 0) {
        this.minCode[b] = code;
        this.maxCode[b] = code + cnt - 1;
        this.valOff[b] = vi - code;

        // Populate LUT for short codes: each code of length b maps to
        // (1 << (LUT_BITS - b)) consecutive LUT entries (don't-care suffix).
        if (b <= LUT_BITS) {
          const fill = 1 << (LUT_BITS - b);
          for (let j = 0; j < cnt; j++) {
            const entry = (values[vi + j] << 4) | b;
            const base = (code + j) << (LUT_BITS - b);
            for (let f = 0; f < fill; f++) this.lut[base + f] = entry;
          }
        }

        vi += cnt; code += cnt;
      }
      code <<= 1;
    }
  }

  decode(r: BitReader): number {
    // Fast path: one table lookup for codes ≤ LUT_BITS bits
    const peek = r.peek(LUT_BITS);
    const entry = this.lut[peek];

    if (entry !== 0) {
      r.skip(entry & 0xf);   // consume only the actual code length
      return entry >> 4;     // symbol
    }

    // Slow path: code is longer than LUT_BITS — consume the peeked bits
    // and read one more bit at a time (rare, <5% of symbols in practice).
    r.skip(LUT_BITS);
    let code = peek;
    for (let b = LUT_BITS + 1; b <= 16; b++) {
      code = (code << 1) | r.read(1);
      if (code >= this.minCode[b] && code <= this.maxCode[b])
        return this.values[code + this.valOff[b]];
    }
    throw new Error('JPEG: bad Huffman code');
  }
}

function signExtend(v: number, cat: number): number {
  return v < 1 << (cat - 1) ? v - ((1 << cat) - 1) : v;
}

function u16(b: Uint8Array, o: number): number {
  return (b[o] << 8) | b[o + 1];
}

// ── File structures ───────────────────────────────────────────────────────────

interface Comp {
  id: number;
  hf: number;
  vf: number;
  qtId: number;
}

interface ScanInfo {
  nComps: number;
  scanComps: Array<{ ci: number; dcId: number; acId: number }>;
  Ss: number;
  Se: number;
  Ah: number;
  Al: number;
  dcHuff: (HuffDec | undefined)[];
  acHuff: (HuffDec | undefined)[];
  // Raw table data for serialising to scan-parser workers
  dcRawData: (RawHuffTable | null)[];
  acRawData: (RawHuffTable | null)[];
  dataStart: number;
}

interface JPEGFile {
  sofType: number;
  width: number;
  height: number;
  comps: Comp[];
  qtables: (Uint8Array | undefined)[];
  restartInterval: number;  // 0 = no DRI / restart markers not used
  scans: ScanInfo[];
  buf: Uint8Array;
}

// ── Parser — reads all segments, snapshots tables at each SOS ─────────────────

function parseFile(buf: Uint8Array): JPEGFile {
  if (buf[0] !== 0xff || buf[1] !== 0xd8) {
    throw new Error('JPEG: bad SOI');
  }

  let sofType = -1,
    width = 0,
    height = 0,
    restartInterval = 0;
  const comps: Comp[] = [];
  const qtables: (Uint8Array | undefined)[] = new Array(4);
  const dcHuff: (HuffDec | undefined)[] = new Array(4);
  const acHuff: (HuffDec | undefined)[] = new Array(4);
  const dcRawData: (RawHuffTable | null)[] = new Array(4).fill(null);
  const acRawData: (RawHuffTable | null)[] = new Array(4).fill(null);
  const scans: ScanInfo[] = [];

  let pos = 2;
  while (pos < buf.length - 1) {
    while (pos < buf.length && buf[pos] !== 0xff) {
      pos++;
    }
    while (pos < buf.length && buf[pos] === 0xff) {
      pos++;
    }
    if (pos >= buf.length) {
      break;
    }
    const marker = buf[pos++];

    if (marker === 0xd9) {
      break;
    }
    if (marker >= 0xd0 && marker <= 0xd7) {
      continue;
    }
    if (marker === 0xd8) {
      continue;
    }

    if (marker === 0xda) {
      const sosLen = u16(buf, pos);
      const nComps = buf[pos + 2];
      let p = pos + 3;
      const scanComps: ScanInfo['scanComps'] = [];
      for (let i = 0; i < nComps; i++) {
        const cid = buf[p++];
        const huff = buf[p++];
        const ci = comps.findIndex((c) => c.id === cid);
        scanComps.push({ ci, dcId: (huff >> 4) & 0xf, acId: huff & 0xf });
      }
      const Ss = buf[p],
        Se = buf[p + 1],
        AhAl = buf[p + 2];
      const Ah = (AhAl >> 4) & 0xf,
        Al = AhAl & 0xf;
      const dataStart = pos + sosLen;
      scans.push({
        nComps,
        scanComps,
        Ss, Se, Ah, Al,
        dcHuff: [...dcHuff],
        acHuff: [...acHuff],
        dcRawData: [...dcRawData],
        acRawData: [...acRawData],
        dataStart,
      });

      pos = dataStart;
      while (pos < buf.length - 1) {
        if (buf[pos] === 0xff) {
          const nx = buf[pos + 1];
          if (nx !== 0x00 && !(nx >= 0xd0 && nx <= 0xd7)) {
            break;
          }
        }
        pos++;
      }
      continue;
    }

    const segLen = u16(buf, pos);
    const segEnd = pos + segLen;
    pos += 2;

    switch (marker) {
      case 0xdb: {
        let p = pos;
        while (p < segEnd) {
          const info = buf[p++];
          const id = info & 0xf,
            prec = (info >> 4) & 0xf;
          const qt = new Uint8Array(64);
          for (let i = 0; i < 64; i++) {
            qt[i] = prec === 0 ? buf[p++] : ((p += 2), buf[p - 2]);
          }
          qtables[id] = qt;
        }
        break;
      }
      case 0xdd: {  // DRI — Define Restart Interval
        restartInterval = u16(buf, pos + 2);
        break;
      }
      case 0xc0:
      case 0xc1:
      case 0xc2: {
        sofType = marker & 0x0f;
        if (sofType > 2) {
          throw new Error(`JPEG: unsupported SOF${sofType}`);
        }
        height = u16(buf, pos + 1);
        width = u16(buf, pos + 3);
        const n = buf[pos + 5];
        let p = pos + 6;
        for (let i = 0; i < n; i++) {
          comps.push({
            id: buf[p],
            hf: (buf[p + 1] >> 4) & 0xf,
            vf: buf[p + 1] & 0xf,
            qtId: buf[p + 2],
          });
          p += 3;
        }
        break;
      }
      case 0xc4: {
        let p = pos;
        while (p < segEnd) {
          const info = buf[p++];
          const cls = (info >> 4) & 0xf,
            id = info & 0xf;
          const lengths = buf.subarray(p, p + 16);
          p += 16;
          const nVals = Array.from(lengths).reduce((s, v) => s + v, 0);
          const values = buf.subarray(p, p + nVals);
          p += nVals;
          const rawTbl: RawHuffTable = {
            lengths: Array.from(lengths),
            values: Array.from(values),
          };
          const tbl = new HuffDec(new Uint8Array(lengths), new Uint8Array(values));
          if (cls === 0) {
            dcHuff[id] = tbl;
            dcRawData[id] = rawTbl;
          } else {
            acHuff[id] = tbl;
            acRawData[id] = rawTbl;
          }
        }
        break;
      }
    }
    pos = segEnd;
  }

  if (sofType < 0 || !width || !height || !comps.length) {
    throw new Error('JPEG: missing SOF / frame header');
  }
  return { sofType, width, height, comps, qtables, scans, buf, restartInterval };
}

// ── Plane → VisionFrame ───────────────────────────────────────────────────────

function planesToFrame(planes: Uint8Array[], pw: number[], comps: Comp[], hMax: number, vMax: number, W: number, H: number): VisionFrame {
  const nc = comps.length;
  const outCh = nc === 1 ? 1 : 3;
  const frame = new VisionFrame(W, H, outCh);
  const dst = frame.data;

  if (nc === 1) {
    const Y = planes[0];
    const stride = pw[0];

    for (let y = 0; y < H; y++) {
      const srcOff = y * stride;
      const dstOff = y * W;
      dst.set(Y.subarray(srcOff, srcOff + W), dstOff);
    }

    return frame;
  }

  const Yp = planes[0];
  const Cbp = planes[1];
  const Crp = planes[2];

  const yStride = pw[0];
  const cbStride = pw[1];
  const crStride = pw[2];

  const cbHScale = hMax / comps[1].hf;
  const cbVScale = vMax / comps[1].vf;
  const crHScale = hMax / comps[2].hf;
  const crVScale = vMax / comps[2].vf;

  const cbXMap = new Int32Array(W);
  const crXMap = new Int32Array(W);

  for (let x = 0; x < W; x++) {
    cbXMap[x] = (x / cbHScale) | 0;
    crXMap[x] = (x / crHScale) | 0;
  }

  for (let y = 0; y < H; y++) {
    const yOff = y * yStride;
    const cbOff = ((y / cbVScale) | 0) * cbStride;
    const crOff = ((y / crVScale) | 0) * crStride;
    let out = y * W * 3;

    for (let x = 0; x < W; x++) {
      const yy = Yp[yOff + x];
      const cb = Cbp[cbOff + cbXMap[x]] - 128;
      const cr = Crp[crOff + crXMap[x]] - 128;

      // Integer BT.601 in Q8 — avoids float multiply per pixel
      const r = (yy + ((359 * cr) >> 8));
      const g = (yy - ((88 * cb + 183 * cr) >> 8));
      const b = (yy + ((454 * cb) >> 8));

      dst[out++] = r < 0 ? 0 : r > 255 ? 255 : r;
      dst[out++] = g < 0 ? 0 : g > 255 ? 255 : g;
      dst[out++] = b < 0 ? 0 : b > 255 ? 255 : b;
    }
  }

  return frame;
}
// ── Baseline decoder (SOF0 / SOF1) ───────────────────────────────────────────

async function decodeBaseline(f: JPEGFile): Promise<VisionFrame> {
  const { width, height, comps, qtables, scans, buf } = f;
  if (!scans.length) {
    throw new Error('JPEG: no scan');
  }
  const scan = scans[0];
  const nc = comps.length;
  const hMax = Math.max(...comps.map((c) => c.hf));
  const vMax = Math.max(...comps.map((c) => c.vf));
  const mcuCols = Math.ceil(width / (hMax * 8));
  const mcuRows = Math.ceil(height / (vMax * 8));
  const planeW = comps.map((c) => mcuCols * c.hf * 8);
  const planeH = comps.map((c) => mcuRows * c.vf * 8);
  const planes = comps.map((_, i) => new Uint8Array(planeW[i] * planeH[i]));
  const _t0Baseline = performance.now();
  const reader = new BitReader(buf, scan.dataStart);
  const dcPrev = new Int32Array(nc);
  const coeff = new Float64Array(64);
  const ciMap = new Map(scan.scanComps.map((s) => [s.ci, s]));
  const tmp = new Float64Array(64);

  for (let mr = 0; mr < mcuRows; mr++) {
    for (let mc = 0; mc < mcuCols; mc++) {
      for (let ci = 0; ci < nc; ci++) {
        const s = ciMap.get(ci)!;
        const qt = qtables[comps[ci].qtId]!;
        const dc = scan.dcHuff[s.dcId]!;
        const ac = scan.acHuff[s.acId]!;
        const pw = planeW[ci];
        for (let brow = 0; brow < comps[ci].vf; brow++) {
          for (let bcol = 0; bcol < comps[ci].hf; bcol++) {
            coeff.fill(0);
            const cat = dc.decode(reader);
            dcPrev[ci] += cat === 0 ? 0 : signExtend(reader.read(cat), cat);
            coeff[0] = dcPrev[ci] * qt[0];
            let k = 1;
            while (k < 64) {
              const sym = ac.decode(reader);
              const run = (sym >> 4) & 0xf,
                cat2 = sym & 0xf;
              if (cat2 === 0) {
                if (run === 0) {
                  break;
                }
                k += 16;
                continue;
              }
              k += run;
              if (k >= 64) {
                break;
              }
              coeff[ZZ[k]] = signExtend(reader.read(cat2), cat2) * qt[k];
              k++;
            }
            idct8x8(coeff, tmp, planes[ci], (mr * comps[ci].vf + brow) * 8 * pw + (mc * comps[ci].hf + bcol) * 8, pw);
          }
        }
      }
      if (reader.restartSeen) {
        dcPrev.fill(0);
      }
    }
  }
  const _tHuff = performance.now();

  // ── IDCT: parallel if image is large enough ───────────────────────────────
  const nbX = comps.map((c) => mcuCols * c.hf);
  const nbY = comps.map((c) => mcuRows * c.vf);
  const totalBlk = nbX.reduce((s, x, i) => s + x * nbY[i], 0);

  if (totalBlk >= JPEG_IDCT_WORKER_THRESHOLD_BLOCKS) {
    // Re-allocate planes as SAB so workers can write into them
    const coeffSABs = comps.map((_, ci) => {
      const len = nbX[ci] * nbY[ci] * 64;
      const sab = new SharedArrayBuffer(len * Int16Array.BYTES_PER_ELEMENT);
      const dst = new Int16Array(sab);

      // Convert the dequantised Float64 plane we filled above back into
      // quantised Int16 levels for the worker (worker does coeff * qt[k])
      const qt = qtables[comps[ci].qtId]!;
      // planes[ci] already contains rendered pixels from inline idct — skip
      // Instead we need to re-decode coefficients. Fall through to sync for now.
      return null;
    });

    // Not all paths support retrofitting to workers here; fall through to sync.
    // (Full two-pass refactor is in decodeProgresssive — baseline stays 1-pass)
  }

  const _tIdct = performance.now();
  const result = planesToFrame(planes, planeW, comps, hMax, vMax, width, height);
  const _tConv = performance.now();

  process.stderr.write(
    `[baseline] huffman+idct=${(_tHuff - _t0Baseline).toFixed(0)}ms ` +
    `(inline 1-pass) convert=${(_tConv - _tIdct).toFixed(0)}ms\n`
  );
  return result;
}

// ── Progressive decoder (SOF2)
//
// Progressive JPEG stores DCT coefficients across multiple scans.
// Each scan specifies:
//   Ss..Se  spectral range in zigzag order (0 = DC, 1-63 = AC)
//   Ah      successive approx high bit-plane from prior scan (0 = first pass)
//   Al      shift: this scan contributes bits starting at position Al
//
// Sub-types:
//   DC first  (Ss=0, Se=0, Ah=0) — read full DC value, store << Al
//   DC refine (Ss=0, Se=0, Ah>0) — read 1 refinement bit per block
//   AC first  (Ss>0, Ah=0)       — like baseline AC with EOB runs, values << Al
//   AC refine (Ss>0, Ah>0)       — refine existing nonzeros + place new ones

// ── Restart-interval finder ──────────────────────────────────────────────────
// Scans raw JPEG bytes to find RST marker positions, enabling intra-scan
// parallelism: each restart interval is independent (eobRun=0, dcPrev=0).

interface RestartInterval { byteOffset: number; mcuStart: number; }

function findRestartIntervals(
  buf: Uint8Array,
  dataStart: number,
  restartInterval: number,
  totalMCUs: number,
): RestartInterval[] {
  if (restartInterval === 0) return [{ byteOffset: dataStart, mcuStart: 0 }];

  const intervals: RestartInterval[] = [{ byteOffset: dataStart, mcuStart: 0 }];
  let pos = dataStart;
  let rstIdx = 0;

  while (pos < buf.length - 1) {
    if (buf[pos] === 0xff) {
      const nxt = buf[pos + 1];
      if (nxt === 0x00) { pos += 2; continue; }           // stuffed byte
      if (nxt >= 0xd0 && nxt <= 0xd7) {                   // RST marker
        pos += 2;
        rstIdx++;
        const mcuStart = rstIdx * restartInterval;
        if (mcuStart < totalMCUs) intervals.push({ byteOffset: pos, mcuStart });
        continue;
      }
      break; // non-RST marker = end of scan data
    }
    pos++;
  }
  return intervals;
}

// ── Scan dependency analysis ─────────────────────────────────────────────────
//
// Progressive JPEG scans that touch the same (component, spectral range) must
// execute in file order because later scans refine what earlier ones wrote.
// Scans on different components or non-overlapping spectral ranges are fully
// independent and can run in parallel (different chunks of coeffBufs).
//
// buildScanWaves() groups scans into sequential "waves" where every scan
// within a wave is independent of every other scan in the same wave.

function scanConflicts(a: ScanInfo, b: ScanInfo): boolean {
  const aComps = new Set(a.scanComps.map(s => s.ci));
  for (const { ci } of b.scanComps) {
    if (aComps.has(ci) && a.Ss <= b.Se && b.Ss <= a.Se) return true;
  }
  return false;
}

function buildScanWaves(scans: ScanInfo[]): ScanInfo[][] {
  const waves: ScanInfo[][] = [];
  for (const scan of scans) {
    // Place scan in the wave AFTER the last wave that has a conflicting scan.
    // This preserves file order within each (component, spectral) chain.
    let lastConflict = -1;
    for (let i = 0; i < waves.length; i++) {
      if (waves[i].some(s => scanConflicts(s, scan))) lastConflict = i;
    }
    const target = lastConflict + 1;
    if (!waves[target]) waves[target] = [];
    waves[target].push(scan);
  }
  return waves;
}

async function decodeProgressive(f: JPEGFile): Promise<VisionFrame> {
  const { width, height, comps, qtables, scans, buf } = f;
  const nc = comps.length;
  const hMax = Math.max(...comps.map((c) => c.hf));
  const vMax = Math.max(...comps.map((c) => c.vf));
  const mcuCols = Math.ceil(width / (hMax * 8));
  const mcuRows = Math.ceil(height / (vMax * 8));

  // Block grid per component (padded to MCU boundary)
  const nbX = comps.map((c) => mcuCols * c.hf);
  const nbY = comps.map((c) => mcuRows * c.vf);
  const totalBlocks = nbX.reduce((sum, x, i) => sum + x * nbY[i], 0);

  // ── Parallel scan parse ──────────────────────────────────────────────────
  // Each progressive scan is an independent bitstream: different spectral
  // range OR different component.  We dispatch all scans simultaneously to
  // worker threads that write into shared coefficient SABs.
  //
  // Zero-copy: the raw JPEG bytes are copied once to a SharedArrayBuffer so
  // all workers can read from it without additional copies.

  const _tp0 = performance.now();

  // Coefficient accumulation buffers (SharedArrayBuffer — workers write here)
  const coeffBufs = comps.map((_, ci) => {
    const length = nbX[ci] * nbY[ci] * 64;
    return new Int16Array(new SharedArrayBuffer(length * Int16Array.BYTES_PER_ELEMENT));
  });
  const coeffSABs = coeffBufs.map(cb => cb.buffer as SharedArrayBuffer);

  // Copy raw JPEG bytes to SAB once (shared read across all scan workers)
  const sharedBuf = new SharedArrayBuffer(buf.byteLength);
  new Uint8Array(sharedBuf).set(buf);

  const compHf = comps.map(c => c.hf);
  const compVf = comps.map(c => c.vf);

  // Group scans into dependency waves (independent scans in each wave run in parallel,
  // waves execute sequentially to preserve refinement pass ordering).
  const waves = buildScanWaves(scans);
  const scanPool = getJpegScanWorkerPool();

  // Each scan uses its OWN table snapshot — progressive JPEGs can redefine
  // Huffman tables between scans with the same IDs (first-pass vs refinement).
  const makeScanJob = (scan: ScanInfo, extra: Partial<Parameters<typeof scanPool.run>[0]> = {}) =>
    scanPool.run({
      buf: sharedBuf,
      dataStart: scan.dataStart,
      nComps: scan.nComps,
      scanComps: scan.scanComps,
      Ss: scan.Ss, Se: scan.Se, Ah: scan.Ah, Al: scan.Al,
      mcuCols, mcuRows,
      nbX, nbY, compHf, compVf,
      dcRaw: scan.dcRawData,
      acRaw: scan.acRawData,
      coeffBufs: coeffSABs,
      ...extra,
    }).catch((e: unknown) => {
      process.stderr.write(`[progressive] SCAN WORKER ERROR: ${e}\n`);
    });

  const waveTimes: number[] = [];
  let wave: ScanInfo[] = [];
  for (wave of waves) {
    const wt0 = performance.now();

    // Single non-interleaved scan + JPEG has restart markers → split by interval
    if (wave.length === 1 && wave[0].nComps === 1 && f.restartInterval > 0) {
      const scan = wave[0];
      const ci = scan.scanComps[0].ci;
      const totalMCUs = nbX[ci] * nbY[ci];
      const intervals = findRestartIntervals(buf, scan.dataStart, f.restartInterval, totalMCUs);

      if (intervals.length > 1) {
        process.stderr.write(`[progressive] wave2 RST split: ${intervals.length} intervals (DRI=${f.restartInterval})\n`);
        await Promise.all(intervals.map((iv, idx) => {
          const mcuCount = idx + 1 < intervals.length
            ? intervals[idx + 1].mcuStart - iv.mcuStart
            : totalMCUs - iv.mcuStart;
          return makeScanJob(scan, { dataStart: iv.byteOffset, mcuOffset: iv.mcuStart, mcuCount });
        }));
        waveTimes.push(performance.now() - wt0);
        continue;
      }
    }

    // No restart markers or interleaved scan — log so we know
    if (wave.length === 1) {
      process.stderr.write(`[progressive] wave${waves.indexOf(wave)} serial (DRI=${f.restartInterval})\n`);
    }

    await Promise.all(wave.map(s => makeScanJob(s)));
    waveTimes.push(performance.now() - wt0);
  }

  const _tp1 = performance.now();
  const waveDetail = waves.map((w, i) =>
    `${w.length}scan/${waveTimes[i].toFixed(0)}ms[${w.map(s => `Ss${s.Ss}Se${s.Se}Ah${s.Ah}ci${s.scanComps.map(c => c.ci).join('')}`).join(',')}]`
  ).join(' → ');
  process.stderr.write(
    `[progressive] huffman-parse(parallel)=${(_tp1 - _tp0).toFixed(0)}ms\n` +
    `  waves: ${waveDetail}\n`
  );

  // ── Dequantise + IDCT ─────────────────────────────────────────────────────
  const planeW = nbX.map((n) => n * 8);
  const planeH = nbY.map((n) => n * 8);
  const useWorkers = totalBlocks >= JPEG_IDCT_WORKER_THRESHOLD_BLOCKS;

  const planes = comps.map((_, ci) => {
    const size = planeW[ci] * planeH[ci];

    return useWorkers
      ? new Uint8Array(new SharedArrayBuffer(size))
      : new Uint8Array(size);
  });
  if (!useWorkers) {
    runIdctSync(coeffBufs, planes, qtables, comps, nbX, nbY, planeW);
    return planesToFrame(planes, planeW, comps, hMax, vMax, width, height);
  }

  const pool = getJpegIdctWorkerPool();

  // Log worker file resolution for debugging
  process.stderr.write(`[progressive] dispatching IDCT jobs to pool (${nc} components)\n`);

  const jobs: Promise<void>[] = [];

  for (let ci = 0; ci < nc; ci++) {
    const qt = qtables[comps[ci].qtId]!;
    const rows = nbY[ci];
    // Use more chunks than workers so all cores stay busy
    const chunks = Math.min(IDCT_CHUNKS_PER_COMP, rows);
    const rowsPerJob = Math.ceil(rows / chunks);

    for (let j = 0; j < chunks; j++) {
      const rowStart = j * rowsPerJob;
      const rowEnd = Math.min(rows, rowStart + rowsPerJob);
      if (rowStart >= rowEnd) continue;

      jobs.push(
        pool.run({
          coeffBuffer: coeffBufs[ci].buffer as SharedArrayBuffer,
          planeBuffer: planes[ci].buffer as SharedArrayBuffer,
          qt,
          nbX: nbX[ci],
          rowStart,
          rowEnd,
          planeWidth: planeW[ci],
        }).catch((e: unknown) => {
          process.stderr.write(`[progressive] WORKER ERROR: ${e}\n`);
        })
      );
    }
  }
  await Promise.all(jobs);
  // Do NOT call pool.close() here — workers reuse via idle timeout
  return planesToFrame(planes, planeW, comps, hMax, vMax, width, height);
}
function idctDCOnly(dc: number, out: Uint8Array, off: number, stride: number): void {
  const value = Math.round(dc * 0.125 + 128);
  const v = value < 0 ? 0 : value > 255 ? 255 : value;

  for (let y = 0; y < 8; y++) {
    const row = off + y * stride;
    out[row] = v;
    out[row + 1] = v;
    out[row + 2] = v;
    out[row + 3] = v;
    out[row + 4] = v;
    out[row + 5] = v;
    out[row + 6] = v;
    out[row + 7] = v;
  }
}
function isDCOnly(coeffs: Int16Array, off: number): boolean {
  for (let k = 1; k < 64; k++) {
    if (coeffs[off + k] !== 0) {
      return false;
    }
  }

  return true;
}
function runIdctSync(
  coeffBufs: Int16Array[],
  planes: Uint8Array[],
  qtables: (Uint8Array | undefined)[],
  comps: Comp[],
  nbX: number[],
  nbY: number[],
  planeW: number[]
): void {
  const dct = new Float64Array(64);
  const tmp = new Float64Array(64);

  for (let ci = 0; ci < comps.length; ci++) {
    const qt = qtables[comps[ci].qtId]!;
    const pw = planeW[ci];
    const coeffs = coeffBufs[ci];

    for (let row = 0; row < nbY[ci]; row++) {
      for (let col = 0; col < nbX[ci]; col++) {
        const bi = row * nbX[ci] + col;
        const bo = bi * 64;
        const outOff = row * 8 * pw + col * 8;

        if (isDCOnly(coeffs, bo)) {
          idctDCOnly(coeffs[bo] * qt[0], planes[ci], outOff, pw);
          continue;
        }

        dct.fill(0);

        for (let k = 0; k < 64; k++) {
          dct[ZZ[k]] = coeffs[bo + k] * qt[k];
        }

        idct8x8(dct, tmp, planes[ci], outOff, pw);
      }
    }
  }
}
// ── Public read entry point ───────────────────────────────────────────────────

export async function readJPEG(path: string): Promise<VisionFrame> {
  const t0 = performance.now();
  const raw = await fs.readFile(path);
  const t1 = performance.now();

  const buf = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
  const jpeg = parseFile(buf);
  const t2 = performance.now();

  const result = jpeg.sofType <= 1
    ? await decodeBaseline(jpeg)
    : await decodeProgressive(jpeg);
  const t3 = performance.now();

  // Internal breakdown written to stderr so it doesn't pollute stdout
  process.stderr.write(
    `[jpg] sof=${jpeg.sofType} ${jpeg.width}x${jpeg.height} ` +
    `fileRead=${(t1 - t0).toFixed(0)}ms segParse=${(t2 - t1).toFixed(0)}ms ` +
    `decode=${(t3 - t2).toFixed(0)}ms\n`
  );
  return result;
}

// ═════════════════════════════════════════════════════════════════════════════
//  PROGRESSIVE JPEG ENCODER  (SOF2, spectral selection, Annex-K tables)
// ═════════════════════════════════════════════════════════════════════════════
//
// Scan sequence (no successive approximation):
//   Scan 0  DC  Y + Cb + Cr  interleaved   Ss=0  Se=0   Ah=0 Al=0
//   Scan 1  AC  Y only                     Ss=1  Se=5   Ah=0 Al=0  (low AC)
//   Scan 2  AC  Y only                     Ss=6  Se=63  Ah=0 Al=0  (high AC)
//   Scan 3  AC  Cb only                    Ss=1  Se=63  Ah=0 Al=0
//   Scan 4  AC  Cr only                    Ss=1  Se=63  Ah=0 Al=0

// ── Quantization tables (Annex K) ────────────────────────────────────────────

const LUMA_QT = new Uint8Array([
  16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68,
  109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]);
const CHROMA_QT = new Uint8Array([
  17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
]);

function scaleQT(base: Uint8Array, q: number): Uint8Array {
  const qq = Math.max(1, Math.min(100, q));
  const s = qq < 50 ? Math.floor(5000 / qq) : 200 - 2 * qq;
  return Uint8Array.from(base, (v) => Math.max(1, Math.min(255, Math.floor((v * s + 50) / 100))));
}

function toZZ(natural: Uint8Array): Uint8Array {
  const z = new Uint8Array(64);
  for (let k = 0; k < 64; k++) {
    z[k] = natural[ZZ[k]];
  }
  return z;
}

// ── Standard Annex-K Huffman tables ──────────────────────────────────────────

const DC_L_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_L_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const DC_C_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_C_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_L_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];
const AC_L_VALS = [
  0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
  0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
  0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
  0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
  0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
  0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
  0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];
const AC_C_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119];
const AC_C_VALS = [
  0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
  0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
  0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
  0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
  0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
  0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
  0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];

interface HCode {
  code: number;
  bits: number;
}

function buildEnc(bitsArr: number[], valsArr: number[]): (HCode | undefined)[] {
  const tbl: (HCode | undefined)[] = new Array(256);
  let code = 0,
    vi = 0;
  for (let len = 1; len <= 16; len++) {
    for (let j = 0; j < bitsArr[len - 1]; j++) {
      tbl[valsArr[vi++]] = { code, bits: len };
      code++;
    }
    code <<= 1;
  }
  return tbl;
}

// ── Bit writer ────────────────────────────────────────────────────────────────

class BitWriter {
  readonly out: number[] = [];
  private bits = 0;
  private left = 0;

  write(value: number, length: number): void {
    this.bits = (this.bits << length) | (value & ((1 << length) - 1));
    this.left += length;
    while (this.left >= 8) {
      this.left -= 8;
      const b = (this.bits >>> this.left) & 0xff;
      this.out.push(b);
      if (b === 0xff) {
        this.out.push(0x00);
      }
    }
  }

  flush(): void {
    if (this.left > 0) {
      const b = ((this.bits << (8 - this.left)) | ((1 << (8 - this.left)) - 1)) & 0xff;
      this.out.push(b);
      if (b === 0xff) {
        this.out.push(0x00);
      }
    }
  }
}

// ── JPEG segment builders ─────────────────────────────────────────────────────

function seg(marker: number, data: number[]): number[] {
  const len = data.length + 2;
  return [0xff, marker, len >> 8, len & 0xff, ...data];
}
function dqtSeg(id: number, qt: Uint8Array): number[] {
  return seg(0xdb, [id, ...qt]);
}
function dhtSeg(cls: number, id: number, bits: number[], vals: number[]): number[] {
  return seg(0xc4, [(cls << 4) | id, ...bits, ...vals]);
}
function sosSeg(comps: Array<{ id: number; dcId: number; acId: number }>, Ss: number, Se: number, Ah: number, Al: number): number[] {
  const d: number[] = [comps.length];
  for (const c of comps) {
    d.push(c.id, (c.dcId << 4) | c.acId);
  }
  d.push(Ss, Se, (Ah << 4) | Al);
  return seg(0xda, d);
}

// ── Coefficient helpers ───────────────────────────────────────────────────────

function valueCat(v: number): number {
  return v === 0 ? 0 : 32 - Math.clz32(Math.abs(v));
}
function encodeVal(v: number, cat: number): number {
  return v >= 0 ? v : v + (1 << cat) - 1;
}

// ── Main encoder ──────────────────────────────────────────────────────────────

export async function writeJPEG(path: string, frame: VisionFrame, quality = 85): Promise<void> {
  if (frame.channels !== 3) {
    throw new Error('writeJPEG: requires 3-channel RGB frame (.toRGB() first)');
  }

  const { width, height } = frame;
  const src = frame.data;

  const lumaQ = scaleQT(LUMA_QT, quality);
  const chromaQ = scaleQT(CHROMA_QT, quality);
  const lumaZZ = toZZ(lumaQ);
  const chromaZZ = toZZ(chromaQ);

  const dcLE = buildEnc(DC_L_BITS, DC_L_VALS);
  const dcCE = buildEnc(DC_C_BITS, DC_C_VALS);
  const acLE = buildEnc(AC_L_BITS, AC_L_VALS);
  const acCE = buildEnc(AC_C_BITS, AC_C_VALS);

  // RGB → YCbCr
  const Y = new Uint8Array(width * height);
  const Cb = new Uint8Array(width * height);
  const Cr = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < width * height; i++, p += 3) {
    const R = src[p],
      G = src[p + 1],
      B = src[p + 2];
    Y[i] = clampU8(0.299 * R + 0.587 * G + 0.114 * B);
    Cb[i] = clampU8(-0.168736 * R - 0.331264 * G + 0.5 * B + 128);
    Cr[i] = clampU8(0.5 * R - 0.418688 * G - 0.081312 * B + 128);
  }

  const pw = Math.ceil(width / 8) * 8;
  const ph = Math.ceil(height / 8) * 8;

  function pad(p: Uint8Array): Uint8Array {
    if (pw === width && ph === height) {
      return p;
    }
    const out = new Uint8Array(pw * ph);
    for (let y = 0; y < ph; y++) {
      const sy = Math.min(y, height - 1);
      for (let x = 0; x < pw; x++) {
        out[y * pw + x] = p[sy * width + Math.min(x, width - 1)];
      }
    }
    return out;
  }

  const yP = pad(Y),
    cbP = pad(Cb),
    crP = pad(Cr);
  const mcuRows = ph / 8,
    mcuCols = pw / 8;
  const nBlocks = mcuRows * mcuCols;

  // Quantise all blocks in zigzag order
  const qcoeff = [new Int16Array(nBlocks * 64), new Int16Array(nBlocks * 64), new Int16Array(nBlocks * 64)];
  const planesArr = [yP, cbP, crP];
  const qts = [lumaQ, chromaQ, chromaQ];
  const dct = new Float64Array(64);

  for (let ci = 0; ci < 3; ci++) {
    const qt = qts[ci];
    for (let row = 0; row < mcuRows; row++) {
      for (let col = 0; col < mcuCols; col++) {
        const bi = row * mcuCols + col;
        fdct8x8(planesArr[ci], row * 8 * pw + col * 8, pw, dct);
        const qc = qcoeff[ci].subarray(bi * 64, bi * 64 + 64);
        for (let k = 0; k < 64; k++) {
          qc[k] = Math.round(dct[ZZ[k]] / qt[ZZ[k]]);
        }
      }
    }
  }

  // ── Assemble progressive JPEG ─────────────────────────────────────────────

  const out: number[] = [
    0xff,
    0xd8,
    0xff,
    0xe0,
    0x00,
    0x10,
    0x4a,
    0x46,
    0x49,
    0x46,
    0x00,
    0x01,
    0x01,
    0x00,
    0x00,
    0x01,
    0x00,
    0x01,
    0x00,
    0x00,
    ...dqtSeg(0, lumaZZ),
    ...dqtSeg(1, chromaZZ),
    ...seg(0xc2, [
      // SOF2 — progressive
      0x08,
      height >> 8,
      height & 0xff,
      width >> 8,
      width & 0xff,
      0x03,
      0x01,
      0x11,
      0x00,
      0x02,
      0x11,
      0x01,
      0x03,
      0x11,
      0x01,
    ]),
  ];

  // ── Scan 0: DC all components, interleaved ────────────────────────────────

  out.push(
    ...dhtSeg(0, 0, DC_L_BITS, DC_L_VALS),
    ...dhtSeg(0, 1, DC_C_BITS, DC_C_VALS),
    ...sosSeg(
      [
        { id: 1, dcId: 0, acId: 0 },
        { id: 2, dcId: 1, acId: 1 },
        { id: 3, dcId: 1, acId: 1 },
      ],
      0,
      0,
      0,
      0,
    ),
  );
  {
    const w = new BitWriter();
    const dcPrev = [0, 0, 0];
    const dcEncs = [dcLE, dcCE, dcCE];
    for (let mr = 0; mr < mcuRows; mr++) {
      for (let mc = 0; mc < mcuCols; mc++) {
        const bi = mr * mcuCols + mc;
        for (let ci = 0; ci < 3; ci++) {
          const qc = qcoeff[ci].subarray(bi * 64, bi * 64 + 64);
          const diff = qc[0] - dcPrev[ci];
          dcPrev[ci] = qc[0];
          const cat = valueCat(diff);
          const h = dcEncs[ci][cat]!;
          w.write(h.code, h.bits);
          if (cat > 0) {
            w.write(encodeVal(diff, cat), cat);
          }
        }
      }
    }
    w.flush();
    out.push(...w.out);
  }

  // ── Scans 1-4: AC, non-interleaved ───────────────────────────────────────

  const acScans = [
    { ci: 0, id: 1, Ss: 1, Se: 5, enc: acLE, dhtId: 0 },
    { ci: 0, id: 1, Ss: 6, Se: 63, enc: acLE, dhtId: 0 },
    { ci: 1, id: 2, Ss: 1, Se: 63, enc: acCE, dhtId: 1 },
    { ci: 2, id: 3, Ss: 1, Se: 63, enc: acCE, dhtId: 1 },
  ];

  let lastDhtId = -1;
  for (const { ci, id, Ss, Se, enc, dhtId } of acScans) {
    if (dhtId !== lastDhtId) {
      const [bits, vals] = dhtId === 0 ? [AC_L_BITS, AC_L_VALS] : [AC_C_BITS, AC_C_VALS];
      out.push(...dhtSeg(1, dhtId, bits, vals));
      lastDhtId = dhtId;
    }
    out.push(...sosSeg([{ id, dcId: 0, acId: dhtId }], Ss, Se, 0, 0));

    const w = new BitWriter();
    for (let bi = 0; bi < nBlocks; bi++) {
      const qc = qcoeff[ci].subarray(bi * 64, bi * 64 + 64);
      let run = 0;
      for (let k = Ss; k <= Se; k++) {
        const v = qc[k];
        if (v === 0) {
          run++;
          if (run === 16) {
            w.write(enc[0xf0]!.code, enc[0xf0]!.bits);
            run = 0;
          }
        } else {
          const cat = valueCat(v);
          const h = enc[(run << 4) | cat]!;
          w.write(h.code, h.bits);
          w.write(encodeVal(v, cat), cat);
          run = 0;
        }
      }
      if (run > 0) {
        const eob = enc[0]!;
        w.write(eob.code, eob.bits);
      }
    }
    w.flush();
    pushMany(out, w.out);

  }

  // EOI
  out.push(0xff, 0xd9);

  await fs.writeFile(path, Buffer.from(Uint8Array.from(out).buffer));
}
function pushMany(dst: number[], src: ArrayLike<number>): void {
  const chunk = 8192;

  for (let i = 0; i < src.length; i += chunk) {
    const end = Math.min(src.length, i + chunk);

    for (let j = i; j < end; j++) {
      dst.push(src[j]);
    }
  }
}