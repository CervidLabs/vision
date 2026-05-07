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
import { VisionFrame } from '../core/VisionFrame.js';
import { getJpegIdctWorkerPool } from './JpegIdctWorkerPool.js';
// ── Zigzag scan order ─────────────────────────────────────────────────────────
export interface JPEGReadOptions {
  resize?: {
    width?: number;
    height?: number;
    method?: 'nearest' | 'bilinear' | 'area';
    shrinkOnLoad?: boolean;
  };
}
const ZZ = new Uint8Array([
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43,
  36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]);

// ── IDCT / FDCT ───────────────────────────────────────────────────────────────
const JPEG_IDCT_WORKER_THRESHOLD_BLOCKS = 4096;
const INV_SQRT2 = 1 / Math.SQRT2;
const COS8 = new Float64Array(64);
const COS4 = new Float64Array(32);

for (let u = 0; u < 8; u++) {
  for (let x = 0; x < 4; x++) {
    // Sampleamos centros de pares de pixeles del bloque 8×8.
    // x8 = 2x + 0.5 => cos(((2*x8 + 1)uπ)/16) = cos(((4x + 2)uπ)/16)
    COS4[u * 4 + x] = Math.cos(((4 * x + 2) * u * Math.PI) / 16);
  }
}
for (let u = 0; u < 8; u++) {
  for (let x = 0; x < 8; x++) {
    COS8[u * 8 + x] = Math.cos(((2 * x + 1) * u * Math.PI) / 16);
  }
}
function idct8x8To4x4(coeff: Float64Array, tmp: Float64Array, out: Uint8Array, off: number, stride: number): void {
  // tmp necesita length 32: 8 filas × 4 columnas.
  // Row pass: 8 frecuencias horizontales → 4 samples.
  for (let v = 0; v < 8; v++) {
    const b = v << 3;
    const tb = v << 2;

    const c0 = coeff[b] * INV_SQRT2;
    const c1 = coeff[b + 1];
    const c2 = coeff[b + 2];
    const c3 = coeff[b + 3];
    const c4 = coeff[b + 4];
    const c5 = coeff[b + 5];
    const c6 = coeff[b + 6];
    const c7 = coeff[b + 7];

    tmp[tb] = (c0 + c1 * COS4[4] + c2 * COS4[8] + c3 * COS4[12] + c4 * COS4[16] + c5 * COS4[20] + c6 * COS4[24] + c7 * COS4[28]) * 0.5;

    tmp[tb + 1] = (c0 + c1 * COS4[5] + c2 * COS4[9] + c3 * COS4[13] + c4 * COS4[17] + c5 * COS4[21] + c6 * COS4[25] + c7 * COS4[29]) * 0.5;

    tmp[tb + 2] = (c0 + c1 * COS4[6] + c2 * COS4[10] + c3 * COS4[14] + c4 * COS4[18] + c5 * COS4[22] + c6 * COS4[26] + c7 * COS4[30]) * 0.5;

    tmp[tb + 3] = (c0 + c1 * COS4[7] + c2 * COS4[11] + c3 * COS4[15] + c4 * COS4[19] + c5 * COS4[23] + c6 * COS4[27] + c7 * COS4[31]) * 0.5;
  }

  // Column pass: 8 frecuencias verticales → 4 samples.
  for (let x = 0; x < 4; x++) {
    const t0 = tmp[x] * INV_SQRT2;
    const t1 = tmp[4 + x];
    const t2 = tmp[8 + x];
    const t3 = tmp[12 + x];
    const t4 = tmp[16 + x];
    const t5 = tmp[20 + x];
    const t6 = tmp[24 + x];
    const t7 = tmp[28 + x];

    let v: number;

    v = ((t0 + t1 * COS4[4] + t2 * COS4[8] + t3 * COS4[12] + t4 * COS4[16] + t5 * COS4[20] + t6 * COS4[24] + t7 * COS4[28]) * 0.5 + 128.5) | 0;

    out[off + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS4[5] + t2 * COS4[9] + t3 * COS4[13] + t4 * COS4[17] + t5 * COS4[21] + t6 * COS4[25] + t7 * COS4[29]) * 0.5 + 128.5) | 0;

    out[off + stride + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS4[6] + t2 * COS4[10] + t3 * COS4[14] + t4 * COS4[18] + t5 * COS4[22] + t6 * COS4[26] + t7 * COS4[30]) * 0.5 + 128.5) | 0;

    out[off + stride * 2 + x] = v < 0 ? 0 : v > 255 ? 255 : v;

    v = ((t0 + t1 * COS4[7] + t2 * COS4[11] + t3 * COS4[15] + t4 * COS4[19] + t5 * COS4[23] + t6 * COS4[27] + t7 * COS4[31]) * 0.5 + 128.5) | 0;

    out[off + stride * 3 + x] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}
function idctDCOnlyHalf(dc: number, out: Uint8Array, off: number, stride: number): void {
  const value = Math.round(dc * 0.125 + 128);
  const v = value < 0 ? 0 : value > 255 ? 255 : value;

  for (let y = 0; y < 4; y++) {
    const row = off + y * stride;

    out[row] = v;
    out[row + 1] = v;
    out[row + 2] = v;
    out[row + 3] = v;
  }
}
interface ProgressiveCoefficients {
  width: number;
  height: number;
  comps: Comp[];
  qtables: (Uint8Array | undefined)[];
  nc: number;
  hMax: number;
  vMax: number;
  nbX: number[];
  nbY: number[];
  totalBlocks: number;
  coeffBufs: Int16Array[];
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
// tmp is caller-provided Float64Array(64) — reused across all block calls to avoid GC.
// Uses flat COS8[u*8+x] instead of COS[u][x] for better cache behaviour.
function fdct8x8(src: Uint8Array, srcOff: number, srcStride: number, out: Float64Array, tmp: Float64Array): void {
  for (let y = 0; y < 8; y++) {
    const b = y * srcStride + srcOff,
      tb = y * 8;
    const s0 = src[b] - 128,
      s1 = src[b + 1] - 128,
      s2 = src[b + 2] - 128,
      s3 = src[b + 3] - 128;
    const s4 = src[b + 4] - 128,
      s5 = src[b + 5] - 128,
      s6 = src[b + 6] - 128,
      s7 = src[b + 7] - 128;
    for (let u = 0; u < 8; u++) {
      const ub = u * 8;
      tmp[tb + u] =
        (s0 * COS8[ub] +
          s1 * COS8[ub + 1] +
          s2 * COS8[ub + 2] +
          s3 * COS8[ub + 3] +
          s4 * COS8[ub + 4] +
          s5 * COS8[ub + 5] +
          s6 * COS8[ub + 6] +
          s7 * COS8[ub + 7]) *
        (u === 0 ? INV_SQRT2 : 1) *
        0.5;
    }
  }
  for (let u = 0; u < 8; u++) {
    const t0 = tmp[u],
      t1 = tmp[8 + u],
      t2 = tmp[16 + u],
      t3 = tmp[24 + u];
    const t4 = tmp[32 + u],
      t5 = tmp[40 + u],
      t6 = tmp[48 + u],
      t7 = tmp[56 + u];
    for (let v = 0; v < 8; v++) {
      const vb = v * 8;
      out[v * 8 + u] =
        (t0 * COS8[vb] +
          t1 * COS8[vb + 1] +
          t2 * COS8[vb + 2] +
          t3 * COS8[vb + 3] +
          t4 * COS8[vb + 4] +
          t5 * COS8[vb + 5] +
          t6 * COS8[vb + 6] +
          t7 * COS8[vb + 7]) *
        (v === 0 ? INV_SQRT2 : 1) *
        0.5;
    }
  }
}

// ── Bit reader ────────────────────────────────────────────────────────────────

class BitReader {
  bits = 0;
  bitsLeft = 0;
  private _rst = false;

  constructor(
    private buf: Uint8Array,
    public pos: number,
  ) {}

  get restartSeen(): boolean {
    const v = this._rst;
    this._rst = false;
    return v;
  }

  read(n: number): number {
    while (this.bitsLeft < n) {
      if (this.pos >= this.buf.length) {
        this.bits = (this.bits << 8) | 0xff;
        this.bitsLeft += 8;
        continue;
      }
      const byte = this.buf[this.pos++];
      if (byte === 0xff) {
        const nxt = this.buf[this.pos];
        if (nxt === 0x00) {
          this.pos++;
        } else if (nxt >= 0xd0 && nxt <= 0xd7) {
          this.pos++;
          this._rst = true;
          this.bits = 0;
          this.bitsLeft = 0;
          continue;
        }
      }
      this.bits = (this.bits << 8) | byte;
      this.bitsLeft += 8;
    }
    this.bitsLeft -= n;
    return (this.bits >>> this.bitsLeft) & ((1 << n) - 1);
  }
}

// ── Huffman decode table ──────────────────────────────────────────────────────

class HuffDec {
  private minCode = new Int32Array(17).fill(-1);
  private maxCode = new Int32Array(17).fill(-2);
  private valOff = new Int32Array(17);
  private values: Uint8Array;

  constructor(lengths: Uint8Array, values: Uint8Array) {
    this.values = new Uint8Array(values);
    let code = 0,
      vi = 0;
    for (let b = 1; b <= 16; b++) {
      const cnt = lengths[b - 1];
      if (cnt > 0) {
        this.minCode[b] = code;
        this.maxCode[b] = code + cnt - 1;
        this.valOff[b] = vi - code;
        vi += cnt;
        code += cnt;
      }
      code <<= 1;
    }
  }

  decode(r: BitReader): number {
    let code = 0;
    for (let b = 1; b <= 16; b++) {
      code = (code << 1) | r.read(1);
      if (code >= this.minCode[b] && code <= this.maxCode[b]) {
        return this.values[code + this.valOff[b]];
      }
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
  dataStart: number;
}

interface JPEGFile {
  sofType: number;
  width: number;
  height: number;
  comps: Comp[];
  qtables: (Uint8Array | undefined)[];
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
    height = 0;
  const comps: Comp[] = [];
  const qtables: (Uint8Array | undefined)[] = new Array(4);
  const dcHuff: (HuffDec | undefined)[] = new Array(4);
  const acHuff: (HuffDec | undefined)[] = new Array(4);
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
        Ss,
        Se,
        Ah,
        Al,
        dcHuff: [...dcHuff],
        acHuff: [...acHuff],
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
          const tbl = new HuffDec(new Uint8Array(lengths), new Uint8Array(values));
          if (cls === 0) {
            dcHuff[id] = tbl;
          } else {
            acHuff[id] = tbl;
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
  return { sofType, width, height, comps, qtables, scans, buf };
}

// ── Plane → VisionFrame ───────────────────────────────────────────────────────

// ── Fast YCbCr → RGB conversion tables ───────────────────────────────────────
// BT.601 integer approximation in Q8:
// R = Y + 1.402 Cr
// G = Y - 0.344136 Cb - 0.714136 Cr
// B = Y + 1.772 Cb

const CR_R = new Int16Array(256);
const CB_B = new Int16Array(256);
const CB_G = new Int16Array(256);
const CR_G = new Int16Array(256);

for (let i = 0; i < 256; i++) {
  const v = i - 128;
  CR_R[i] = (359 * v) >> 8;
  CB_B[i] = (454 * v) >> 8;
  CB_G[i] = (88 * v) >> 8;
  CR_G[i] = (183 * v) >> 8;
}

// ── Plane → VisionFrame ───────────────────────────────────────────────────────

function planesToFrameGray(planes: Uint8Array[], pw: number[], W: number, H: number): VisionFrame {
  const frame = new VisionFrame(W, H, 1);
  const dst = frame.data;
  const Y = planes[0];
  const stride = pw[0];

  for (let y = 0; y < H; y++) {
    const srcOff = y * stride;
    const dstOff = y * W;
    dst.set(Y.subarray(srcOff, srcOff + W), dstOff);
  }

  return frame;
}

function planesToFrame420(planes: Uint8Array[], pw: number[], W: number, H: number): VisionFrame {
  const frame = new VisionFrame(W, H, 3);
  const dst = frame.data;

  const Yp = planes[0];
  const Cbp = planes[1];
  const Crp = planes[2];

  const yStride = pw[0];
  const cStride = pw[1];

  for (let y = 0; y < H; y++) {
    const yOff = y * yStride;
    const cOff = (y >> 1) * cStride;

    let out = y * W * 3;
    let x = 0;

    for (; x < W - 1; x += 2) {
      const ci = cOff + (x >> 1);
      const cb = Cbp[ci];
      const cr = Crp[ci];

      const rAdd = CR_R[cr];
      const gSub = CB_G[cb] + CR_G[cr];
      const bAdd = CB_B[cb];

      let yy = Yp[yOff + x];
      let r = yy + rAdd;
      let g = yy - gSub;
      let b = yy + bAdd;

      dst[out++] = r < 0 ? 0 : r > 255 ? 255 : r;
      dst[out++] = g < 0 ? 0 : g > 255 ? 255 : g;
      dst[out++] = b < 0 ? 0 : b > 255 ? 255 : b;

      yy = Yp[yOff + x + 1];
      r = yy + rAdd;
      g = yy - gSub;
      b = yy + bAdd;

      dst[out++] = r < 0 ? 0 : r > 255 ? 255 : r;
      dst[out++] = g < 0 ? 0 : g > 255 ? 255 : g;
      dst[out++] = b < 0 ? 0 : b > 255 ? 255 : b;
    }

    if (x < W) {
      const ci = cOff + (x >> 1);
      const cb = Cbp[ci];
      const cr = Crp[ci];
      const yy = Yp[yOff + x];

      const r = yy + CR_R[cr];
      const g = yy - (CB_G[cb] + CR_G[cr]);
      const b = yy + CB_B[cb];

      dst[out++] = r < 0 ? 0 : r > 255 ? 255 : r;
      dst[out++] = g < 0 ? 0 : g > 255 ? 255 : g;
      dst[out] = b < 0 ? 0 : b > 255 ? 255 : b;
    }
  }

  return frame;
}

function planesToFrame444(planes: Uint8Array[], pw: number[], W: number, H: number): VisionFrame {
  const frame = new VisionFrame(W, H, 3);
  const dst = frame.data;

  const Yp = planes[0];
  const Cbp = planes[1];
  const Crp = planes[2];

  const yStride = pw[0];
  const cbStride = pw[1];
  const crStride = pw[2];

  for (let y = 0; y < H; y++) {
    const yOff = y * yStride;
    const cbOff = y * cbStride;
    const crOff = y * crStride;
    let out = y * W * 3;

    for (let x = 0; x < W; x++) {
      const yy = Yp[yOff + x];
      const cb = Cbp[cbOff + x];
      const cr = Crp[crOff + x];

      const r = yy + CR_R[cr];
      const g = yy - (CB_G[cb] + CR_G[cr]);
      const b = yy + CB_B[cb];

      dst[out++] = r < 0 ? 0 : r > 255 ? 255 : r;
      dst[out++] = g < 0 ? 0 : g > 255 ? 255 : g;
      dst[out++] = b < 0 ? 0 : b > 255 ? 255 : b;
    }
  }

  return frame;
}

function planesToFrameGeneric(planes: Uint8Array[], pw: number[], comps: Comp[], hMax: number, vMax: number, W: number, H: number): VisionFrame {
  const frame = new VisionFrame(W, H, 3);
  const dst = frame.data;

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
      const cb = Cbp[cbOff + cbXMap[x]];
      const cr = Crp[crOff + crXMap[x]];

      const r = yy + CR_R[cr];
      const g = yy - (CB_G[cb] + CR_G[cr]);
      const b = yy + CB_B[cb];

      dst[out++] = r < 0 ? 0 : r > 255 ? 255 : r;
      dst[out++] = g < 0 ? 0 : g > 255 ? 255 : g;
      dst[out++] = b < 0 ? 0 : b > 255 ? 255 : b;
    }
  }

  return frame;
}

function planesToFrame(planes: Uint8Array[], pw: number[], comps: Comp[], hMax: number, vMax: number, W: number, H: number): VisionFrame {
  const nc = comps.length;

  if (nc === 1) {
    return planesToFrameGray(planes, pw, W, H);
  }

  if (nc !== 3) {
    throw new Error(`JPEG: unsupported component count ${nc}`);
  }

  // Fast path: common JPEG 4:2:0
  if (
    hMax === 2 &&
    vMax === 2 &&
    comps[0].hf === 2 &&
    comps[0].vf === 2 &&
    comps[1].hf === 1 &&
    comps[1].vf === 1 &&
    comps[2].hf === 1 &&
    comps[2].vf === 1
  ) {
    return planesToFrame420(planes, pw, W, H);
  }

  // Fast path: 4:4:4
  if (
    hMax === 1 &&
    vMax === 1 &&
    comps[0].hf === 1 &&
    comps[0].vf === 1 &&
    comps[1].hf === 1 &&
    comps[1].vf === 1 &&
    comps[2].hf === 1 &&
    comps[2].vf === 1
  ) {
    return planesToFrame444(planes, pw, W, H);
  }

  return planesToFrameGeneric(planes, pw, comps, hMax, vMax, W, H);
}
// ── Baseline decoder (SOF0 / SOF1) ───────────────────────────────────────────

function decodeBaseline(f: JPEGFile): VisionFrame {
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
  return planesToFrame(planes, planeW, comps, hMax, vMax, width, height);
}

// ── Progressive decoder (SOF2) ────────────────────────────────────────────────
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
function decodeProgressiveCoefficients(f: JPEGFile): ProgressiveCoefficients {
  const { width, height, comps, qtables, scans, buf } = f;
  const nc = comps.length;
  const hMax = Math.max(...comps.map((c) => c.hf));
  const vMax = Math.max(...comps.map((c) => c.vf));
  const mcuCols = Math.ceil(width / (hMax * 8));
  const mcuRows = Math.ceil(height / (vMax * 8));

  const nbX = comps.map((c) => mcuCols * c.hf);
  const nbY = comps.map((c) => mcuRows * c.vf);
  const totalBlocks = nbX.reduce((sum, x, i) => sum + x * nbY[i], 0);

  const coeffBufs = comps.map((_, ci) => {
    const length = nbX[ci] * nbY[ci] * 64;
    return new Int16Array(new SharedArrayBuffer(length * Int16Array.BYTES_PER_ELEMENT));
  });

  for (const scan of scans) {
    const { nComps, scanComps, Ss, Se, Ah, Al, dataStart } = scan;
    const reader = new BitReader(buf, dataStart);
    const dcPrev = new Int32Array(nc);
    let eobRun = 0; // EOB run counter — persists across MCUs within the scan

    // Interleaved scans use the standard MCU grid;
    // non-interleaved (single-component) scans iterate over that component's block grid.
    const interleaved = nComps > 1;
    const gridCols = interleaved ? mcuCols : nbX[scanComps[0].ci];
    const gridRows = interleaved ? mcuRows : nbY[scanComps[0].ci];

    for (let mr = 0; mr < gridRows; mr++) {
      for (let mc = 0; mc < gridCols; mc++) {
        for (const { ci, dcId, acId } of scanComps) {
          const dc = scan.dcHuff[dcId];
          const ac = scan.acHuff[acId];

          // In interleaved scans each MCU holds hf×vf blocks per component.
          // In non-interleaved scans each MCU is exactly one block.
          const bRows = interleaved ? comps[ci].vf : 1;
          const bCols = interleaved ? comps[ci].hf : 1;

          for (let brow = 0; brow < bRows; brow++) {
            for (let bcol = 0; bcol < bCols; bcol++) {
              const col = interleaved ? mc * comps[ci].hf + bcol : mc;
              const row = interleaved ? mr * comps[ci].vf + brow : mr;
              const bi = row * nbX[ci] + col;
              const coeffs = coeffBufs[ci];
              const bo = bi * 64;
              // ── DC first pass ─────────────────────────────────────
              if (Ss === 0 && Ah === 0) {
                const cat = dc!.decode(reader);
                dcPrev[ci] += cat === 0 ? 0 : signExtend(reader.read(cat), cat);
                coeffs[bo] = dcPrev[ci] << Al;
              }

              // ── DC refinement ─────────────────────────────────────
              else if (Ss === 0 && Ah > 0) {
                if (reader.read(1)) {
                  coeffs[bo] |= 1 << Al;
                }
              }

              // ── AC first pass ─────────────────────────────────────
              else if (Ss > 0 && Ah === 0) {
                if (eobRun > 0) {
                  eobRun--;
                } else {
                  let k = Ss;
                  while (k <= Se) {
                    const sym = ac!.decode(reader);
                    const ssss = sym & 0xf;
                    const rrrr = (sym >> 4) & 0xf;
                    if (ssss === 0) {
                      if (rrrr === 15) {
                        k += 16; // ZRL: skip 16 positions
                      } else if (rrrr === 0) {
                        break; // EOB1: this block done
                      } else {
                        // EOB run: current block + (decoded_count - 1) more
                        eobRun = (1 << rrrr) + reader.read(rrrr) - 1;
                        break;
                      }
                    } else {
                      k += rrrr;
                      if (k > Se) {
                        break;
                      }
                      coeffs[bo + k] = signExtend(reader.read(ssss), ssss) << Al;
                      k++;
                    }
                  }
                }
              }

              // ── AC refinement ─────────────────────────────────────
              else if (Ss > 0 && Ah > 0) {
                const bit1 = 1 << Al;
                const refine = (v: number) => (v > 0 ? v + bit1 : v - bit1);

                if (eobRun > 0) {
                  // Block is an EOB: just refine any already-nonzero coefficients
                  for (let k = Ss; k <= Se; k++) {
                    if (coeffs[bo + k] !== 0 && reader.read(1)) {
                      coeffs[bo + k] = refine(coeffs[bo + k]);
                    }
                  }
                  eobRun--;
                } else {
                  let k = Ss;
                  outer: while (k <= Se) {
                    const sym = ac!.decode(reader);
                    const ssss = sym & 0xf;
                    let rrrr = (sym >> 4) & 0xf;

                    if (ssss === 0) {
                      if (rrrr === 15) {
                        // ZRL: advance past 16 zero-valued positions,
                        // refining any nonzero coefficients we pass through
                        let zeros = 16;
                        while (k <= Se && zeros > 0) {
                          if (coeffs[bo + k] !== 0) {
                            if (reader.read(1)) {
                              coeffs[bo + k] = refine(coeffs[bo + k]);
                            }
                          } else {
                            zeros--;
                          }
                          k++;
                        }
                      } else {
                        // EOB (possibly run)
                        if (rrrr > 0) {
                          eobRun = (1 << rrrr) + reader.read(rrrr) - 1;
                        }
                        // Refine remaining nonzeros in this block
                        for (; k <= Se; k++) {
                          if (coeffs[bo + k] !== 0 && reader.read(1)) {
                            coeffs[bo + k] = refine(coeffs[bo + k]);
                          }
                        }
                        break outer;
                      }
                    } else {
                      // ssss=1: place a new nonzero.
                      // First advance past `rrrr` zero-valued coefficients,
                      // refining any nonzeros we encounter along the way.
                      while (k <= Se) {
                        if (coeffs[bo + k] !== 0) {
                          if (reader.read(1)) {
                            coeffs[bo + k] = refine(coeffs[bo + k]);
                          }
                          k++;
                        } else {
                          if (rrrr === 0) {
                            break;
                          }
                          rrrr--;
                          k++;
                        }
                      }
                      // Place the new coefficient (1 bit → +/-1 level)
                      if (k <= Se) {
                        coeffs[bo + k] = signExtend(reader.read(1), 1) << Al;
                        k++;
                      }
                    }
                  }
                }
              }
            }
          }
        }

        if (reader.restartSeen) {
          dcPrev.fill(0);
          eobRun = 0;
        }
      }
    }
  }

  return {
    width,
    height,
    comps,
    qtables,
    nc,
    hMax,
    vMax,
    nbX,
    nbY,
    totalBlocks,
    coeffBufs,
  };
}
function runIdctSyncHalf(
  coeffBufs: Int16Array[],
  planes: Uint8Array[],
  qtables: (Uint8Array | undefined)[],
  comps: Comp[],
  nbX: number[],
  nbY: number[],
  planeW: number[],
): void {
  const dct = new Float64Array(64);
  const tmp = new Float64Array(32);

  for (let ci = 0; ci < comps.length; ci++) {
    const qt = qtables[comps[ci].qtId]!;
    const pw = planeW[ci];
    const coeffs = coeffBufs[ci];

    for (let row = 0; row < nbY[ci]; row++) {
      for (let col = 0; col < nbX[ci]; col++) {
        const bi = row * nbX[ci] + col;
        const bo = bi * 64;
        const outOff = row * 4 * pw + col * 4;

        if (isDCOnly(coeffs, bo)) {
          idctDCOnlyHalf(coeffs[bo] * qt[0], planes[ci], outOff, pw);
          continue;
        }

        dct.fill(0);

        for (let k = 0; k < 64; k++) {
          dct[ZZ[k]] = coeffs[bo + k] * qt[k];
        }

        idct8x8To4x4(dct, tmp, planes[ci], outOff, pw);
      }
    }
  }
}
function decodeProgressiveHalf(f: JPEGFile): VisionFrame {
  const decoded = decodeProgressiveCoefficients(f);

  const { width, height, comps, qtables, hMax, vMax, nbX, nbY, coeffBufs } = decoded;

  const halfW = Math.ceil(width / 2);
  const halfH = Math.ceil(height / 2);

  const planeW = nbX.map((n) => n * 4);
  const planeH = nbY.map((n) => n * 4);
  const planes = comps.map((_, ci) => new Uint8Array(planeW[ci] * planeH[ci]));

  runIdctSyncHalf(coeffBufs, planes, qtables, comps, nbX, nbY, planeW);

  return planesToFrame(planes, planeW, comps, hMax, vMax, halfW, halfH);
}
async function decodeProgressive(f: JPEGFile): Promise<VisionFrame> {
  const decoded = decodeProgressiveCoefficients(f);

  const { width, height, comps, qtables, nc, hMax, vMax, nbX, nbY, totalBlocks, coeffBufs } = decoded;

  const planeW = nbX.map((n) => n * 8);
  const planeH = nbY.map((n) => n * 8);

  const useWorkers = totalBlocks >= JPEG_IDCT_WORKER_THRESHOLD_BLOCKS;

  const planes = comps.map((_, ci) => {
    const size = planeW[ci] * planeH[ci];

    return useWorkers ? new Uint8Array(new SharedArrayBuffer(size)) : new Uint8Array(size);
  });

  if (!useWorkers) {
    runIdctSync(coeffBufs, planes, qtables, comps, nbX, nbY, planeW);
    return planesToFrame(planes, planeW, comps, hMax, vMax, width, height);
  }

  const pool = getJpegIdctWorkerPool();
  const jobs: Promise<void>[] = [];

  for (let ci = 0; ci < nc; ci++) {
    const qt = qtables[comps[ci].qtId]!;
    const rows = nbY[ci];

    const chunks = Math.min(4, rows);
    const rowsPerJob = Math.ceil(rows / chunks);

    for (let j = 0; j < chunks; j++) {
      const rowStart = j * rowsPerJob;
      const rowEnd = Math.min(rows, rowStart + rowsPerJob);

      if (rowStart >= rowEnd) {
        continue;
      }

      jobs.push(
        pool.run({
          coeffBuffer: sharedBufferOf(coeffBufs[ci]),
          planeBuffer: sharedBufferOf(planes[ci]),
          qt,
          nbX: nbX[ci],
          rowStart,
          rowEnd,
          planeWidth: planeW[ci],
        }),
      );
    }
  }

  await Promise.all(jobs);

  return planesToFrame(planes, planeW, comps, hMax, vMax, width, height);
}
function sharedBufferOf(view: ArrayBufferView): SharedArrayBuffer {
  if (!(view.buffer instanceof SharedArrayBuffer)) {
    throw new Error('Expected SharedArrayBuffer-backed view');
  }

  return view.buffer;
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
  planeW: number[],
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

export async function readJPEG(path: string, opts: JPEGReadOptions = {}): Promise<VisionFrame> {
  const raw = await fs.readFile(path);
  const buf = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
  const jpeg = parseFile(buf);

  if (opts.resize?.shrinkOnLoad === true) {
    const target = resolveJPEGResizeSize(jpeg.width, jpeg.height, opts.resize);

    if (jpeg.sofType === 2 && shouldUseHalfShrink(jpeg, target)) {
      const half = decodeProgressiveHalf(jpeg);

      const { resize } = await import('../kernels/geometry.js');

      return resize(half, target.width, target.height, target.method);
    }
  }

  const frame = jpeg.sofType <= 1 ? decodeBaseline(jpeg) : await decodeProgressive(jpeg);

  if (!opts.resize) {
    return frame;
  }

  const { resize } = await import('../kernels/geometry.js');

  const target = resolveJPEGResizeSize(frame.width, frame.height, opts.resize);

  return resize(frame, target.width, target.height, target.method);
}
type JPEGResizeMethod = 'nearest' | 'bilinear' | 'area';

interface ResolvedJPEGResize {
  width: number;
  height: number;
  method: JPEGResizeMethod;
}

function resolveJPEGResizeSize(srcW: number, srcH: number, resize: NonNullable<JPEGReadOptions['resize']>): ResolvedJPEGResize {
  const method = resize.method ?? 'bilinear';

  if (resize.width !== undefined && resize.height !== undefined) {
    return {
      width: Math.max(1, Math.round(resize.width)),
      height: Math.max(1, Math.round(resize.height)),
      method,
    };
  }

  if (resize.width !== undefined) {
    const w = Math.max(1, Math.round(resize.width));
    const h = Math.max(1, Math.round((srcH * w) / srcW));

    return { width: w, height: h, method };
  }

  if (resize.height !== undefined) {
    const h = Math.max(1, Math.round(resize.height));
    const w = Math.max(1, Math.round((srcW * h) / srcH));

    return { width: w, height: h, method };
  }

  return {
    width: srcW,
    height: srcH,
    method,
  };
}

function shouldUseHalfShrink(jpeg: JPEGFile, target: ResolvedJPEGResize): boolean {
  const halfW = Math.ceil(jpeg.width / 2);
  const halfH = Math.ceil(jpeg.height / 2);

  // Solo conviene si el objetivo cabe dentro de la imagen half.
  // Evita decodificar half y luego upscalear.
  return target.width <= halfW && target.height <= halfH;
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

// ── Coefficient helpers ───────────────────────────────────────────────────────

function valueCat(v: number): number {
  return v === 0 ? 0 : 32 - Math.clz32(Math.abs(v));
}
function encodeVal(v: number, cat: number): number {
  return v >= 0 ? v : v + (1 << cat) - 1;
}

// ── Main encoder ──────────────────────────────────────────────────────────────

// ── OutputBuffer ─────────────────────────────────────────────────────────────

class OutputBuffer {
  private buf: Uint8Array;
  private pos = 0;

  constructor(cap: number) {
    this.buf = new Uint8Array(cap);
  }

  private grow(n: number): void {
    let c = this.buf.length;
    while (c < this.pos + n) {
      c *= 2;
    }
    const b = new Uint8Array(c);
    b.set(this.buf.subarray(0, this.pos));
    this.buf = b;
  }

  writeByte(b: number): void {
    if (this.pos >= this.buf.length) {
      this.grow(1);
    }
    this.buf[this.pos++] = b;
  }

  writeBytes(data: ArrayLike<number>): void {
    const len = data.length;
    if (this.pos + len > this.buf.length) {
      this.grow(len);
    }
    if (data instanceof Uint8Array) {
      this.buf.set(data, this.pos);
    } else {
      for (let i = 0; i < len; i++) {
        this.buf[this.pos + i] = data[i];
      }
    }
    this.pos += len;
  }

  writeU16(v: number): void {
    if (this.pos + 2 > this.buf.length) {
      this.grow(2);
    }
    this.buf[this.pos++] = (v >> 8) & 0xff;
    this.buf[this.pos++] = v & 0xff;
  }

  appendWriter(w: BitWriter2): void {
    const d = w.data;
    if (this.pos + d.length > this.buf.length) {
      this.grow(d.length);
    }
    this.buf.set(d, this.pos);
    this.pos += d.length;
  }

  get result(): Uint8Array {
    return this.buf.subarray(0, this.pos);
  }
}

// ── BitWriter2 — writes bits directly into Uint8Array ────────────────────────

class BitWriter2 {
  private buf: Uint8Array;
  private pos = 0;
  private bits = 0;
  private left = 0;

  constructor(cap: number) {
    this.buf = new Uint8Array(cap);
  }

  write(value: number, length: number): void {
    this.bits = (this.bits << length) | (value & ((1 << length) - 1));
    this.left += length;
    while (this.left >= 8) {
      this.left -= 8;
      const b = (this.bits >>> this.left) & 0xff;
      if (this.pos + 2 > this.buf.length) {
        const n = new Uint8Array(this.buf.length * 2);
        n.set(this.buf);
        this.buf = n;
      }
      this.buf[this.pos++] = b;
      if (b === 0xff) {
        this.buf[this.pos++] = 0x00;
      }
    }
  }

  flush(): void {
    if (this.left > 0) {
      if (this.pos + 2 > this.buf.length) {
        const n = new Uint8Array(this.buf.length * 2);
        n.set(this.buf);
        this.buf = n;
      }
      const b = ((this.bits << (8 - this.left)) | ((1 << (8 - this.left)) - 1)) & 0xff;
      this.buf[this.pos++] = b;
      if (b === 0xff) {
        this.buf[this.pos++] = 0x00;
      }
    }
  }

  get data(): Uint8Array {
    return this.buf.subarray(0, this.pos);
  }
}

// ── Segment helpers ───────────────────────────────────────────────────────────

function writeDQT(out: OutputBuffer, id: number, qt: Uint8Array): void {
  out.writeByte(0xff);
  out.writeByte(0xdb);
  out.writeU16(67);
  out.writeByte(id);
  out.writeBytes(qt);
}

function writeDHT(out: OutputBuffer, cls: number, id: number, bits: number[], vals: number[]): void {
  out.writeByte(0xff);
  out.writeByte(0xc4);
  out.writeU16(2 + 1 + 16 + vals.length);
  out.writeByte((cls << 4) | id);
  out.writeBytes(bits);
  out.writeBytes(vals);
}

function writeSOS(out: OutputBuffer, comps: Array<{ id: number; dcId: number; acId: number }>, Ss: number, Se: number, Ah: number, Al: number): void {
  out.writeByte(0xff);
  out.writeByte(0xda);
  out.writeU16(2 + 1 + comps.length * 2 + 3);
  out.writeByte(comps.length);
  for (const c of comps) {
    out.writeByte(c.id);
    out.writeByte((c.dcId << 4) | c.acId);
  }
  out.writeByte(Ss);
  out.writeByte(Se);
  out.writeByte((Ah << 4) | Al);
}

export interface JPEGWriteOptions {
  progressive?: boolean;
}

export async function writeJPEG(path: string, frame: VisionFrame, quality = 85, opts: JPEGWriteOptions = {}): Promise<void> {
  if (frame.channels !== 3) {
    throw new Error('writeJPEG: requires 3-channel RGB frame (.toRGB() first)');
  }

  const { width, height } = frame;
  const src = frame.data;
  const n = width * height;

  const lumaQ = scaleQT(LUMA_QT, quality);
  const chromaQ = scaleQT(CHROMA_QT, quality);
  const lumaZZ = toZZ(lumaQ);
  const chromaZZ = toZZ(chromaQ);

  const dcLE = buildEnc(DC_L_BITS, DC_L_VALS);
  const dcCE = buildEnc(DC_C_BITS, DC_C_VALS);
  const acLE = buildEnc(AC_L_BITS, AC_L_VALS);
  const acCE = buildEnc(AC_C_BITS, AC_C_VALS);

  // ── RGB → YCbCr (integer Q8, BT.601) ─────────────────────────────────────
  // Y  = ( 77R + 150G + 29B) >> 8
  // Cb = (-43R -  85G + 128B) >> 8 + 128
  // Cr = (128R - 107G -  21B) >> 8 + 128
  const Y = new Uint8Array(n);
  const Cb = new Uint8Array(n);
  const Cr = new Uint8Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 3) {
    const R = src[p],
      G = src[p + 1],
      B = src[p + 2];
    Y[i] = (77 * R + 150 * G + 29 * B) >> 8;
    const cb = ((-43 * R - 85 * G + 128 * B) >> 8) + 128;
    const cr = ((128 * R - 107 * G - 21 * B) >> 8) + 128;
    Cb[i] = cb < 0 ? 0 : cb > 255 ? 255 : cb;
    Cr[i] = cr < 0 ? 0 : cr > 255 ? 255 : cr;
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

  // ── FDCT + quantise — pre-allocated fdctTmp avoids per-block allocation ───
  const qcoeff = [new Int16Array(nBlocks * 64), new Int16Array(nBlocks * 64), new Int16Array(nBlocks * 64)];
  const progressive = opts.progressive ?? true;

  if (!progressive) {
    return writeJPEGBaselineFromQcoeff({
      path,
      width,
      height,
      nBlocks,
      mcuRows,
      mcuCols,
      qcoeff,
      lumaZZ,
      chromaZZ,
      dcLE,
      dcCE,
      acLE,
      acCE,
    });
  }
  const planesArr = [yP, cbP, crP];
  const qts = [lumaQ, chromaQ, chromaQ];
  const dct = new Float64Array(64);
  const fdctTmp = new Float64Array(64);

  for (let ci = 0; ci < 3; ci++) {
    const qt = qts[ci];
    for (let row = 0; row < mcuRows; row++) {
      for (let col = 0; col < mcuCols; col++) {
        const bi = row * mcuCols + col;
        fdct8x8(planesArr[ci], row * 8 * pw + col * 8, pw, dct, fdctTmp);
        const qc = qcoeff[ci].subarray(bi * 64, bi * 64 + 64);
        for (let k = 0; k < 64; k++) {
          qc[k] = Math.round(dct[ZZ[k]] / qt[ZZ[k]]);
        }
      }
    }
  }

  // ── Assemble into OutputBuffer (zero-copy, no number[] boxing) ─────────────
  const out = new OutputBuffer(Math.max(65536, nBlocks * 32));

  // SOI + APP0
  out.writeBytes([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]);
  writeDQT(out, 0, lumaZZ);
  writeDQT(out, 1, chromaZZ);

  // SOF2
  out.writeByte(0xff);
  out.writeByte(0xc2);
  out.writeU16(17);
  out.writeBytes([0x08, height >> 8, height & 0xff, width >> 8, width & 0xff, 0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);

  // ── Scan 0: DC all, interleaved ──────────────────────────────────────────
  writeDHT(out, 0, 0, DC_L_BITS, DC_L_VALS);
  writeDHT(out, 0, 1, DC_C_BITS, DC_C_VALS);
  writeSOS(
    out,
    [
      { id: 1, dcId: 0, acId: 0 },
      { id: 2, dcId: 1, acId: 1 },
      { id: 3, dcId: 1, acId: 1 },
    ],
    0,
    0,
    0,
    0,
  );
  {
    const w = new BitWriter2(Math.max(4096, nBlocks * 6));
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
    out.appendWriter(w);
  }

  // ── Scans 1-4: AC, non-interleaved ──────────────────────────────────────
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
      writeDHT(out, 1, dhtId, bits, vals);
      lastDhtId = dhtId;
    }
    writeSOS(out, [{ id, dcId: 0, acId: dhtId }], Ss, Se, 0, 0);
    const w = new BitWriter2(Math.max(4096, nBlocks * 4));
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
    out.appendWriter(w);
  }

  out.writeByte(0xff);
  out.writeByte(0xd9);

  // Write directly from Uint8Array — no intermediate number[] conversion
  const res = out.result;
  await fs.writeFile(path, Buffer.from(res.buffer, res.byteOffset, res.byteLength));
}

async function writeJPEGBaselineFromQcoeff(args: {
  path: string;
  width: number;
  height: number;
  nBlocks: number;
  mcuRows: number;
  mcuCols: number;
  qcoeff: Int16Array[];
  lumaZZ: Uint8Array;
  chromaZZ: Uint8Array;
  dcLE: (HCode | undefined)[];
  dcCE: (HCode | undefined)[];
  acLE: (HCode | undefined)[];
  acCE: (HCode | undefined)[];
}): Promise<void> {
  const { path, width, height, nBlocks, mcuRows, mcuCols, qcoeff, lumaZZ, chromaZZ, dcLE, dcCE, acLE, acCE } = args;

  const out = new OutputBuffer(Math.max(65536, nBlocks * 32));

  // SOI + APP0
  out.writeBytes([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]);

  writeDQT(out, 0, lumaZZ);
  writeDQT(out, 1, chromaZZ);

  // SOF0 baseline, 4:4:4, 3 components
  out.writeByte(0xff);
  out.writeByte(0xc0);
  out.writeU16(17);
  out.writeBytes([0x08, height >> 8, height & 0xff, width >> 8, width & 0xff, 0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);

  writeDHT(out, 0, 0, DC_L_BITS, DC_L_VALS);
  writeDHT(out, 1, 0, AC_L_BITS, AC_L_VALS);
  writeDHT(out, 0, 1, DC_C_BITS, DC_C_VALS);
  writeDHT(out, 1, 1, AC_C_BITS, AC_C_VALS);

  // One interleaved baseline scan: Y, Cb, Cr, Ss=0 Se=63 Ah=0 Al=0
  writeSOS(
    out,
    [
      { id: 1, dcId: 0, acId: 0 },
      { id: 2, dcId: 1, acId: 1 },
      { id: 3, dcId: 1, acId: 1 },
    ],
    0,
    63,
    0,
    0,
  );

  const w = new BitWriter2(Math.max(4096, nBlocks * 16));

  const dcPrev = [0, 0, 0];
  const dcEncs = [dcLE, dcCE, dcCE];
  const acEncs = [acLE, acCE, acCE];

  for (let mr = 0; mr < mcuRows; mr++) {
    for (let mc = 0; mc < mcuCols; mc++) {
      const bi = mr * mcuCols + mc;

      for (let ci = 0; ci < 3; ci++) {
        const qc = qcoeff[ci].subarray(bi * 64, bi * 64 + 64);

        const diff = qc[0] - dcPrev[ci];
        dcPrev[ci] = qc[0];

        const dcCat = valueCat(diff);
        const dcH = dcEncs[ci][dcCat]!;

        w.write(dcH.code, dcH.bits);

        if (dcCat > 0) {
          w.write(encodeVal(diff, dcCat), dcCat);
        }

        let run = 0;
        const acEnc = acEncs[ci];

        for (let k = 1; k < 64; k++) {
          const v = qc[k];

          if (v === 0) {
            run++;

            if (run === 16) {
              const zrl = acEnc[0xf0]!;
              w.write(zrl.code, zrl.bits);
              run = 0;
            }

            continue;
          }

          const acCat = valueCat(v);

          while (run > 15) {
            const zrl = acEnc[0xf0]!;
            w.write(zrl.code, zrl.bits);
            run -= 16;
          }

          const h = acEnc[(run << 4) | acCat]!;
          w.write(h.code, h.bits);
          w.write(encodeVal(v, acCat), acCat);

          run = 0;
        }

        if (run > 0) {
          const eob = acEnc[0]!;
          w.write(eob.code, eob.bits);
        }
      }
    }
  }

  w.flush();
  out.appendWriter(w);

  out.writeByte(0xff);
  out.writeByte(0xd9);

  const res = out.result;
  await fs.writeFile(path, Buffer.from(res.buffer, res.byteOffset, res.byteLength));
}
