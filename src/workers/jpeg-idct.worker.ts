import { parentPort } from 'worker_threads';

const INV_SQRT2 = 1 / Math.SQRT2;

const ZZ = new Uint8Array([
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43,
  36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]);

const COS8 = new Float64Array(64);

for (let u = 0; u < 8; u++) {
  for (let x = 0; x < 8; x++) {
    COS8[u * 8 + x] = Math.cos(((2 * x + 1) * u * Math.PI) / 16);
  }
}

function idctDCOnly(dc: number, out: Uint8Array, off: number, stride: number): void {
  const value = (dc * 0.125 + 128.5) | 0;
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

function idct8x8(coeff: Float64Array, tmp: Float64Array, out: Uint8Array, off: number, stride: number): void {
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

interface Job {
  id: number;
  coeffBuffer: SharedArrayBuffer;
  planeBuffer: SharedArrayBuffer;
  qt: Uint8Array;
  nbX: number;
  rowStart: number;
  rowEnd: number;
  planeWidth: number;
}

parentPort?.on('message', (job: Job) => {
  const coeffs = new Int16Array(job.coeffBuffer);
  const plane = new Uint8Array(job.planeBuffer);

  const dct = new Float64Array(64);
  const tmp = new Float64Array(64);

  for (let row = job.rowStart; row < job.rowEnd; row++) {
    for (let col = 0; col < job.nbX; col++) {
      const bi = row * job.nbX + col;
      const bo = bi * 64;
      const outOff = row * 8 * job.planeWidth + col * 8;

      if (isDCOnly(coeffs, bo)) {
        idctDCOnly(coeffs[bo] * job.qt[0], plane, outOff, job.planeWidth);
        continue;
      }

      dct.fill(0);

      for (let k = 0; k < 64; k++) {
        dct[ZZ[k]] = coeffs[bo + k] * job.qt[k];
      }

      idct8x8(dct, tmp, plane, outOff, job.planeWidth);
    }
  }

  parentPort?.postMessage({ id: job.id });
});
