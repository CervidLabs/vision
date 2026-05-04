/**
 * resize.worker.ts
 *
 * Handles one row chunk of a resize operation.
 * Supports two methods dispatched at runtime:
 *
 *   'bilinear' — uses precomputed Q15 coordinate maps (SharedArrayBuffer),
 *                avoiding per-pixel coordinate arithmetic in the worker.
 *                Best for upscaling and mild downscaling (<= 2×).
 *
 *   'area'     — box filter with exact fractional overlap weights.
 *                Correct for downscaling >2× (no aliasing).
 *                Each output pixel averages scaleX × scaleY input pixels.
 */

import { parentPort, workerData } from 'node:worker_threads';

interface BilinearJob {
  method: 'bilinear';
  srcBuf: SharedArrayBuffer;
  dstBuf: SharedArrayBuffer;
  srcW: number;
  dstW: number;
  channels: number;
  rowStart: number;
  rowEnd: number;
  // Precomputed Q15 coordinate maps (Int32Array SABs)
  x0Buf: SharedArrayBuffer; // x0 * channels (pixel byte offset)
  x1Buf: SharedArrayBuffer; // x1 * channels
  wxBuf: SharedArrayBuffer; // fractional x weight in Q15 [0..32768]
  y0Buf: SharedArrayBuffer; // source y0 row index (not byte offset)
  y1Buf: SharedArrayBuffer; // source y1 row index
  wyBuf: SharedArrayBuffer; // fractional y weight in Q15 [0..32768]
}

interface AreaJob {
  method: 'area';
  srcBuf: SharedArrayBuffer;
  dstBuf: SharedArrayBuffer;
  srcW: number;
  srcH: number;
  dstW: number;
  channels: number;
  rowStart: number;
  rowEnd: number;
  scaleX: number; // srcW / dstW
  scaleY: number; // srcH / dstH
}

type ResizeJob = BilinearJob | AreaJob;

const job = workerData as ResizeJob;
const src = new Uint8Array(job.srcBuf);
const dst = new Uint8Array(job.dstBuf);
const { channels, dstW, rowStart, rowEnd } = job;

// ── Bilinear ──────────────────────────────────────────────────────────────────

if (job.method === 'bilinear') {
  const x0 = new Int32Array(job.x0Buf);
  const x1 = new Int32Array(job.x1Buf);
  const wx = new Int32Array(job.wxBuf);
  const y0 = new Int32Array(job.y0Buf);
  const y1 = new Int32Array(job.y1Buf);
  const wy = new Int32Array(job.wyBuf);

  const FRAC = 15;
  const HALF = 1 << (FRAC - 1);
  const srcStride = job.srcW * channels;

  for (let y = rowStart; y < rowEnd; y++) {
    const row0 = y0[y] * srcStride;
    const row1 = y1[y] * srcStride;
    const wY = wy[y];
    const iwY = 32768 - wY;
    const dRow = y * dstW * channels;

    if (channels === 3) {
      for (let x = 0; x < dstW; x++) {
        const p00 = row0 + x0[x],
          p10 = row0 + x1[x];
        const p01 = row1 + x0[x],
          p11 = row1 + x1[x];
        const wX = wx[x],
          iwX = 32768 - wX;
        const d = dRow + x * 3;

        // Unrolled 3-channel interpolation — no inner loop, V8 can inline
        let top, bot, v;

        top = (src[p00] * iwX + src[p10] * wX + HALF) >> FRAC;
        bot = (src[p01] * iwX + src[p11] * wX + HALF) >> FRAC;
        v = (top * iwY + bot * wY + HALF) >> FRAC;
        dst[d] = v < 0 ? 0 : v > 255 ? 255 : v;

        top = (src[p00 + 1] * iwX + src[p10 + 1] * wX + HALF) >> FRAC;
        bot = (src[p01 + 1] * iwX + src[p11 + 1] * wX + HALF) >> FRAC;
        v = (top * iwY + bot * wY + HALF) >> FRAC;
        dst[d + 1] = v < 0 ? 0 : v > 255 ? 255 : v;

        top = (src[p00 + 2] * iwX + src[p10 + 2] * wX + HALF) >> FRAC;
        bot = (src[p01 + 2] * iwX + src[p11 + 2] * wX + HALF) >> FRAC;
        v = (top * iwY + bot * wY + HALF) >> FRAC;
        dst[d + 2] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    } else {
      for (let x = 0; x < dstW; x++) {
        const p00 = row0 + x0[x],
          p10 = row0 + x1[x];
        const p01 = row1 + x0[x],
          p11 = row1 + x1[x];
        const wX = wx[x],
          iwX = 32768 - wX;
        const d = dRow + x * channels;

        for (let c = 0; c < channels; c++) {
          const top = (src[p00 + c] * iwX + src[p10 + c] * wX + HALF) >> FRAC;
          const bot = (src[p01 + c] * iwX + src[p11 + c] * wX + HALF) >> FRAC;
          const v = (top * iwY + bot * wY + HALF) >> FRAC;
          dst[d + c] = v < 0 ? 0 : v > 255 ? 255 : v;
        }
      }
    }
  }
}

// ── Area ──────────────────────────────────────────────────────────────────────
else if (job.method === 'area') {
  const { srcW, srcH, scaleX, scaleY } = job;
  const area = scaleX * scaleY;
  const invArea = 1 / area;
  const srcStride = srcW * channels;
  // Float32 accumulator — sufficient precision for u8 averaging
  const acc = new Float32Array(channels);

  for (let y = rowStart; y < rowEnd; y++) {
    const srcY0 = y * scaleY;
    const srcY1 = srcY0 + scaleY;
    const yStart = srcY0 | 0;
    const yEnd = Math.min(srcH - 1, Math.ceil(srcY1) - 1);
    const dRow = y * dstW * channels;

    for (let x = 0; x < dstW; x++) {
      const srcX0 = x * scaleX;
      const srcX1 = srcX0 + scaleX;
      const xStart = srcX0 | 0;
      const xEnd = Math.min(srcW - 1, Math.ceil(srcX1) - 1);

      acc.fill(0);

      for (let sy = yStart; sy <= yEnd; sy++) {
        const yOv = Math.min(srcY1, sy + 1) - Math.max(srcY0, sy);
        if (yOv <= 0) {
          continue;
        }
        const sRow = sy * srcStride;

        for (let sx = xStart; sx <= xEnd; sx++) {
          const xOv = Math.min(srcX1, sx + 1) - Math.max(srcX0, sx);
          if (xOv <= 0) {
            continue;
          }
          const w = xOv * yOv;
          const off = sRow + sx * channels;
          for (let c = 0; c < channels; c++) {
            acc[c] += src[off + c] * w;
          }
        }
      }

      const dOff = dRow + x * channels;
      for (let c = 0; c < channels; c++) {
        const v = (acc[c] * invArea + 0.5) | 0;
        dst[dOff + c] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }
}

parentPort?.postMessage('done');
