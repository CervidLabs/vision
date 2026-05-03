/**
 * gaussian-blur.worker.ts
 *
 * Worker thread for one pass of a separable Gaussian blur.
 * Receives a SharedArrayBuffer source plane and writes to a SharedArrayBuffer
 * destination plane, processing only the assigned row range.
 *
 * The caller does two passes (horizontal then vertical) using this worker,
 * swapping src/dst roles between passes.
 */

import { parentPort, workerData } from 'node:worker_threads';

interface BlurJob {
  srcBuf: SharedArrayBuffer;
  dstBuf: SharedArrayBuffer;
  width: number;
  height: number;
  channels: number;
  kernel: number[]; // 1-D Gaussian kernel (normalized)
  radius: number;
  direction: 'h' | 'v'; // horizontal or vertical pass
  yStart: number;
  yEnd: number;
}

const { srcBuf, dstBuf, width, height, channels, kernel, radius, direction, yStart, yEnd } = workerData as BlurJob;

const src = new Uint8Array(srcBuf);
const dst = new Uint8Array(dstBuf);
const last = direction === 'h' ? width - 1 : height - 1;

if (direction === 'h') {
  for (let y = yStart; y < yEnd; y++) {
    const rowOff = y * width * channels;
    for (let x = 0; x < width; x++) {
      const base = rowOff + x * channels;
      for (let c = 0; c < channels; c++) {
        let acc = 0;
        for (let k = -radius; k <= radius; k++) {
          const sx = x + k < 0 ? 0 : x + k > last ? last : x + k;
          acc += src[rowOff + sx * channels + c] * kernel[k + radius];
        }
        dst[base + c] = acc < 0 ? 0 : acc > 255 ? 255 : acc | 0;
      }
    }
  }
} else {
  for (let y = yStart; y < yEnd; y++) {
    for (let x = 0; x < width; x++) {
      const base = (y * width + x) * channels;
      for (let c = 0; c < channels; c++) {
        let acc = 0;
        for (let k = -radius; k <= radius; k++) {
          const sy = y + k < 0 ? 0 : y + k > last ? last : y + k;
          acc += src[(sy * width + x) * channels + c] * kernel[k + radius];
        }
        dst[base + c] = acc < 0 ? 0 : acc > 255 ? 255 : acc | 0;
      }
    }
  }
}

parentPort!.postMessage('done');
