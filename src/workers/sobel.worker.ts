import { parentPort, workerData } from 'worker_threads';

interface Data {
  width: number;
  height: number;
  srcBuffer: SharedArrayBuffer;
  dstBuffer: SharedArrayBuffer;
  yStart: number;
  yEnd: number;
}

const { width, height, srcBuffer, dstBuffer, yStart, yEnd } = workerData as Data;

const src = new Uint8Array(srcBuffer);
const dst = new Uint8Array(dstBuffer);

const start = Math.max(1, yStart);
const end = Math.min(height - 1, yEnd);

for (let y = start; y < end; y++) {
  const row = y * width;

  for (let x = 1; x < width - 1; x++) {
    const i = row + x;

    const tl = src[i - width - 1];
    const tc = src[i - width];
    const tr = src[i - width + 1];

    const ml = src[i - 1];
    const mr = src[i + 1];

    const bl = src[i + width - 1];
    const bc = src[i + width];
    const br = src[i + width + 1];

    const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
    const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;

    const mag = Math.abs(gx) + Math.abs(gy);
    dst[i] = mag > 255 ? 255 : mag;
  }
}

parentPort?.postMessage('done');
