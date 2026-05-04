import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { VisionFrame } from '../core/VisionFrame.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function runWorker(file: string, workerData: unknown): Promise<void> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(file, { workerData });

    worker.once('message', () => resolve());
    worker.once('error', reject);
    worker.once('exit', (code) => {
      if (code !== 0) {
        reject(new Error(`Worker stopped with exit code ${code}`));
      }
    });
  });
}

export async function sobelParallel(frame: VisionFrame, workers = Math.max(1, cpus().length - 1)): Promise<VisionFrame> {
  if (frame.channels !== 1) {
    throw new Error('sobelParallel requires a 1-channel grayscale frame');
  }

  if (frame.width < 3 || frame.height < 3) {
    return new VisionFrame(frame.width, frame.height, 1);
  }

  workers = Math.max(1, Math.min(workers, frame.height - 2));

  const out = new VisionFrame(frame.width, frame.height, 1);

  const workerFile = join(__dirname, '../workers/sobel.worker.js');

  const rowsPerWorker = Math.ceil(frame.height / workers);
  const jobs: Promise<void>[] = [];

  for (let w = 0; w < workers; w++) {
    const yStart = w * rowsPerWorker;
    const yEnd = Math.min(frame.height, yStart + rowsPerWorker);

    jobs.push(
      runWorker(workerFile, {
        width: frame.width,
        height: frame.height,
        srcBuffer: frame.buffer,
        dstBuffer: out.buffer,
        yStart,
        yEnd,
      }),
    );
  }

  await Promise.all(jobs);

  return out;
}

// ── Parallel Gaussian blur ────────────────────────────────────────────────────
// Splits both H and V passes across all available CPU cores using
// SharedArrayBuffer so workers write directly into the output frame.

function gaussianKernel1D(radius: number, sigma: number): number[] {
  const size = radius * 2 + 1;
  const k: number[] = new Array(size);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    k[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma));
    sum += k[i + radius];
  }
  return k.map((v) => v / sum);
}

export async function gaussianBlurParallel(
  frame: VisionFrame,
  radius = 1,
  sigma?: number,
  numWorkers = Math.max(1, cpus().length - 1),
): Promise<VisionFrame> {
  if (radius <= 0) {
    return frame.clone();
  }

  const { width, height, channels } = frame;
  const sig = sigma ?? radius * 0.6 + 0.4;
  const kernel = gaussianKernel1D(radius, sig);

  const workerFile = join(__dirname, '../workers/gaussian-blur.worker.js');

  // Temp and output frames — SharedArrayBuffer so workers write zero-copy
  const tmp = new VisionFrame(width, height, channels);
  const out = new VisionFrame(width, height, channels);

  const rowsPerWorker = Math.ceil(height / numWorkers);

  async function pass(srcBuf: SharedArrayBuffer, dstBuf: SharedArrayBuffer, direction: 'h' | 'v'): Promise<void> {
    const jobs: Promise<void>[] = [];
    for (let w = 0; w < numWorkers; w++) {
      const yStart = w * rowsPerWorker;
      const yEnd = Math.min(height, yStart + rowsPerWorker);
      if (yStart >= yEnd) {
        continue;
      }
      jobs.push(
        runWorker(workerFile, {
          srcBuf,
          dstBuf,
          width,
          height,
          channels,
          kernel,
          radius,
          direction,
          yStart,
          yEnd,
        }),
      );
    }
    await Promise.all(jobs);
  }

  // H pass: frame.buffer → tmp.buffer
  // V pass: tmp.buffer   → out.buffer
  await pass(frame.buffer, tmp.buffer, 'h');
  await pass(tmp.buffer, out.buffer, 'v');

  return out;
}

// ── Parallel resize ───────────────────────────────────────────────────────────
//
// Threshold: workers add ~30ms startup each. Only worth spawning when the
// sequential resize would take longer than total worker overhead.
// Rule of thumb: use workers when output pixels > RESIZE_WORKER_THRESHOLD.
//
// Method auto-selection:
//   scale < 0.5 in BOTH axes → 'area'    (box filter, no aliasing for >2× downscale)
//   otherwise                → 'bilinear' (Q15 interpolation, good for upscale/mild downscale)

const RESIZE_WORKER_THRESHOLD_PX = 3_000_000;   // ~3MP output (e.g. 2000×1500)

export async function resizeParallel(
  frame: VisionFrame,
  newW: number,
  newH: number,
  method?: 'bilinear' | 'area',
  numWorkers = Math.max(1, cpus().length - 1),
): Promise<VisionFrame> {
  const { width, height, channels } = frame;
  if (newW <= 0 || newH <= 0) throw new Error('resizeParallel: dimensions must be positive');

  const scaleX = newW / width;
  const scaleY = newH / height;

  // Auto-select method: area for significant downscaling (avoids aliasing),
  // bilinear for everything else.
  const chosenMethod = method ?? (scaleX < 0.5 && scaleY < 0.5 ? 'area' : 'bilinear');

  // For small outputs, worker startup cost (30ms×N) exceeds the computation.
  // Fall back to the sequential implementation imported from geometry.ts.
  if (newW * newH < RESIZE_WORKER_THRESHOLD_PX) {
    const { resizeBilinear, resizeArea } = await import('./geometry.js');
    return chosenMethod === 'area'
      ? resizeArea(frame, newW, newH)
      : resizeBilinear(frame, newW, newH);
  }

  const out = new VisionFrame(newW, newH, channels as 1 | 3 | 4);
  const workerFile = join(__dirname, '../workers/resize.worker.js');
  const rowsPerWorker = Math.ceil(newH / numWorkers);
  const jobs: Promise<void>[] = [];

  if (chosenMethod === 'bilinear') {
    // Pre-compute coordinate maps once on the main thread, share via SAB.
    // Workers receive zero-copy views — no per-worker coordinate arithmetic.
    const x0SAB = new SharedArrayBuffer(newW * 4);
    const x1SAB = new SharedArrayBuffer(newW * 4);
    const wxSAB = new SharedArrayBuffer(newW * 4);
    const y0SAB = new SharedArrayBuffer(newH * 4);
    const y1SAB = new SharedArrayBuffer(newH * 4);
    const wySAB = new SharedArrayBuffer(newH * 4);

    const x0A = new Int32Array(x0SAB), x1A = new Int32Array(x1SAB), wxA = new Int32Array(wxSAB);
    const y0A = new Int32Array(y0SAB), y1A = new Int32Array(y1SAB), wyA = new Int32Array(wySAB);

    for (let x = 0; x < newW; x++) {
      const fx = (x + 0.5) * width / newW - 0.5;
      const x0 = Math.max(0, Math.floor(fx));
      const x1 = Math.min(width - 1, x0 + 1);
      x0A[x] = x0 * channels;   // store as byte offset directly
      x1A[x] = x1 * channels;
      wxA[x] = ((fx - x0) * 32768 + 0.5) | 0;
    }
    for (let y = 0; y < newH; y++) {
      const fy = (y + 0.5) * height / newH - 0.5;
      const y0 = Math.max(0, Math.floor(fy));
      const y1 = Math.min(height - 1, y0 + 1);
      y0A[y] = y0;   // row index (worker multiplies by srcStride)
      y1A[y] = y1;
      wyA[y] = ((fy - y0) * 32768 + 0.5) | 0;
    }

    for (let w = 0; w < numWorkers; w++) {
      const rowStart = w * rowsPerWorker;
      const rowEnd = Math.min(newH, rowStart + rowsPerWorker);
      if (rowStart >= rowEnd) continue;
      jobs.push(runWorker(workerFile, {
        method: 'bilinear',
        srcBuf: frame.buffer, dstBuf: out.buffer,
        srcW: width, dstW: newW, channels,
        x0Buf: x0SAB, x1Buf: x1SAB, wxBuf: wxSAB,
        y0Buf: y0SAB, y1Buf: y1SAB, wyBuf: wySAB,
        rowStart, rowEnd,
      }));
    }
  } else {
    // Area: each worker gets scale factors and computes independently.
    const aScaleX = width / newW;
    const aScaleY = height / newH;

    for (let w = 0; w < numWorkers; w++) {
      const rowStart = w * rowsPerWorker;
      const rowEnd = Math.min(newH, rowStart + rowsPerWorker);
      if (rowStart >= rowEnd) continue;
      jobs.push(runWorker(workerFile, {
        method: 'area',
        srcBuf: frame.buffer, dstBuf: out.buffer,
        srcW: width, srcH: height, dstW: newW, channels,
        scaleX: aScaleX, scaleY: aScaleY,
        rowStart, rowEnd,
      }));
    }
  }

  await Promise.all(jobs);
  return out;
}