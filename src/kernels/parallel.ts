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
