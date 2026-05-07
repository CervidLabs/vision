import { VisionFrame } from '../core/VisionFrame.js';

export interface IntegralImage {
  width: number;
  height: number;
  data: Float64Array;
}

export interface AdaptiveThresholdOptions {
  blockSize?: number;
  c?: number;
  invert?: boolean;
  /** Weighting method. 'mean' (default) or 'gaussian' for better document scans. */
  method?: 'mean' | 'gaussian';
}

function assertGray(frame: VisionFrame, name: string): void {
  if (frame.channels !== 1) {
    throw new Error(`${name}: expected 1-channel grayscale frame`);
  }
}

// ── Integral image ────────────────────────────────────────────────────────────

export function integralImage(frame: VisionFrame): IntegralImage {
  assertGray(frame, 'integralImage');

  const { width, height } = frame;
  const src = frame.data;
  const iw = width + 1;
  const ih = height + 1;
  const data = new Float64Array(iw * ih);

  for (let y = 1; y <= height; y++) {
    let rowSum = 0;
    const srcRow = (y - 1) * width;
    const curRow = y * iw;
    const prevRow = (y - 1) * iw;

    for (let x = 1; x <= width; x++) {
      rowSum += src[srcRow + x - 1];
      data[curRow + x] = data[prevRow + x] + rowSum;
    }
  }

  return { width: iw, height: ih, data };
}

export function sumRect(integral: IntegralImage, x: number, y: number, width: number, height: number): number {
  const x0 = x,
    y0 = y,
    x1 = x + width,
    y1 = y + height;
  const iw = integral.width;
  const d = integral.data;
  return d[y1 * iw + x1] - d[y0 * iw + x1] - d[y1 * iw + x0] + d[y0 * iw + x0];
}

// ── Mean adaptive threshold (original) ───────────────────────────────────────

function adaptiveThresholdMean(frame: VisionFrame, blockSize: number, c: number, invert: boolean): VisionFrame {
  const { width, height } = frame;
  const src = frame.data;
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;
  const ii = integralImage(frame);
  const r = (blockSize / 2) | 0;

  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(height - 1, y + r);
    const h = y1 - y0 + 1;

    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(width - 1, x + r);
      const w = x1 - x0 + 1;

      const mean = sumRect(ii, x0, y0, w, h) / (w * h);
      const pass = src[y * width + x] > mean - c;
      dst[y * width + x] = invert ? (pass ? 0 : 255) : pass ? 255 : 0;
    }
  }

  return out;
}

// ── Gaussian adaptive threshold ───────────────────────────────────────────────
//
// Uses a precomputed Gaussian-weighted local mean via a 2D Gaussian kernel.
// More accurate than mean-based for document scanning / text extraction.
// The Gaussian is applied via separable 1D convolution on the grayscale image.

function gaussianKernel1D(radius: number): Float64Array {
  const sigma = radius / 3; // tighter sigma gives more local response
  const size = radius * 2 + 1;
  const k = new Float64Array(size);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    k[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma));
    sum += k[i + radius];
  }
  for (let i = 0; i < size; i++) {
    k[i] /= sum;
  }
  return k;
}

function separableFilter(src: Uint8Array, width: number, height: number, kernel: Float64Array, radius: number): Float64Array {
  const tmp = new Float64Array(width * height);
  const out = new Float64Array(width * height);

  // Horizontal pass
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const sx = Math.max(0, Math.min(width - 1, x + k));
        acc += src[row + sx] * kernel[k + radius];
      }
      tmp[row + x] = acc;
    }
  }

  // Vertical pass
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const sy = Math.max(0, Math.min(height - 1, y + k));
        acc += tmp[sy * width + x] * kernel[k + radius];
      }
      out[row + x] = acc;
    }
  }

  return out;
}

function adaptiveThresholdGaussian(frame: VisionFrame, blockSize: number, c: number, invert: boolean): VisionFrame {
  const { width, height } = frame;
  const src = frame.data;
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;

  const radius = (blockSize / 2) | 0;
  const kernel = gaussianKernel1D(radius);
  const localMean = separableFilter(src, width, height, kernel, radius);

  for (let i = 0; i < width * height; i++) {
    const pass = src[i] > localMean[i] - c;
    dst[i] = invert ? (pass ? 0 : 255) : pass ? 255 : 0;
  }

  return out;
}

// ── Public API ────────────────────────────────────────────────────────────────

export function adaptiveThreshold(frame: VisionFrame, options: AdaptiveThresholdOptions = {}): VisionFrame {
  assertGray(frame, 'adaptiveThreshold');

  let blockSize = options.blockSize ?? 15;
  if (blockSize < 3) {
    blockSize = 3;
  }
  if (blockSize % 2 === 0) {
    blockSize++;
  }

  const c = options.c ?? 5;
  const invert = options.invert ?? false;
  const method = options.method ?? 'mean';

  if (method === 'gaussian') {
    return adaptiveThresholdGaussian(frame, blockSize, c, invert);
  }

  return adaptiveThresholdMean(frame, blockSize, c, invert);
}
