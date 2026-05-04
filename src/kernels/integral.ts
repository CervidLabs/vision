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
}

function assertGray(frame: VisionFrame, name: string): void {
  if (frame.channels !== 1) {
    throw new Error(`${name}: expected 1-channel grayscale frame`);
  }
}

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

  return {
    width: iw,
    height: ih,
    data,
  };
}

export function sumRect(integral: IntegralImage, x: number, y: number, width: number, height: number): number {
  const x0 = x;
  const y0 = y;
  const x1 = x + width;
  const y1 = y + height;
  const iw = integral.width;
  const d = integral.data;

  return d[y1 * iw + x1] - d[y0 * iw + x1] - d[y1 * iw + x0] + d[y0 * iw + x0];
}

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
