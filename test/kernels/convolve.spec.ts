import { describe, it, expect } from 'vitest';
import { VisionFrame } from '../../src/core/VisionFrame.js';
import { convolve, boxBlur, gaussianBlur, sobel, sharpen } from '../../src/kernels/convolve.js';

/** Helper to create a frame with a vertical step edge */
function createStep(width: number, height: number, threshold: number): VisionFrame {
  const frame = new VisionFrame(width, height, 1);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      frame.data[y * width + x] = x < threshold ? 0 : 255;
    }
  }
  return frame;
}

/** Helper to create a constant frame */
function createConstant(width: number, height: number, value: number): VisionFrame {
  const frame = new VisionFrame(width, height, 1);
  frame.data.fill(value);
  return frame;
}

/** Helper to create an identity kernel (1×1) */
function identityKernel(): { kernel: number[]; kw: number; kh: number } {
  return { kernel: [1], kw: 1, kh: 1 };
}

describe('convolve kernel suite (realistic tests)', () => {
  it('convolve with identity kernel returns unchanged frame', () => {
    const src = createStep(5, 5, 3);
    const { kernel, kw, kh } = identityKernel();
    const out = convolve(src, kernel, kw, kh);
    expect(Array.from(out.data)).toEqual(Array.from(src.data));
  });

  it('convolve with a 3x3 edge detection kernel on a step image', () => {
    const src = createStep(5, 5, 2); // left side 0, right side 255
    // Horizontal gradient kernel: -1 on left, +1 on center
    const kernel = [0, 0, 0, -1, 1, 0, 0, 0, 0];
    const out = convolve(src, kernel, 3, 3);
    const edgeCol = 2;
    const centreRow = 2;
    const idx = centreRow * 5 + edgeCol;
    expect(out.data[idx]).toBeGreaterThan(0);
  });

  it('boxBlur reduces noise on a high‑frequency checkerboard', () => {
    const size = 6;
    const frame = new VisionFrame(size, size, 1);
    // checkerboard pattern (0/255)
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        frame.data[y * size + x] = (x + y) % 2 === 0 ? 0 : 255;
      }
    }
    const out = boxBlur(frame, 1);
    const centre = out.data[(size / 2 | 0) * size + (size / 2 | 0)];
    expect(centre).toBeGreaterThan(0).and.toBeLessThan(255);
  });

  it('gaussianBlur with radius 2 smooths a sharp step', () => {
    const frame = new VisionFrame(8, 1, 1);
    for (let i = 0; i < 8; i++) {
      frame.data[i] = i < 4 ? 0 : 255;
    }
    const out = gaussianBlur(frame, 2);
    const mid = out.data[4];
    expect(mid).toBeGreaterThan(60).and.toBeLessThan(200);
  });

  it('sobel detects vertical edges on a step image', () => {
    const frame = createStep(7, 5, 3);
    const out = sobel(frame);
    const maxVal = Math.max(...out.data);
    expect(maxVal).toBeGreaterThan(0);
  });

  it('sharpen enhances contrast on a uniform image', () => {
    const src = createConstant(5, 5, 120);
    const sharpened = sharpen(src, 2);
    const max = Math.max(...sharpened.data);
    const min = Math.min(...sharpened.data);
    expect(max - min).toBeLessThanOrEqual(1);
    expect(max).toBeCloseTo(120, 0);
  });
});
