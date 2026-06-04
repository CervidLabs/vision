import { describe, it, expect } from 'vitest';
import { VisionFrame } from '../../src/core/VisionFrame.js';
import { convolve, boxBlur, gaussianBlur, sharpen } from '../../src/kernels/convolve.js';

/** Helper to create a simple frame */
function createFrame(width: number, height: number, channels: number, fillFn: (x: number, y: number, c: number) => number): VisionFrame {
  const frame = new VisionFrame(width, height, channels);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < channels; c++) {
        frame.data[(y * width + x) * channels + c] = fillFn(x, y, c);
      }
    }
  }
  return frame;
}

describe('convolve additional edge‑case tests', () => {
  it('throws when kernel dimensions are even', () => {
    const frame = createFrame(3, 3, 1, () => 1);
    const kernel = [1, 1, 1, 1]; // 2×2 (even)
    expect(() => convolve(frame, kernel, 2, 2)).toThrow('Kernel dimensions must be odd');
  });

  it('throws when kernel length does not match kw*kh', () => {
    const frame = createFrame(3, 3, 1, () => 1);
    const kernel = [1, 2, 3]; // length 3 but kw*kh = 9
    expect(() => convolve(frame, kernel, 3, 3)).toThrow('Kernel length must equal kw * kh');
  });

  it('boxBlur with radius < 1 returns a clone (different reference but identical data)', () => {
    const src = createFrame(2, 2, 1, (x, y) => x + y * 2);
    const out = boxBlur(src, 0);
    expect(out).not.toBe(src);
    expect(Array.from(out.data)).toEqual(Array.from(src.data));
  });

  it('gaussianBlur with radius <= 0 returns a clone', () => {
    const src = createFrame(3, 3, 1, (x, y) => x + y * 3);
    const out = gaussianBlur(src, 0);
    expect(out).not.toBe(src);
    expect(Array.from(out.data)).toEqual(Array.from(src.data));
  });

  it('convolve with identity kernel on multi‑channel frame leaves data unchanged', () => {
    const src = createFrame(2, 2, 3, (x, y, c) => x + y * 2 + c * 10);
    const { kernel, kw, kh } = { kernel: [1], kw: 1, kh: 1 };
    const out = convolve(src, kernel, kw, kh);
    expect(Array.from(out.data)).toEqual(Array.from(src.data));
  });

  it('convolve on a 2×2 frame with a 3×3 averaging kernel triggers border clamping', () => {
    // 3×3 kernel of all 1's (will be averaged by clampU8 after sum)
    const kernel = Array(9).fill(1);
    const src = createFrame(2, 2, 1, (x, y) => x + y * 2); // values: 0,1,2,3
    const out = convolve(src, kernel, 3, 3);
    // Manual calculation with clamped edges (replicate border pixels)
    // For pixel (0,0): neighbours sum = 9 -> clampU8(9) = 9
    // For pixel (0,1): neighbours sum = 12 -> 12
    // For pixel (1,0): neighbours sum = 15 -> 15
    // For pixel (1,1): neighbours sum = 18 -> 18
    const expected = [9, 12, 15, 18];
    expect(Array.from(out.data)).toEqual(expected);
  });

  it('gaussianBlur on a 3‑channel frame returns a clone when radius <= 0', () => {
    const src = createFrame(2, 2, 3, (x, y, c) => x + y * 2 + c * 10);
    const out = gaussianBlur(src, 0);
    expect(out).not.toBe(src);
    expect(Array.from(out.data)).toEqual(Array.from(src.data));
  });

  it('gaussianBlur on a 3‑channel frame with radius 1 produces values within 0‑255', () => {
    const src = createFrame(3, 3, 3, (x, y, c) => (x + y * 3) * (c + 1));
    const out = gaussianBlur(src, 1);
    // Ensure output has same dimensions and channel count
    expect(out.width).toBe(src.width);
    expect(out.height).toBe(src.height);
    expect(out.channels).toBe(3);
    // All values should be clamped to [0,255]
    out.data.forEach(v => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(255);
    });
  });

  it('sharpen on a simple gradient image enhances contrast', () => {
    const src = createFrame(3, 1, 1, (x) => (x === 1 ? 200 : 0)); // step pattern 0,200,0
    const sharpened = sharpen(src, 2);
    const originalMid = src.data[1];
    const sharpenedMid = sharpened.data[1];
    expect(sharpenedMid).toBeGreaterThan(originalMid);
    // Values still within valid range
    sharpened.data.forEach(v => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(255);
    });
  });
});
