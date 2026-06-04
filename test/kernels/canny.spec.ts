import { describe, it, expect } from 'vitest';
import { VisionFrame } from '../../src/core/VisionFrame.js';
import { canny } from '../../src/kernels/canny.js';

describe('canny.ts kernels', () => {
  it('throws if input is not grayscale', () => {
    const rgb = new VisionFrame(2, 1, 3);
    rgb.data.set([255, 0, 0, 0, 255, 0]);
    expect(() => canny(rgb)).toThrow('canny: requires grayscale (1-channel) input');
  });

  it('detects edges on a simple vertical gradient', () => {
    // 3x3 image: left column 0, right columns 255 – clear vertical edge
    const frame = new VisionFrame(3, 3, 1);
    frame.data.set([
      0, 0, 255,
      0, 0, 255,
      0, 0, 255,
    ]);
    const out = canny(frame, { lowThreshold: 10, highThreshold: 100, radius: 0 });
    // Expect output length matches input and contains at least one edge pixel (255)
    expect(out.data.length).toBe(9);
    expect(Array.from(out.data).some(v => v === 255)).toBe(true);
  });
});
