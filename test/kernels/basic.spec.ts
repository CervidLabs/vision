import { describe, it, expect } from 'vitest';
import { VisionFrame } from '../../src/core/VisionFrame.js';
import { grayscale, threshold, grayscaleToRGB, invert } from '../../src/kernels/basic.js';

describe('basic.ts kernels', () => {
  it('grayscale conversion works', () => {
    const src = new VisionFrame(2, 1, 3);
    src.data.set([255, 0, 0,   // red
      0, 255, 0]); // green
    const out = grayscale(src);
    // BT.601 integer approximation: (77R + 150G + 29B) >> 8
    // Red yields 76, Green yields 149
    expect(Array.from(out.data)).toEqual([76, 149]);
  });



  it('grayscale throws on non‑RGB', () => {
    const src = new VisionFrame(1, 1, 1);
    expect(() => grayscale(src)).toThrow('grayscale requires RGB or RGBA input');
  });

  it('threshold default and custom', () => {
    const src = new VisionFrame(2, 1, 1);
    src.data.set([100, 200]);
    const outDefault = threshold(src);
    const outCustom = threshold(src, 150);
    expect(Array.from(outDefault.data)).toEqual([0, 255]);
    expect(Array.from(outCustom.data)).toEqual([0, 255]);
  });

  it('grayscaleToRGB expands single channel', () => {
    const src = new VisionFrame(2, 1, 1);
    src.data.set([42, 255]);
    const out = grayscaleToRGB(src);
    expect(Array.from(out.data)).toEqual([42, 42, 42, 255, 255, 255]);
  });

  it('invert flips values', () => {
    const src = new VisionFrame(2, 1, 1);
    src.data.set([0, 255]);
    const out = invert(src);
    expect(Array.from(out.data)).toEqual([255, 0]);
  });
});
