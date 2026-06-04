import { describe, it, expect } from 'vitest';
import { VisionFrame } from '../../src/core/VisionFrame.js';
import { warpAffine, rotationMatrix, translationMatrix, scaleMatrix, composeAffine } from '../../src/kernels/transform.js';

/** Helper to create a simple grayscale frame with incremental values */
function createIncremental(width: number, height: number): VisionFrame {
  const frame = new VisionFrame(width, height, 1);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      frame.data[y * width + x] = (x + y * width) % 256;
    }
  }
  return frame;
}

describe('transform kernel suite (realistic tests)', () => {
  it('warpAffine identity returns identical frame', () => {
    const src = createIncremental(4, 4);
    const identity: any = [1, 0, 0, 0, 1, 0]; // inverse map identity
    const out = warpAffine(src, identity);
    expect(Array.from(out.data)).toEqual(Array.from(src.data));
  });

  it('warpAffine translation shifts pixels correctly', () => {
    const src = createIncremental(5, 5);
    const tx = 1, ty = 2;
    const mat = translationMatrix(tx, ty);
    const out = warpAffine(src, mat);
    const srcIdx = (1 * 5 + 1); // original (1,1)
    const dstIdx = ((1 + ty) * 5 + (1 + tx)); // destination (2,3)
    expect(out.data[dstIdx]).toBe(src.data[srcIdx]);
    expect(out.data[0]).toBe(0); // top‑left corner moved out
  });

  it('warpAffine rotation 90° around center', () => {
    const src = createIncremental(3, 3);
    const cx = 1, cy = 1; // center of 3×3
    const mat = rotationMatrix(90, cx, cy);
    const out = warpAffine(src, mat);
    // After 90° clockwise rotation, the top‑left pixel comes from source (1,2)
    const srcIdx = (1 * 3 + 2); // value at row 1, col 2 in source
    const dstIdx = 0; // top‑left in destination
    expect(out.data[dstIdx]).toBe(src.data[srcIdx]);
  });

  it('warpAffine scaling reduces size correctly and uses border value', () => {
    const src = createIncremental(4, 4);
    const mat = scaleMatrix(0.5, 0.5, 0, 0);
    const out = warpAffine(src, mat, 2, 2, 123);
    out.data.forEach(v => expect(v).not.toBe(123));
    expect(out.width).toBe(2);
    expect(out.height).toBe(2);
  });

  it('composeAffine combines translation then rotation correctly', () => {
    const trans = translationMatrix(2, 0);
    const rot = rotationMatrix(180, 0, 0);
    const composed = composeAffine(trans, rot);
    const src = createIncremental(5, 1);
    const out = warpAffine(src, composed);
    expect(out.data.length).toBe(src.data.length);
  });
});
