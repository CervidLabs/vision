import { describe, it, expect } from 'vitest';
import { hysteresis } from '../../src/kernels/canny.js';

/** Helper to create Float32Array from 2D array */
function createNMS(values: number[][]): Float32Array {
  const height = values.length;
  const width = values[0].length;
  const arr = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      arr[y * width + x] = values[y][x];
    }
  }
  return arr;
}

describe('canny hysteresis internal tests', () => {
  it('converts adjacent weak pixel to strong and handles out‑of‑bounds checks', () => {
    // 2x2 grid: strong at top‑left, weak at top‑right (adjacent)
    const nms = createNMS([
      [30, 15], // row 0
      [0, 0],   // row 1
    ]);
    const width = 2;
    const height = 2;
    const low = 10;
    const high = 20;
    const out = hysteresis(nms, width, height, low, high);
    // Expect both pixels become strong (255)
    expect(out[0]).toBe(255);
    expect(out[1]).toBe(255);
    // The out‑of‑bounds branch (dx=-1 or dy=-1) is exercised during neighbor checks
  });

  it('removes isolated weak edges that are not connected to any strong pixel', () => {
    // 3x3 grid: strong at (0,0), weak at (2,2) (diagonal distance >1)
    const nms = createNMS([
      [30, 0, 0],
      [0, 0, 0],
      [0, 0, 15],
    ]);
    const width = 3;
    const height = 3;
    const low = 10;
    const high = 20;
    const out = hysteresis(nms, width, height, low, high);
    // Strong pixel should stay 255
    expect(out[0]).toBe(255);
    // Isolated weak pixel should be cleared to 0 (line 146)
    expect(out[8]).toBe(0);
    // Ensure no other pixels are marked as edges
    out.forEach((v, i) => {
      if (i !== 0 && i !== 8) {
        expect(v).toBe(0);
      }
    });
  });
});
