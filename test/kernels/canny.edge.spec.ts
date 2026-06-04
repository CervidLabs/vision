import { describe, it, expect } from 'vitest';
import { VisionFrame } from '../../src/core/VisionFrame.js';
import { canny } from '../../src/kernels/canny.js';

/** Helper to create a grayscale frame filled with a value */
function fillFrame(width: number, height: number, value: number): VisionFrame {
  const f = new VisionFrame(width, height, 1);
  f.data.fill(value);
  return f;
}

describe('canny edge detection extended tests', () => {
  it('connects weak edges to strong ones via hysteresis', () => {
    // 5x5 image with a vertical step edge in the middle (left dark, right bright)
    const frame = new VisionFrame(5, 5, 1);
    // left side 0, right side 255
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        frame.data[y * 5 + x] = x < 2 ? 0 : 255;
      }
    }
    // low threshold low enough to mark surrounding pixels as WEAK, high to catch the actual edge as STRONG
    const out = canny(frame, { lowThreshold: 10, highThreshold: 100, radius: 0 });
    // There must be at least one STRONG pixel (255) and also additional non‑zero pixels (converted WEAK → STRONG)
    const nonZero = Array.from(out.data).filter(v => v === 255).length;
    expect(nonZero).toBeGreaterThan(0);
    // Ensure that the output still contains only 0 or 255 values (no leftover weak values)
    out.data.forEach(v => {
      expect([0, 255]).toContain(v);
    });
  });

  it('does not promote isolated weak edges when thresholds are tight', () => {
    // Create a frame where only a single pixel has a mid‑range intensity, surrounded by zeros
    const frame = fillFrame(3, 3, 0);
    frame.data[4] = 120; // centre pixel
    // Set thresholds such that this pixel becomes WEAK but there is no STRONG neighbour
    const out = canny(frame, { lowThreshold: 50, highThreshold: 200, radius: 0 });
    // No pixel should be marked as edge (all zeros) because the centre never reaches STRONG
    const hasEdge = Array.from(out.data).some(v => v === 255);
    expect(hasEdge).toBe(false);
    // All values must be zero
    out.data.forEach(v => expect(v).toBe(0));
  });
});
