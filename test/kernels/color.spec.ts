import { describe, it, expect } from 'vitest';
import { VisionFrame } from '../../src/core/VisionFrame.js';
import {
  brightnessContrast,
  gamma,
  extractChannel,
  histogram,
  equalizeHistogram,
  normalize,
  rgbToHSV,
  hsvToRGB,
  inRangeHSV,
  inRangeGray,
  inRangeRGB,
} from '../../src/kernels/color.js';

describe('color.ts kernels', () => {
  describe('brightnessContrast', () => {
    it('adjusts brightness and contrast correctly', () => {
      const frame = new VisionFrame(2, 2, 1);
      frame.data.set([100, 150, 0, 255]);

      // brightness + 10
      const b10 = brightnessContrast(frame, 10, 1);
      expect(Array.from(b10.data)).toEqual([110, 160, 10, 255]);

      // contrast 2.0 (stretches around 128)
      // (100 - 128)*2 + 128 = 72
      // (150 - 128)*2 + 128 = 172
      // (0 - 128)*2 + 128 = -128 -> 0
      // (255 - 128)*2 + 128 = 382 -> 255
      const c2 = brightnessContrast(frame, 0, 2);
      expect(Array.from(c2.data)).toEqual([72, 172, 0, 255]);
    });
  });

  describe('gamma', () => {
    it('throws if gamma <= 0', () => {
      const frame = new VisionFrame(1, 1, 1);
      expect(() => gamma(frame, 0)).toThrow('gamma must be > 0');
      expect(() => gamma(frame, -1)).toThrow('gamma must be > 0');
    });

    it('applies gamma correction', () => {
      const frame = new VisionFrame(2, 1, 1);
      frame.data.set([64, 128]);

      // gamma 2.0
      const g2 = gamma(frame, 2);
      // Math.pow(64/255, 0.5)*255 ≈ 128
      // Math.pow(128/255, 0.5)*255 ≈ 181
      expect(g2.data[0]).toBeCloseTo(128, -1);
      expect(g2.data[1]).toBeCloseTo(181, -1);
    });
  });

  describe('extractChannel', () => {
    it('throws if channel out of bounds', () => {
      const frame = new VisionFrame(1, 1, 3);
      expect(() => extractChannel(frame, 3)).toThrow('Channel 3 out of range');
    });

    it('extracts correct channel', () => {
      const frame = new VisionFrame(2, 1, 3);
      frame.data.set([10, 20, 30, 40, 50, 60]);

      const g = extractChannel(frame, 1);
      expect(g.channels).toBe(1);
      expect(g.width).toBe(2);
      expect(Array.from(g.data)).toEqual([20, 50]);
    });
  });

  describe('histogram', () => {
    it('computes per-channel histogram', () => {
      const frame = new VisionFrame(2, 1, 3);
      frame.data.set([10, 20, 0, 10, 30, 0]);

      const hists = histogram(frame);
      expect(hists.length).toBe(3);
      expect(hists[0][10]).toBe(2);
      expect(hists[0][20]).toBe(0);
      expect(hists[1][20]).toBe(1);
      expect(hists[1][30]).toBe(1);
    });
  });

  describe('equalizeHistogram', () => {
    it('throws if not 1-channel', () => {
      const frame = new VisionFrame(1, 1, 3);
      expect(() => equalizeHistogram(frame)).toThrow('requires a 1-channel grayscale frame');
    });

    it('equalizes histogram', () => {
      const frame = new VisionFrame(3, 1, 1);
      frame.data.set([100, 100, 150]);
      // hist: 100->2, 150->1
      // cdf: 100->2, 150->3. cdfMin=2
      // lut[100] = round((2-2)/(3-2)*255) = 0
      // lut[150] = round((3-2)/(3-2)*255) = 255
      const eq = equalizeHistogram(frame);
      expect(Array.from(eq.data)).toEqual([0, 0, 255]);
    });

    it('handles flat histogram safely', () => {
      const frame = new VisionFrame(2, 1, 1);
      frame.data.set([0, 0]);
      const eq = equalizeHistogram(frame);
      expect(Array.from(eq.data)).toEqual([0, 0]);
    });

  });

  describe('normalize', () => {
    it('stretches to 0-255 per channel', () => {
      const frame = new VisionFrame(2, 1, 3);
      frame.data.set([50, 100, 0, 100, 150, 0]);
      // ch0: min=50, max=100 -> range=50
      // ch1: min=100, max=150 -> range=50
      // ch2: min=0, max=0 -> range=0 (becomes 1)
      const norm = normalize(frame);
      expect(Array.from(norm.data)).toEqual([0, 0, 0, 255, 255, 0]);
    });

    it('handles uniform channel without division by zero', () => {
      const frame = new VisionFrame(2, 1, 1);
      frame.data.set([100, 100]);
      const norm = normalize(frame);
      expect(Array.from(norm.data)).toEqual([0, 0]);
    });
  });

  describe('rgbToHSV and hsvToRGB', () => {
    it('rgbToHSV throws if not 3 or 4 channels', () => {
      expect(() => rgbToHSV(new VisionFrame(1, 1, 1))).toThrow('expected 3 or 4-channel');
    });
    it('hsvToRGB throws if not 3 channels', () => {
      expect(() => hsvToRGB(new VisionFrame(1, 1, 1))).toThrow('expected 3-channel HSV');
    });

    it('covers all HSV hue branches', () => {
      const rgb = new VisionFrame(7, 1, 3);
      // R=max, G=max, B=max, and a gray pixel for delta=0, and black pixel for max=0
      // Red, Green, Blue, Yellow (R=G), Magenta (R=B), Gray, Black
      rgb.data.set([
        255, 0, 0,       // Red (h=0)
        0, 255, 0,       // Green (h=60)
        0, 0, 255,       // Blue (h=120)
        255, 255, 0,     // Yellow (h=30)
        255, 0, 255,     // Magenta (h=150)
        128, 128, 128,   // Gray (delta=0)
        0, 0, 0          // Black (max=0)
      ]);

      const hsv = rgbToHSV(rgb);
      // H in OpenCV [0, 179]
      // S, V in [0, 255]
      // Red: H=0, S=255, V=255
      expect(hsv.data[0]).toBe(0);
      expect(hsv.data[1]).toBe(255);
      expect(hsv.data[2]).toBe(255);

      // Green: H=120/360*179 ≈ 60
      expect(hsv.data[3]).toBe(60);

      // Blue: H=240/360*179 ≈ 119
      expect(hsv.data[6]).toBe(119);

      // Yellow: H=60/360*179 ≈ 30
      // Max is R (or G), so covers R=G branch
      expect(hsv.data[9]).toBe(30);

      // Magenta: H=300/360*179 ≈ 149
      // R-G/delta + 4 branch
      expect(hsv.data[12]).toBe(149);

      // Gray: delta=0
      expect(hsv.data[15]).toBe(0); // H=0
      expect(hsv.data[16]).toBe(0); // S=0
      expect(hsv.data[17]).toBe(128); // V=128

      // Black: max=0
      expect(hsv.data[18]).toBe(0); // H=0
      expect(hsv.data[19]).toBe(0); // S=0
      expect(hsv.data[20]).toBe(0); // V=0

      // Now reverse hsvToRGB to cover all hsvToRGB branches
      const back = hsvToRGB(hsv);
      // Should be roughly equal to original
      for (let i = 0; i < rgb.data.length; i++) {
        expect(Math.abs(rgb.data[i] - back.data[i])).toBeLessThanOrEqual(5);
      }
    });

    it('covers H < 0 branch in rgbToHSV', () => {
      const rgb = new VisionFrame(1, 1, 3);
      rgb.data.set([255, 0, 200]); // Max=R, G < B -> H becomes negative initially
      const hsv = rgbToHSV(rgb);
      // H should be > 0 and correct
      expect(hsv.data[0]).toBeGreaterThan(100);
    });

    it('covers all hsvToRGB hue segments', () => {
      const hsv = new VisionFrame(6, 1, 3);
      // H: 10, 40, 70, 100, 130, 160 (scaled 0-179)
      // This corresponds to approx 20, 80, 140, 200, 260, 320 degrees
      hsv.data.set([
        10, 255, 255, // <60
        40, 255, 255, // <120
        70, 255, 255, // <180
        100, 255, 255, // <240
        130, 255, 255, // <300
        160, 255, 255  // else
      ]);
      const rgb = hsvToRGB(hsv);
      expect(rgb.channels).toBe(3);
      // Just verify it doesn't crash and outputs sensible numbers
      expect(rgb.data[0]).toBe(255); // Red segment max R
    });
  });

  describe('inRangeHSV', () => {
    it('throws if not 3 channels', () => {
      expect(() => inRangeHSV(new VisionFrame(1, 1, 1), {})).toThrow('expected 3-channel HSV frame');
    });

    it('filters based on range without wrap', () => {
      const hsv = new VisionFrame(2, 1, 3);
      hsv.data.set([10, 100, 100, 90, 100, 100]); // H=10, H=90
      const mask = inRangeHSV(hsv, { h: [0, 50], s: [50, 255], v: [50, 255] });
      expect(Array.from(mask.data)).toEqual([255, 0]);
    });

    it('filters based on range WITH wrap (reds)', () => {
      const hsv = new VisionFrame(3, 1, 3);
      // H=170, H=10, H=90
      hsv.data.set([170, 100, 100, 10, 100, 100, 90, 100, 100]);
      // Range [160, 20]
      const mask = inRangeHSV(hsv, { h: [160, 20] });
      expect(Array.from(mask.data)).toEqual([255, 255, 0]);
    });

    it('filters when h range is omitted', () => {
      const hsv = new VisionFrame(2, 1, 3);
      hsv.data.set([10, 100, 100, 90, 0, 100]);
      const mask = inRangeHSV(hsv, { s: [50, 255] });
      expect(Array.from(mask.data)).toEqual([255, 0]);
    });
  });

  describe('inRangeGray', () => {
    it('throws if not 1 channel', () => {
      expect(() => inRangeGray(new VisionFrame(1, 1, 3), 0, 255)).toThrow('expected 1-channel frame');
    });

    it('filters gray correctly', () => {
      const frame = new VisionFrame(3, 1, 1);
      frame.data.set([50, 150, 250]);
      const mask = inRangeGray(frame, 100, 200);
      expect(Array.from(mask.data)).toEqual([0, 255, 0]);
    });
  });

  describe('inRangeRGB', () => {
    it('throws if not 3 or 4 channels', () => {
      expect(() => inRangeRGB(new VisionFrame(1, 1, 1), {})).toThrow('expected 3 or 4-channel');
    });

    it('filters RGB correctly', () => {
      const frame = new VisionFrame(2, 1, 3);
      frame.data.set([100, 50, 0, 200, 200, 200]);

      const mask = inRangeRGB(frame, { r: [90, 110], g: [0, 60] }); // No B range -> unrestricted
      expect(Array.from(mask.data)).toEqual([255, 0]);
    });
  });
});
