import { clampU8, VisionFrame } from '../core/VisionFrame.js';

/** Adjust brightness (-255 to +255) and contrast (0.0 = flat, 1.0 = no change, >1 = more). */
export function brightnessContrast(frame: VisionFrame, brightness = 0, contrast = 1): VisionFrame {
  const out = new VisionFrame(frame.width, frame.height, frame.channels);
  const src = frame.data;
  const dst = out.data;

  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    lut[i] = clampU8((i - 128) * contrast + 128 + brightness);
  }
  for (let i = 0; i < src.length; i++) {
    dst[i] = lut[src[i]];
  }
  return out;
}

/** Gamma correction. gamma < 1 brightens; gamma > 1 darkens. */
export function gamma(frame: VisionFrame, g: number): VisionFrame {
  if (g <= 0) {
    throw new Error('gamma must be > 0');
  }

  const out = new VisionFrame(frame.width, frame.height, frame.channels);
  const src = frame.data;
  const dst = out.data;

  const lut = new Uint8Array(256);
  const inv = 1 / g;
  for (let i = 0; i < 256; i++) {
    lut[i] = clampU8(Math.pow(i / 255, inv) * 255);
  }
  for (let i = 0; i < src.length; i++) {
    dst[i] = lut[src[i]];
  }
  return out;
}

/** Extract a single channel (0=R, 1=G, 2=B, 3=A) as a 1-channel grayscale frame. */
export function extractChannel(frame: VisionFrame, channel: 0 | 1 | 2 | 3): VisionFrame {
  if (channel >= frame.channels) {
    throw new Error(`Channel ${channel} out of range for ${frame.channels}-channel frame`);
  }

  const out = new VisionFrame(frame.width, frame.height, 1);
  const src = frame.data;
  const dst = out.data;
  const ch = frame.channels;

  for (let i = 0, j = 0; i < src.length; i += ch, j++) {
    dst[j] = src[i + channel];
  }
  return out;
}

/**
 * Compute per-channel histogram(s).
 * Returns an array of Uint32Array(256) — one per channel.
 */
export function histogram(frame: VisionFrame): Uint32Array[] {
  const hists = Array.from({ length: frame.channels }, () => new Uint32Array(256));
  const src = frame.data;
  const ch = frame.channels;

  for (let i = 0; i < src.length; i++) {
    hists[i % ch][src[i]]++;
  }
  return hists;
}

/**
 * Histogram equalisation (grayscale 1-channel only).
 * Stretches contrast to use the full 0–255 range.
 */
export function equalizeHistogram(frame: VisionFrame): VisionFrame {
  if (frame.channels !== 1) {
    throw new Error('equalizeHistogram requires a 1-channel grayscale frame');
  }

  const src = frame.data;
  const n = src.length;

  const hist = new Uint32Array(256);
  for (let i = 0; i < n; i++) {
    hist[src[i]]++;
  }

  const cdf = new Uint32Array(256);
  let running = 0;
  for (let i = 0; i < 256; i++) {
    running += hist[i];
    cdf[i] = running;
  }

  const cdfMin = cdf.find((v) => v > 0) ?? 0;
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    lut[i] = clampU8(Math.round(((cdf[i] - cdfMin) / (n - cdfMin)) * 255));
  }

  const out = new VisionFrame(frame.width, frame.height, 1);
  const dst = out.data;
  for (let i = 0; i < n; i++) {
    dst[i] = lut[src[i]];
  }
  return out;
}

/** Normalize frame values to the full 0–255 range (per-channel). */
export function normalize(frame: VisionFrame): VisionFrame {
  const out = new VisionFrame(frame.width, frame.height, frame.channels);
  const src = frame.data;
  const dst = out.data;
  const ch = frame.channels;

  for (let c = 0; c < ch; c++) {
    let lo = 255,
      hi = 0;
    for (let i = c; i < src.length; i += ch) {
      if (src[i] < lo) {
        lo = src[i];
      }
      if (src[i] > hi) {
        hi = src[i];
      }
    }
    const range = hi - lo || 1;
    for (let i = c; i < src.length; i += ch) {
      dst[i] = clampU8(((src[i] - lo) / range) * 255);
    }
  }
  return out;
}

// ── HSV conversion ────────────────────────────────────────────────────────────
//
// H ∈ [0, 360), S ∈ [0, 255], V ∈ [0, 255]
// Stored as a 3-channel u8 frame: H scaled to [0,179] (×179/360), S [0,255], V [0,255]
// (same convention as OpenCV so downstream interop is easy)

/** Convert an RGB (3-channel) frame to HSV. Returns a 3-channel frame with H∈[0,179], S∈[0,255], V∈[0,255]. */
export function rgbToHSV(frame: VisionFrame): VisionFrame {
  if (frame.channels !== 3 && frame.channels !== 4) {
    throw new Error('rgbToHSV: expected 3 or 4-channel RGB/RGBA frame');
  }

  const { width, height, channels } = frame;
  const src = frame.data;
  const out = new VisionFrame(width, height, 3);
  const dst = out.data;
  const n = width * height;

  for (let i = 0; i < n; i++) {
    const p = i * channels;
    const r = src[p] / 255;
    const g = src[p + 1] / 255;
    const b = src[p + 2] / 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;

    let h = 0;
    if (delta > 0) {
      if (max === r) {
        h = ((g - b) / delta) % 6;
      } else if (max === g) {
        h = (b - r) / delta + 2;
      } else {
        h = (r - g) / delta + 4;
      }
      h *= 60;
      if (h < 0) {
        h += 360;
      }
    }

    const s = max === 0 ? 0 : delta / max;
    const v = max;

    const o = i * 3;
    dst[o] = (h * (179 / 360) + 0.5) | 0; // H: [0, 179]
    dst[o + 1] = (s * 255 + 0.5) | 0; // S: [0, 255]
    dst[o + 2] = (v * 255 + 0.5) | 0; // V: [0, 255]
  }

  return out;
}

/** Convert an HSV frame (H∈[0,179], S∈[0,255], V∈[0,255]) back to RGB. */
export function hsvToRGB(frame: VisionFrame): VisionFrame {
  if (frame.channels !== 3) {
    throw new Error('hsvToRGB: expected 3-channel HSV frame');
  }

  const { width, height } = frame;
  const src = frame.data;
  const out = new VisionFrame(width, height, 3);
  const dst = out.data;
  const n = width * height;

  for (let i = 0; i < n; i++) {
    const p = i * 3;
    const h = src[p] * (360 / 179);
    const s = src[p + 1] / 255;
    const v = src[p + 2] / 255;

    const c = v * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = v - c;

    let r = 0,
      g = 0,
      b = 0;
    if (h < 60) {
      r = c;
      g = x;
    } else if (h < 120) {
      r = x;
      g = c;
    } else if (h < 180) {
      g = c;
      b = x;
    } else if (h < 240) {
      g = x;
      b = c;
    } else if (h < 300) {
      r = x;
      b = c;
    } else {
      r = c;
      b = x;
    }

    dst[p] = ((r + m) * 255 + 0.5) | 0;
    dst[p + 1] = ((g + m) * 255 + 0.5) | 0;
    dst[p + 2] = ((b + m) * 255 + 0.5) | 0;
  }

  return out;
}

export interface HSVRange {
  /** H range [min, max], values in [0, 179]. Wraps around for reds (e.g. [170,10]). */
  h?: [number, number];
  /** S range [min, max], values in [0, 255]. */
  s?: [number, number];
  /** V range [min, max], values in [0, 255]. */
  v?: [number, number];
}

/**
 * Create a binary mask from an HSV frame where pixels fall within the given range.
 * Returns a 1-channel frame (255 = in range, 0 = out of range).
 *
 * Supports hue wrapping for red detection (e.g. h=[170,10] wraps around 0).
 *
 * @example
 * // Detect red regions
 * const mask = inRangeHSV(rgbToHSV(frame), { h: [170, 10], s: [100, 255], v: [50, 255] });
 */
export function inRangeHSV(frame: VisionFrame, range: HSVRange): VisionFrame {
  if (frame.channels !== 3) {
    throw new Error('inRangeHSV: expected 3-channel HSV frame (output of rgbToHSV)');
  }

  const { width, height } = frame;
  const src = frame.data;
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;
  const n = width * height;

  const [hMin, hMax] = range.h ?? [0, 179];
  const [sMin, sMax] = range.s ?? [0, 255];
  const [vMin, vMax] = range.v ?? [0, 255];
  const hWraps = hMin > hMax; // e.g. [170, 10] wraps through 0

  for (let i = 0; i < n; i++) {
    const p = i * 3;
    const h = src[p],
      s = src[p + 1],
      v = src[p + 2];

    const hOk = hWraps ? h >= hMin || h <= hMax : h >= hMin && h <= hMax;
    const sOk = s >= sMin && s <= sMax;
    const vOk = v >= vMin && v <= vMax;

    dst[i] = hOk && sOk && vOk ? 255 : 0;
  }

  return out;
}

export interface RGBRange {
  r?: [number, number];
  g?: [number, number];
  b?: [number, number];
}

function inside(v: number, range?: [number, number]): boolean {
  if (!range) {
    return true;
  }
  return v >= range[0] && v <= range[1];
}

export function inRangeGray(frame: VisionFrame, min: number, max: number): VisionFrame {
  if (frame.channels !== 1) {
    throw new Error('inRangeGray: expected 1-channel frame');
  }

  const out = new VisionFrame(frame.width, frame.height, 1);
  const src = frame.data;
  const dst = out.data;

  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    dst[i] = v >= min && v <= max ? 255 : 0;
  }
  return out;
}

export function inRangeRGB(frame: VisionFrame, range: RGBRange): VisionFrame {
  if (frame.channels !== 3 && frame.channels !== 4) {
    throw new Error('inRangeRGB: expected 3 or 4-channel RGB/RGBA frame');
  }

  const { width, height, channels } = frame;
  const out = new VisionFrame(width, height, 1);
  const src = frame.data;
  const dst = out.data;

  for (let i = 0, p = 0; i < width * height; i++, p += channels) {
    dst[i] = inside(src[p], range.r) && inside(src[p + 1], range.g) && inside(src[p + 2], range.b) ? 255 : 0;
  }
  return out;
}
