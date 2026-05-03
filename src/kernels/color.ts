import { clampU8, VisionFrame } from '../core/VisionFrame.js';

/** Adjust brightness (-255 to +255) and contrast (0.0 = flat, 1.0 = no change, >1 = more). */
export function brightnessContrast(frame: VisionFrame, brightness = 0, contrast = 1): VisionFrame {
  const out = new VisionFrame(frame.width, frame.height, frame.channels);
  const src = frame.data;
  const dst = out.data;

  // Precompute LUT for speed
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

/** Extract a single channel (0=R, 1=G, 2=B, 3=A) as a grayscale 1-channel frame. */
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
 * For a grayscale frame, returns [hist].
 * For RGB, returns [histR, histG, histB].
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
 * Histogram equalisation (works on grayscale 1-channel frames).
 * Stretches contrast to use the full 0–255 range.
 */
export function equalizeHistogram(frame: VisionFrame): VisionFrame {
  if (frame.channels !== 1) {
    throw new Error('equalizeHistogram requires a 1-channel grayscale frame');
  }

  const src = frame.data;
  const n = src.length;

  // Build CDF
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
