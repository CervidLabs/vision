import { VisionFrame } from '../core/VisionFrame.js';
import { gaussianBlur } from './convolve.js';

export interface CannyOptions {
  /** Lower hysteresis threshold. Default 50. */
  lowThreshold?: number;
  /** Upper hysteresis threshold. Default 150. */
  highThreshold?: number;
  /** Gaussian blur radius applied before detection. Default 1. */
  radius?: number;
  /** Gaussian sigma. Defaults to radius * 0.6 + 0.4. */
  sigma?: number;
}

// ── Sobel gradients → magnitude + quantized direction ────────────────────────

function sobelGradients(src: Uint8Array, width: number, height: number): { mag: Float32Array; dir: Uint8Array } {
  const mag = new Float32Array(width * height);
  const dir = new Uint8Array(width * height); // 0=0°, 1=45°, 2=90°, 3=135°

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;

      const tl = src[i - width - 1],
        tc = src[i - width],
        tr = src[i - width + 1];
      const ml = src[i - 1],
        mr = src[i + 1];
      const bl = src[i + width - 1],
        bc = src[i + width],
        br = src[i + width + 1];

      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;

      mag[i] = Math.sqrt(gx * gx + gy * gy);

      // Quantize angle to 4 directions (using atan2 of absolute values)
      const angle = Math.atan2(Math.abs(gy), Math.abs(gx)) * (180 / Math.PI);
      if (angle < 22.5) {
        dir[i] = 0;
      } // 0°   → compare E/W
      else if (angle < 67.5) {
        dir[i] = 1;
      } // 45°  → compare NE/SW
      else if (angle < 112.5) {
        dir[i] = 2;
      } // 90°  → compare N/S
      else {
        dir[i] = 3;
      } // 135° → compare NW/SE
    }
  }

  return { mag, dir };
}

// ── Non-maximum suppression ───────────────────────────────────────────────────

function nonMaxSuppression(mag: Float32Array, dir: Uint8Array, width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const m = mag[i];
      let q: number, r: number;

      switch (dir[i]) {
        case 0:
          q = mag[i + 1];
          r = mag[i - 1];
          break; // E/W
        case 1:
          q = mag[i - width + 1];
          r = mag[i + width - 1];
          break; // NE/SW
        case 2:
          q = mag[i - width];
          r = mag[i + width];
          break; // N/S
        default:
          q = mag[i - width - 1];
          r = mag[i + width + 1];
          break; // NW/SE
      }

      out[i] = m >= q && m >= r ? m : 0;
    }
  }

  return out;
}

// ── Double threshold + hysteresis ─────────────────────────────────────────────

function hysteresis(nms: Float32Array, width: number, height: number, low: number, high: number): Uint8Array {
  const out = new Uint8Array(width * height);
  const STRONG = 255;
  const WEAK = 128;

  for (let i = 0; i < nms.length; i++) {
    if (nms[i] >= high) {
      out[i] = STRONG;
    } else if (nms[i] >= low) {
      out[i] = WEAK;
    }
  }

  // BFS from strong pixels to connect weak edges
  const stack: number[] = [];
  for (let i = 0; i < out.length; i++) {
    if (out[i] === STRONG) {
      stack.push(i);
    }
  }

  while (stack.length > 0) {
    const i = stack.pop()!;
    const y = (i / width) | 0;
    const x = i - y * width;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) {
          continue;
        }
        const ny = y + dy,
          nx = x + dx;
        if (ny < 0 || nx < 0 || ny >= height || nx >= width) {
          continue;
        }
        const ni = ny * width + nx;
        if (out[ni] === WEAK) {
          out[ni] = STRONG;
          stack.push(ni);
        }
      }
    }
  }

  // Kill remaining disconnected weak edges
  for (let i = 0; i < out.length; i++) {
    if (out[i] === WEAK) {
      out[i] = 0;
    }
  }

  return out;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Canny edge detection.
 * Input must be grayscale (1-channel).
 *
 * Pipeline: Gaussian blur → Sobel gradients → NMS → double threshold + hysteresis.
 */
export function canny(frame: VisionFrame, options: CannyOptions = {}): VisionFrame {
  if (frame.channels !== 1) {
    throw new Error('canny: requires grayscale (1-channel) input. Chain .grayscale().canny()');
  }

  const { lowThreshold = 50, highThreshold = 150, radius = 1, sigma } = options;

  const blurred = gaussianBlur(frame, radius, sigma);
  const { mag, dir } = sobelGradients(blurred.data, blurred.width, blurred.height);
  const nms = nonMaxSuppression(mag, dir, frame.width, frame.height);
  const edges = hysteresis(nms, frame.width, frame.height, lowThreshold, highThreshold);

  const out = new VisionFrame(frame.width, frame.height, 1);
  out.data.set(edges);
  return out;
}
