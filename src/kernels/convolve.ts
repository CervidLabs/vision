import { clampU8, VisionFrame } from '../core/VisionFrame.js';

/**
 * Apply a separable or non-separable convolution kernel to a frame.
 * Works on 1, 3, or 4-channel frames. Alpha (ch=4) is included.
 *
 * @param kernel  Row-major flat array of length `kw * kh`
 * @param kw      Kernel width  (must be odd)
 * @param kh      Kernel height (must be odd)
 */
export function convolve(frame: VisionFrame, kernel: number[], kw: number, kh: number): VisionFrame {
  if (kw % 2 === 0 || kh % 2 === 0) {
    throw new Error('Kernel dimensions must be odd');
  }
  if (kernel.length !== kw * kh) {
    throw new Error('Kernel length must equal kw * kh');
  }

  const { width, height, channels } = frame;
  const out = new VisionFrame(width, height, channels);
  const src = frame.data;
  const dst = out.data;

  const rx = (kw - 1) >> 1;
  const ry = (kh - 1) >> 1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dstBase = (y * width + x) * channels;

      for (let c = 0; c < channels; c++) {
        let acc = 0;

        for (let ky = 0; ky < kh; ky++) {
          let sy = y + ky - ry;
          if (sy < 0) {
            sy = 0;
          } else if (sy >= height) {
            sy = height - 1;
          }

          const srcRow = sy * width * channels;

          for (let kx = 0; kx < kw; kx++) {
            let sx = x + kx - rx;
            if (sx < 0) {
              sx = 0;
            } else if (sx >= width) {
              sx = width - 1;
            }

            acc += src[srcRow + sx * channels + c] * kernel[ky * kw + kx];
          }
        }

        dst[dstBase + c] = clampU8(acc);
      }
    }
  }

  return out;
}

// ── Box blur — O(w×h) via integral image ─────────────────────────────────────
//
// Uses a per-channel running-sum approach (2-pass prefix sum) which is equivalent
// to integralImage but avoids a separate allocation for non-critical paths.

/**
 * Box blur (mean filter) of given radius. O(w×h) regardless of radius.
 * Replaces the previous convolve-based implementation which was O(w×h×k²).
 */
export function boxBlur(frame: VisionFrame, radius = 1): VisionFrame {
  if (radius < 1) {
    return frame.clone();
  }

  const { width, height, channels } = frame;
  const src = frame.data;
  const tmp = new VisionFrame(width, height, channels);
  const out = new VisionFrame(width, height, channels);
  const tmpData = tmp.data;
  const dstData = out.data;

  // ── Horizontal pass ──────────────────────────────────────────────────────
  for (let y = 0; y < height; y++) {
    const rowOff = y * width * channels;

    for (let c = 0; c < channels; c++) {
      // Prefix sum along row for this channel
      let sum = 0;
      const r = radius;

      // Build initial window sum: [0 .. r]
      for (let x = 0; x <= Math.min(r, width - 1); x++) {
        sum += src[rowOff + x * channels + c];
      }
      // Account for left border replication
      sum += src[rowOff + c] * Math.max(0, r - (width > 1 ? 0 : 0));

      for (let x = 0; x < width; x++) {
        // Incoming right edge: x+r (clamped)
        const addX = Math.min(x + r, width - 1);
        const remX = x - r - 1;

        if (x > 0) {
          sum += src[rowOff + addX * channels + c];
          sum -= src[rowOff + Math.max(0, remX) * channels + c];
          // Extra removal when left edge clamps
          if (remX < 0) {
            sum += src[rowOff + c];
          } // subtract the replicated left pixel
        }

        const winW = Math.min(x + r, width - 1) - Math.max(x - r, 0) + 1;
        tmpData[rowOff + x * channels + c] = (sum / winW + 0.5) | 0;
      }
    }
  }

  // ── Vertical pass ────────────────────────────────────────────────────────
  for (let x = 0; x < width; x++) {
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < height; y++) {
        let sum = 0;
        const y0 = Math.max(0, y - radius);
        const y1 = Math.min(height - 1, y + radius);
        const winH = y1 - y0 + 1;
        for (let sy = y0; sy <= y1; sy++) {
          sum += tmpData[(sy * width + x) * channels + c];
        }
        dstData[(y * width + x) * channels + c] = (sum / winH + 0.5) | 0;
      }
    }
  }

  return out;
}

// ── Gaussian blur — separable, supports 1/3/4 channels ───────────────────────

function gaussianKernel1D(radius: number, sigma: number): Float64Array {
  const size = radius * 2 + 1;
  const kernel = new Float64Array(size);
  let sum = 0;

  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = v;
    sum += v;
  }

  for (let i = 0; i < size; i++) {
    kernel[i] /= sum;
  }
  return kernel;
}

function gaussianBlurGrayR1(frame: VisionFrame, sigma?: number): VisionFrame {
  const { width, height } = frame;
  const src = frame.data;

  const s = sigma ?? 1.0;
  const a = Math.exp(-1 / (2 * s * s));
  const center = 1 / (1 + 2 * a);
  const side = a * center;

  const temp = new VisionFrame(width, height, 1);
  const out = new VisionFrame(width, height, 1);
  const tmp = temp.data;
  const dst = out.data;

  // Horizontal
  for (let y = 0; y < height; y++) {
    const row = y * width;
    tmp[row] = (src[row] * (center + side) + src[row + 1] * side) | 0;
    for (let x = 1; x < width - 1; x++) {
      const i = row + x;
      tmp[i] = (src[i - 1] * side + src[i] * center + src[i + 1] * side) | 0;
    }
    const last = row + width - 1;
    tmp[last] = (src[last - 1] * side + src[last] * (center + side)) | 0;
  }

  // Vertical
  for (let x = 0; x < width; x++) {
    dst[x] = (tmp[x] * (center + side) + tmp[width + x] * side) | 0;
  }
  for (let y = 1; y < height - 1; y++) {
    const row = y * width,
      prev = row - width,
      next = row + width;
    for (let x = 0; x < width; x++) {
      const i = row + x;
      dst[i] = (tmp[prev + x] * side + tmp[i] * center + tmp[next + x] * side) | 0;
    }
  }
  const lastRow = (height - 1) * width,
    prevRow = lastRow - width;
  for (let x = 0; x < width; x++) {
    dst[lastRow + x] = (tmp[prevRow + x] * side + tmp[lastRow + x] * (center + side)) | 0;
  }

  return out;
}

function gaussianBlurGray(frame: VisionFrame, radius = 1, sigma?: number): VisionFrame {
  if (radius === 1) {
    return gaussianBlurGrayR1(frame, sigma);
  }

  const { width, height } = frame;
  const src = frame.data;
  const s = sigma ?? radius * 0.6 + 0.4;
  const kernel = gaussianKernel1D(radius, s);
  const temp = new VisionFrame(width, height, 1);
  const out = new VisionFrame(width, height, 1);
  const tmp = temp.data;
  const dst = out.data;
  const lastX = width - 1,
    lastY = height - 1;

  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let sx = x + k;
        if (sx < 0) {
          sx = 0;
        } else if (sx > lastX) {
          sx = lastX;
        }
        acc += src[row + sx] * kernel[k + radius];
      }
      tmp[row + x] = acc < 0 ? 0 : acc > 255 ? 255 : acc | 0;
    }
  }

  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let sy = y + k;
        if (sy < 0) {
          sy = 0;
        } else if (sy > lastY) {
          sy = lastY;
        }
        acc += tmp[sy * width + x] * kernel[k + radius];
      }
      dst[row + x] = acc < 0 ? 0 : acc > 255 ? 255 : acc | 0;
    }
  }

  return out;
}

/**
 * Gaussian blur — multichannel (RGB, RGBA, or grayscale).
 * All channels including alpha are blurred.
 *
 * @param radius  Blur radius in pixels (kernel = 2r+1). Default 1.
 * @param sigma   Standard deviation. Defaults to `radius * 0.6 + 0.4`.
 */
function gaussianBlurMultiChannel(frame: VisionFrame, radius = 1, sigma?: number): VisionFrame {
  const { width, height, channels } = frame;
  const src = frame.data;
  const s = sigma ?? radius * 0.6 + 0.4;
  const kernel = gaussianKernel1D(radius, s);
  const temp = new VisionFrame(width, height, channels);
  const out = new VisionFrame(width, height, channels);
  const tmp = temp.data;
  const dst = out.data;
  const lastX = width - 1,
    lastY = height - 1;

  // Horizontal pass
  for (let y = 0; y < height; y++) {
    const rowOff = y * width * channels;
    for (let x = 0; x < width; x++) {
      const base = rowOff + x * channels;
      for (let c = 0; c < channels; c++) {
        let acc = 0;
        for (let k = -radius; k <= radius; k++) {
          let sx = x + k;
          if (sx < 0) {
            sx = 0;
          } else if (sx > lastX) {
            sx = lastX;
          }
          acc += src[rowOff + sx * channels + c] * kernel[k + radius];
        }
        tmp[base + c] = acc < 0 ? 0 : acc > 255 ? 255 : acc | 0;
      }
    }
  }

  // Vertical pass
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const base = (y * width + x) * channels;
      for (let c = 0; c < channels; c++) {
        let acc = 0;
        for (let k = -radius; k <= radius; k++) {
          let sy = y + k;
          if (sy < 0) {
            sy = 0;
          } else if (sy > lastY) {
            sy = lastY;
          }
          acc += tmp[(sy * width + x) * channels + c] * kernel[k + radius];
        }
        dst[base + c] = acc < 0 ? 0 : acc > 255 ? 255 : acc | 0;
      }
    }
  }

  return out;
}

export function gaussianBlurSeparable(frame: VisionFrame, radius = 1, sigma?: number): VisionFrame {
  if (radius <= 0) {
    return frame.clone();
  }
  if (frame.channels === 1) {
    return gaussianBlurGray(frame, radius, sigma);
  }
  return gaussianBlurMultiChannel(frame, radius, sigma);
}

/**
 * Gaussian blur. Supports 1, 3, and 4-channel frames.
 * @param radius  Blur radius in pixels. Default 1.
 * @param sigma   Standard deviation. Defaults to `radius * 0.6 + 0.4`.
 */
export function gaussianBlur(frame: VisionFrame, radius = 1, sigma?: number): VisionFrame {
  return gaussianBlurSeparable(frame, radius, sigma);
}

/**
 * Sobel edge detection. Returns a 1-channel frame where bright pixels = edges.
 * Input must be grayscale (1 channel).
 */
export function sobel(frame: VisionFrame): VisionFrame {
  if (frame.channels !== 1) {
    throw new Error('sobel requires a 1-channel grayscale frame');
  }

  const { width, height } = frame;
  const src = frame.data;
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;

  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const i = row + x;
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
      const mag = Math.abs(gx) + Math.abs(gy);
      dst[i] = mag > 255 ? 255 : mag;
    }
  }

  return out;
}

/** Unsharp mask sharpening (3×3). */
export function sharpen(frame: VisionFrame, strength = 1): VisionFrame {
  const s = strength;
  const center = 1 + 4 * s;
  const kernel = [0, -s, 0, -s, center, -s, 0, -s, 0];
  return convolve(frame, kernel, 3, 3);
}
