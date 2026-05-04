import { clampU8, VisionFrame } from '../core/VisionFrame.js';

/**
 * Apply a separable or non-separable convolution kernel to a frame.
 * Works on 1 or 3-channel frames. Alpha (ch=4) is left untouched.
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

  if (frame.channels !== 1 && frame.channels !== 3) {
    throw new Error('convolve supports 1 or 3-channel frames only');
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
    const row = y * width;
    const prev = row - width;
    const next = row + width;

    for (let x = 0; x < width; x++) {
      const i = row + x;
      dst[i] = (tmp[prev + x] * side + tmp[i] * center + tmp[next + x] * side) | 0;
    }
  }

  const lastRow = (height - 1) * width;
  const prevRow = lastRow - width;

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

  const lastX = width - 1;
  const lastY = height - 1;

  // Horizontal pass
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

  // Vertical pass
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
function gaussianBlurRGB(frame: VisionFrame, radius = 1, sigma?: number): VisionFrame {
  const { width, height, channels } = frame;
  const src = frame.data;

  const s = sigma ?? radius * 0.6 + 0.4;
  const kernel = gaussianKernel1D(radius, s);

  const temp = new VisionFrame(width, height, channels);
  const out = new VisionFrame(width, height, channels);

  const tmp = temp.data;
  const dst = out.data;

  const lastX = width - 1;
  const lastY = height - 1;

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
export function gaussianBlurSeparable(frame: VisionFrame, radius = 1, sigma?: number): VisionFrame {
  if (radius <= 0) {
    return frame.clone();
  }

  if (frame.channels === 1) {
    return gaussianBlurGray(frame, radius, sigma);
  }

  if (frame.channels !== 3) {
    throw new Error('gaussianBlurSeparable supports 1 or 3-channel frames only');
  }

  return gaussianBlurRGB(frame, radius, sigma);
}
/**
 * Gaussian blur.
 * @param radius  Blur radius in pixels (kernel size = 2*radius+1). Default 1.
 * @param sigma   Standard deviation. Defaults to `radius * 0.6`.
 */
export function gaussianBlur(frame: VisionFrame, radius = 1, sigma?: number): VisionFrame {
  return gaussianBlurSeparable(frame, radius, sigma);
}
/**
 * Sobel edge detection.
 * Returns a 1-channel frame where bright pixels = edges.
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

      const tl = src[i - width - 1];
      const tc = src[i - width];
      const tr = src[i - width + 1];
      const ml = src[i - 1];
      const mr = src[i + 1];
      const bl = src[i + width - 1];
      const bc = src[i + width];
      const br = src[i + width + 1];

      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      const mag = Math.abs(gx) + Math.abs(gy);

      dst[i] = mag > 255 ? 255 : mag;
    }
  }

  return out;
}
/** Unsharp mask sharpening kernel (3×3). */
export function sharpen(frame: VisionFrame, strength = 1): VisionFrame {
  const s = strength;
  const center = 1 + 4 * s;
  const kernel = [0, -s, 0, -s, center, -s, 0, -s, 0];
  return convolve(frame, kernel, 3, 3);
}

/** Box blur (mean filter) of given radius. Faster than gaussian for large radii. */
export function boxBlur(frame: VisionFrame, radius = 1): VisionFrame {
  const size = radius * 2 + 1;
  const v = 1 / (size * size);
  const kernel = Array(size * size).fill(v);
  return convolve(frame, kernel, size, size);
}
