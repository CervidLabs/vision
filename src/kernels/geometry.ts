import { clampU8, VisionFrame } from '../core/VisionFrame.js';

export type ResizeMethod = 'nearest' | 'bilinear' | 'area';

export interface ResizeOptions {
  width?: number;
  height?: number;
  fit?: 'exact' | 'contain' | 'cover';
  method?: ResizeMethod;
}

/** Crop a rectangular region from a frame. */
export function crop(frame: VisionFrame, x: number, y: number, width: number, height: number): VisionFrame {
  if (x < 0 || y < 0 || x + width > frame.width || y + height > frame.height) {
    throw new Error(`crop(${x},${y},${width},${height}) out of bounds for ${frame.width}×${frame.height} frame`);
  }

  const out = new VisionFrame(width, height, frame.channels);
  const src = frame.data;
  const dst = out.data;
  const ch = frame.channels;

  for (let row = 0; row < height; row++) {
    const srcOff = ((y + row) * frame.width + x) * ch;
    const dstOff = row * width * ch;
    dst.set(src.subarray(srcOff, srcOff + width * ch), dstOff);
  }

  return out;
}

export function resizeNearest(frame: VisionFrame, newW: number, newH: number): VisionFrame {
  const out = new VisionFrame(newW, newH, frame.channels);
  const src = frame.data;
  const dst = out.data;

  const { width, height, channels } = frame;
  const xMap = new Int32Array(newW);
  const yMap = new Int32Array(newH);

  for (let x = 0; x < newW; x++) {
    xMap[x] = Math.min(width - 1, Math.floor((x * width) / newW));
  }

  for (let y = 0; y < newH; y++) {
    yMap[y] = Math.min(height - 1, Math.floor((y * height) / newH));
  }

  for (let y = 0; y < newH; y++) {
    const srcRow = yMap[y] * width * channels;
    const dstRow = y * newW * channels;

    for (let x = 0; x < newW; x++) {
      const srcOff = srcRow + xMap[x] * channels;
      const dstOff = dstRow + x * channels;

      for (let c = 0; c < channels; c++) {
        dst[dstOff + c] = src[srcOff + c];
      }
    }
  }

  return out;
}

export function resizeBilinear(frame: VisionFrame, newW: number, newH: number): VisionFrame {
  const out = new VisionFrame(newW, newH, frame.channels);

  const src = frame.data;
  const dst = out.data;

  const { width, height, channels } = frame;

  const x0Map = new Int32Array(newW);
  const x1Map = new Int32Array(newW);
  const wxMap = new Float64Array(newW);

  const y0Map = new Int32Array(newH);
  const y1Map = new Int32Array(newH);
  const wyMap = new Float64Array(newH);

  for (let x = 0; x < newW; x++) {
    const fx = ((x + 0.5) * width) / newW - 0.5;
    const x0 = Math.max(0, Math.floor(fx));
    const x1 = Math.min(width - 1, x0 + 1);

    x0Map[x] = x0;
    x1Map[x] = x1;
    wxMap[x] = fx - x0;
  }

  for (let y = 0; y < newH; y++) {
    const fy = ((y + 0.5) * height) / newH - 0.5;
    const y0 = Math.max(0, Math.floor(fy));
    const y1 = Math.min(height - 1, y0 + 1);

    y0Map[y] = y0;
    y1Map[y] = y1;
    wyMap[y] = fy - y0;
  }

  for (let y = 0; y < newH; y++) {
    const y0 = y0Map[y];
    const y1 = y1Map[y];
    const wy = wyMap[y];

    const row0 = y0 * width * channels;
    const row1 = y1 * width * channels;
    const dstRow = y * newW * channels;

    for (let x = 0; x < newW; x++) {
      const x0 = x0Map[x];
      const x1 = x1Map[x];
      const wx = wxMap[x];

      const w00 = (1 - wx) * (1 - wy);
      const w10 = wx * (1 - wy);
      const w01 = (1 - wx) * wy;
      const w11 = wx * wy;

      const p00 = row0 + x0 * channels;
      const p10 = row0 + x1 * channels;
      const p01 = row1 + x0 * channels;
      const p11 = row1 + x1 * channels;
      const d = dstRow + x * channels;

      for (let c = 0; c < channels; c++) {
        dst[d + c] = clampU8(src[p00 + c] * w00 + src[p10 + c] * w10 + src[p01 + c] * w01 + src[p11 + c] * w11);
      }
    }
  }

  return out;
}

export function resizeArea(frame: VisionFrame, newW: number, newH: number): VisionFrame {
  const { width, height, channels } = frame;

  if (newW <= 0 || newH <= 0) {
    throw new Error('resizeArea: dimensions must be positive');
  }

  if (newW >= width || newH >= height) {
    return resizeBilinear(frame, newW, newH);
  }

  const src = frame.data;
  const out = new VisionFrame(newW, newH, channels);
  const dst = out.data;

  const scaleX = width / newW;
  const scaleY = height / newH;
  const area = scaleX * scaleY;

  const acc = new Float64Array(channels);

  for (let y = 0; y < newH; y++) {
    const srcY0 = y * scaleY;
    const srcY1 = srcY0 + scaleY;

    const yStart = Math.floor(srcY0);
    const yEnd = Math.min(height - 1, Math.ceil(srcY1) - 1);

    for (let x = 0; x < newW; x++) {
      const srcX0 = x * scaleX;
      const srcX1 = srcX0 + scaleX;

      const xStart = Math.floor(srcX0);
      const xEnd = Math.min(width - 1, Math.ceil(srcX1) - 1);

      acc.fill(0);

      for (let sy = yStart; sy <= yEnd; sy++) {
        const yOverlap = Math.min(srcY1, sy + 1) - Math.max(srcY0, sy);
        if (yOverlap <= 0) {
          continue;
        }

        const srcRow = sy * width * channels;

        for (let sx = xStart; sx <= xEnd; sx++) {
          const xOverlap = Math.min(srcX1, sx + 1) - Math.max(srcX0, sx);
          if (xOverlap <= 0) {
            continue;
          }

          const weight = xOverlap * yOverlap;
          const srcOff = srcRow + sx * channels;

          for (let c = 0; c < channels; c++) {
            acc[c] += src[srcOff + c] * weight;
          }
        }
      }

      const dstOff = (y * newW + x) * channels;

      for (let c = 0; c < channels; c++) {
        const v = (acc[c] / area + 0.5) | 0;
        dst[dstOff + c] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }

  return out;
}

export function resolveResizeSize(frame: VisionFrame, opts: ResizeOptions): { width: number; height: number; method: ResizeMethod } {
  const method = opts.method ?? 'bilinear';
  const fit = opts.fit ?? 'exact';

  if (!opts.width && !opts.height) {
    throw new Error('resize requires width, height, or both');
  }

  if (fit === 'exact') {
    if (opts.width && opts.height) {
      return { width: opts.width, height: opts.height, method };
    }

    if (opts.width) {
      return {
        width: opts.width,
        height: Math.max(1, Math.round(frame.height * (opts.width / frame.width))),
        method,
      };
    }

    return {
      width: Math.max(1, Math.round(frame.width * (opts.height! / frame.height))),
      height: opts.height!,
      method,
    };
  }

  if (!opts.width || !opts.height) {
    throw new Error(`${fit} resize requires both width and height`);
  }

  const scale =
    fit === 'contain'
      ? Math.min(opts.width / frame.width, opts.height / frame.height)
      : Math.max(opts.width / frame.width, opts.height / frame.height);

  return {
    width: Math.max(1, Math.round(frame.width * scale)),
    height: Math.max(1, Math.round(frame.height * scale)),
    method,
  };
}

export function resize(frame: VisionFrame, newW: number, newH: number, method: ResizeMethod = 'bilinear'): VisionFrame {
  if (newW <= 0 || newH <= 0) {
    throw new Error('resize: dimensions must be positive');
  }

  switch (method) {
    case 'nearest':
      return resizeNearest(frame, newW, newH);
    case 'bilinear':
      return resizeBilinear(frame, newW, newH);
    case 'area':
      return resizeArea(frame, newW, newH);
    default:
      throw new Error(`Unknown resize method: ${method as string}`);
  }
}

export function flipH(frame: VisionFrame): VisionFrame {
  const out = new VisionFrame(frame.width, frame.height, frame.channels);
  const src = frame.data;
  const dst = out.data;
  const ch = frame.channels;
  const { width, height } = frame;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const srcOff = (y * width + (width - 1 - x)) * ch;
      const dstOff = (y * width + x) * ch;

      for (let c = 0; c < ch; c++) {
        dst[dstOff + c] = src[srcOff + c];
      }
    }
  }

  return out;
}

export function flipV(frame: VisionFrame): VisionFrame {
  const out = new VisionFrame(frame.width, frame.height, frame.channels);
  const src = frame.data;
  const dst = out.data;
  const rowBytes = frame.width * frame.channels;

  for (let y = 0; y < frame.height; y++) {
    const srcOff = (frame.height - 1 - y) * rowBytes;
    dst.set(src.subarray(srcOff, srcOff + rowBytes), y * rowBytes);
  }

  return out;
}

/** Rotate 90° clockwise. Output dimensions are swapped (width↔height). */
export function rotate90(frame: VisionFrame): VisionFrame {
  const { width, height, channels } = frame;
  const out = new VisionFrame(height, width, channels);
  const src = frame.data;
  const dst = out.data;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const srcOff = (y * width + x) * channels;
      const dstOff = (x * height + (height - 1 - y)) * channels;

      for (let c = 0; c < channels; c++) {
        dst[dstOff + c] = src[srcOff + c];
      }
    }
  }

  return out;
}

/** Rotate 180°. Output dimensions are the same. */
export function rotate180(frame: VisionFrame): VisionFrame {
  const { width, height, channels } = frame;
  const out = new VisionFrame(width, height, channels);
  const src = frame.data;
  const dst = out.data;

  // Reverse pixel order (works for all channel counts)
  for (let i = 0; i < width * height; i++) {
    const srcOff = i * channels;
    const dstOff = (width * height - 1 - i) * channels;
    for (let c = 0; c < channels; c++) {
      dst[dstOff + c] = src[srcOff + c];
    }
  }

  return out;
}

/** Rotate 270° clockwise (= 90° counter-clockwise). Output dimensions are swapped (width↔height). */
export function rotate270(frame: VisionFrame): VisionFrame {
  const { width, height, channels } = frame;
  const out = new VisionFrame(height, width, channels);
  const src = frame.data;
  const dst = out.data;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const srcOff = (y * width + x) * channels;
      // 270° CW: dst(y, width-1-x) = src(x, y)  — output is (height×width)
      const dstOff = ((width - 1 - x) * height + y) * channels;
      for (let c = 0; c < channels; c++) {
        dst[dstOff + c] = src[srcOff + c];
      }
    }
  }

  return out;
}
