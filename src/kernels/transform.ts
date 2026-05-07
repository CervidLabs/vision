import { VisionFrame, clampU8 } from '../core/VisionFrame.js';

/**
 * A 2×3 affine matrix in row-major order:
 * [ a  b  c ]   → x' = a*x + b*y + c
 * [ d  e  f ]   → y' = d*x + e*y + f
 *
 * warpAffine uses an **inverse map**: for each destination pixel (x,y),
 * the matrix defines where to sample in the source image.
 */
export type AffineMatrix = [number, number, number, number, number, number];

/**
 * Apply an affine transformation to a frame.
 *
 * @param frame        Source frame.
 * @param matrix       2×3 inverse-map matrix [a,b,c,d,e,f].
 * @param outWidth     Output width.  Defaults to source width.
 * @param outHeight    Output height. Defaults to source height.
 * @param borderValue  Pixel value used for out-of-bounds samples. Default 0.
 */
export function warpAffine(frame: VisionFrame, matrix: AffineMatrix, outWidth?: number, outHeight?: number, borderValue = 0): VisionFrame {
  const { width, height, channels } = frame;
  const src = frame.data;
  const dw = outWidth ?? width;
  const dh = outHeight ?? height;
  const out = new VisionFrame(dw, dh, channels);
  const dst = out.data;

  const [a, b, c, d, e, f] = matrix;

  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const sx = a * x + b * y + c;
      const sy = d * x + e * y + f;

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = x0 + 1;
      const y1 = y0 + 1;
      const wx = sx - x0;
      const wy = sy - y0;

      const dstOff = (y * dw + x) * channels;

      for (let ch = 0; ch < channels; ch++) {
        const p00 = inBounds(x0, y0, width, height) ? src[(y0 * width + x0) * channels + ch] : borderValue;
        const p10 = inBounds(x1, y0, width, height) ? src[(y0 * width + x1) * channels + ch] : borderValue;
        const p01 = inBounds(x0, y1, width, height) ? src[(y1 * width + x0) * channels + ch] : borderValue;
        const p11 = inBounds(x1, y1, width, height) ? src[(y1 * width + x1) * channels + ch] : borderValue;

        dst[dstOff + ch] = clampU8(p00 * (1 - wx) * (1 - wy) + p10 * wx * (1 - wy) + p01 * (1 - wx) * wy + p11 * wx * wy);
      }
    }
  }

  return out;
}

function inBounds(x: number, y: number, w: number, h: number): boolean {
  return x >= 0 && y >= 0 && x < w && y < h;
}

// ── Matrix helpers ─────────────────────────────────────────────────────────────

/**
 * Build an inverse-map affine matrix for rotation around a center point.
 *
 * @param angleDeg  Rotation angle in degrees (counter-clockwise).
 * @param cx        Center X.
 * @param cy        Center Y.
 * @param scale     Uniform scale factor. Default 1.
 */
export function rotationMatrix(angleDeg: number, cx: number, cy: number, scale = 1): AffineMatrix {
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad) * scale;
  const sin = Math.sin(rad) * scale;

  // Inverse map (rotate destination back to source):
  // sx = cos*(x-cx) + sin*(y-cy) + cx
  // sy = -sin*(x-cx) + cos*(y-cy) + cy
  return [cos, sin, cx * (1 - cos) - cy * sin, -sin, cos, cy * (1 - cos) + cx * sin];
}

/**
 * Build an inverse-map affine matrix for translation.
 *
 * @param tx  X translation (positive = shift image right).
 * @param ty  Y translation (positive = shift image down).
 */
export function translationMatrix(tx: number, ty: number): AffineMatrix {
  return [1, 0, -tx, 0, 1, -ty];
}

/**
 * Build an inverse-map affine matrix for uniform scaling around a center.
 */
export function scaleMatrix(sx: number, sy: number, cx = 0, cy = 0): AffineMatrix {
  return [1 / sx, 0, cx * (1 - 1 / sx), 0, 1 / sy, cy * (1 - 1 / sy)];
}

/**
 * Compose two inverse-map affine matrices (apply A then B → returns B∘A).
 */
export function composeAffine(A: AffineMatrix, B: AffineMatrix): AffineMatrix {
  const [a0, b0, c0, d0, e0, f0] = A;
  const [a1, b1, c1, d1, e1, f1] = B;
  return [a1 * a0 + b1 * d0, a1 * b0 + b1 * e0, a1 * c0 + b1 * f0 + c1, d1 * a0 + e1 * d0, d1 * b0 + e1 * e0, d1 * c0 + e1 * f0 + f1];
}
