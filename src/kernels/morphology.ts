import { VisionFrame } from '../core/VisionFrame.js';

function assertGray(frame: VisionFrame, name: string): void {
  if (frame.channels !== 1) {
    throw new Error(`${name}: expected 1-channel grayscale/binary frame`);
  }
}

function isActive(v: number): boolean {
  return v > 0;
}

export function erode(frame: VisionFrame, radius = 1): VisionFrame {
  assertGray(frame, 'erode');

  if (radius < 1) {
    return frame.clone();
  }

  const { width, height } = frame;
  const src = frame.data;
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;

  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height - 1, y + radius);

    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width - 1, x + radius);

      let ok = true;

      for (let yy = y0; yy <= y1 && ok; yy++) {
        const row = yy * width;

        for (let xx = x0; xx <= x1; xx++) {
          if (!isActive(src[row + xx])) {
            ok = false;
            break;
          }
        }
      }

      dst[y * width + x] = ok ? 255 : 0;
    }
  }

  return out;
}

export function dilate(frame: VisionFrame, radius = 1): VisionFrame {
  assertGray(frame, 'dilate');

  if (radius < 1) {
    return frame.clone();
  }

  const { width, height } = frame;
  const src = frame.data;
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;

  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height - 1, y + radius);

    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width - 1, x + radius);

      let ok = false;

      for (let yy = y0; yy <= y1 && !ok; yy++) {
        const row = yy * width;

        for (let xx = x0; xx <= x1; xx++) {
          if (isActive(src[row + xx])) {
            ok = true;
            break;
          }
        }
      }

      dst[y * width + x] = ok ? 255 : 0;
    }
  }

  return out;
}

export function open(frame: VisionFrame, radius = 1): VisionFrame {
  return dilate(erode(frame, radius), radius);
}

export function close(frame: VisionFrame, radius = 1): VisionFrame {
  return erode(dilate(frame, radius), radius);
}
