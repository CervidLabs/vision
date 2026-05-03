import { VisionFrame } from '../core/VisionFrame.js';

export function grayscale(frame: VisionFrame): VisionFrame {
  if (frame.channels !== 3 && frame.channels !== 4) {
    throw new Error('grayscale requires RGB or RGBA input');
  }

  const out = new VisionFrame(frame.width, frame.height, 1);
  const src = frame.data;
  const dst = out.data;
  const ch = frame.channels;

  // BT.601 integer approximation: (77R + 150G + 29B) >> 8
  // Max: (77+150+29)*255 = 65535 >> 8 = 255 — no clamp needed.
  for (let i = 0, j = 0; i < src.length; i += ch, j++) {
    dst[j] = (77 * src[i] + 150 * src[i + 1] + 29 * src[i + 2]) >> 8;
  }

  return out;
}

export function threshold(frame: VisionFrame, value = 128): VisionFrame {
  if (frame.channels !== 1) {
    throw new Error('threshold requires grayscale input');
  }

  const out = new VisionFrame(frame.width, frame.height, 1);
  const src = frame.data;
  const dst = out.data;

  for (let i = 0; i < src.length; i++) {
    dst[i] = src[i] >= value ? 255 : 0;
  }

  return out;
}

export function grayscaleToRGB(frame: VisionFrame): VisionFrame {
  if (frame.channels !== 1) {
    throw new Error('grayscaleToRGB requires 1-channel input');
  }

  const out = new VisionFrame(frame.width, frame.height, 3);
  const src = frame.data;
  const dst = out.data;

  for (let i = 0, j = 0; i < src.length; i++, j += 3) {
    const v = src[i];
    dst[j] = v;
    dst[j + 1] = v;
    dst[j + 2] = v;
  }

  return out;
}

export function invert(frame: VisionFrame): VisionFrame {
  const out = new VisionFrame(frame.width, frame.height, frame.channels);
  const src = frame.data;
  const dst = out.data;

  for (let i = 0; i < src.length; i++) {
    dst[i] = 255 - src[i];
  }

  return out;
}
