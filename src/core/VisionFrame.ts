export type PixelType = 'u8';

export class VisionFrame {
  readonly width: number;
  readonly height: number;
  readonly channels: number;
  readonly buffer: SharedArrayBuffer;
  readonly data: Uint8Array;
  readonly type: PixelType;

  constructor(width: number, height: number, channels = 3, buffer?: SharedArrayBuffer) {
    if (width <= 0 || height <= 0) {
      throw new Error('Invalid frame dimensions');
    }

    if (channels !== 1 && channels !== 3 && channels !== 4) {
      throw new Error('Only 1, 3, or 4 channels are supported');
    }

    this.width = width;
    this.height = height;
    this.channels = channels;
    this.type = 'u8';

    const size = width * height * channels;
    this.buffer = buffer ?? new SharedArrayBuffer(size);
    this.data = new Uint8Array(this.buffer);

    if (this.data.length !== size) {
      throw new Error('Buffer size does not match frame dimensions');
    }
  }

  get length(): number {
    return this.data.length;
  }

  clone(): VisionFrame {
    const out = new VisionFrame(this.width, this.height, this.channels);
    out.data.set(this.data);
    return out;
  }

  index(x: number, y: number, c = 0): number {
    return (y * this.width + x) * this.channels + c;
  }

  get(x: number, y: number, c = 0): number {
    return this.data[this.index(x, y, c)] ?? 0;
  }

  set(x: number, y: number, c: number, value: number): void {
    this.data[this.index(x, y, c)] = clampU8(value);
  }
}

export function clampU8(value: number): number {
  if (value < 0) {
    return 0;
  }
  if (value > 255) {
    return 255;
  }
  return value | 0;
}
