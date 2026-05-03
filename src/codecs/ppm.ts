import { promises as fs } from 'fs';
import { VisionFrame } from '../core/VisionFrame.js';

function isWhitespace(byte: number): boolean {
  return byte === 10 || byte === 13 || byte === 32 || byte === 9;
}

function readToken(bytes: Uint8Array, cursor: { i: number }): string {
  while (cursor.i < bytes.length) {
    const b = bytes[cursor.i];

    if (b === 35) {
      while (cursor.i < bytes.length && bytes[cursor.i] !== 10) {
        cursor.i++;
      }
    } else if (isWhitespace(b)) {
      cursor.i++;
    } else {
      break;
    }
  }

  const start = cursor.i;

  while (cursor.i < bytes.length && !isWhitespace(bytes[cursor.i])) {
    cursor.i++;
  }

  return Buffer.from(bytes.subarray(start, cursor.i)).toString('ascii');
}

export async function readPPM(path: string): Promise<VisionFrame> {
  const file = await fs.readFile(path);
  const bytes = new Uint8Array(file.buffer, file.byteOffset, file.byteLength);
  const cursor = { i: 0 };

  const magic = readToken(bytes, cursor);
  if (magic !== 'P6') {
    throw new Error('Only binary PPM P6 is supported');
  }

  const width = Number(readToken(bytes, cursor));
  const height = Number(readToken(bytes, cursor));
  const max = Number(readToken(bytes, cursor));

  if (!Number.isInteger(width) || !Number.isInteger(height) || max !== 255) {
    throw new Error('Invalid PPM header');
  }

  while (cursor.i < bytes.length && isWhitespace(bytes[cursor.i])) {
    cursor.i++;
  }

  const expected = width * height * 3;
  const pixelBytes = bytes.subarray(cursor.i, cursor.i + expected);

  if (pixelBytes.length !== expected) {
    throw new Error('PPM pixel data is incomplete');
  }

  const frame = new VisionFrame(width, height, 3);
  frame.data.set(pixelBytes);

  return frame;
}

export async function writePPM(path: string, frame: VisionFrame): Promise<void> {
  if (frame.channels !== 3) {
    throw new Error('PPM writer requires RGB frame with 3 channels');
  }

  const header = Buffer.from(`P6\n${frame.width} ${frame.height}\n255\n`, 'ascii');
  const pixels = Buffer.from(frame.data.buffer, frame.data.byteOffset, frame.data.byteLength);

  await fs.writeFile(path, Buffer.concat([header, pixels]));
}
