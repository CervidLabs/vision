import { extname } from 'node:path';
import type { VisionFrame } from '../core/VisionFrame.js';
import { readJPEG, writeJPEG, type JPEGReadOptions } from './jpg.js';
import { readPNG, writePNG } from './png.js';
import { readPPM, writePPM } from './ppm.js';

export interface WriteOptions {
  quality?: number;
  compressionLevel?: number;
  pngFilter?: "none" | "sub";
}
export interface ReadOptions {
  resize?: {
    width?: number;
    height?: number;
    method?: 'nearest' | 'bilinear' | 'area';
  };
}

export interface WriteOptions {
  quality?: number;
  compressionLevel?: number;
}

export async function readFrame(path: string, opts: ReadOptions = {}): Promise<VisionFrame> {
  const ext = extname(path).toLowerCase();

  switch (ext) {
    case '.jpg':
    case '.jpeg':
      return readJPEG(path);

    case '.png':
      return readPNG(path);

    case '.ppm':
      return readPPM(path);

    default:
      throw new Error(`Unsupported image format: ${ext}`);
  }
}

export async function writeFrame(
  path: string,
  frame: VisionFrame,
  opts: WriteOptions = {}
): Promise<void> {
  const ext = extname(path).toLowerCase();

  switch (ext) {
    case '.jpg':
    case '.jpeg':
      return writeJPEG(path, frame, opts.quality ?? 85);

    case '.png':
    case ".png":
      return writePNG(path, frame, {
        level: opts.compressionLevel ?? 1,
        filter: opts.pngFilter ?? "sub",
      });
    case '.ppm':
      return writePPM(path, frame);

    default:
      throw new Error(`Unsupported image format: ${ext}`);
  }
}