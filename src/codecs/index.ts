import { extname } from 'node:path';
import type { VisionFrame } from '../core/VisionFrame.js';
import { readJPEG, writeJPEG } from './jpg.js';
import { readPNG, writePNG } from './png.js';
import { readPPM, writePPM } from './ppm.js';
import { readBMP, writeBMP } from './bmp.js';

export interface WriteOptions {
  quality?: number;
  compressionLevel?: number;
  pngFilter?: 'none' | 'sub';
  /**
   * JPEG only.
   * true/default = progressive SOF2.
   * false = baseline SOF0.
   */
  progressive?: boolean;
}

export interface ReadOptions {
  resize?: {
    width?: number;
    height?: number;
    method?: 'nearest' | 'bilinear' | 'area';
    shrinkOnLoad?: boolean;
  };
}

export async function readFrame(path: string, opts: ReadOptions = {}): Promise<VisionFrame> {
  const ext = extname(path).toLowerCase();

  switch (ext) {
    case '.jpg':
    case '.jpeg':
      return readJPEG(path, opts);

    case '.png':
      return readPNG(path);

    case '.ppm':
      return readPPM(path);

    case '.bmp':
      return readBMP(path);

    default:
      throw new Error(`Unsupported image format: ${ext}`);
  }
}

export async function writeFrame(path: string, frame: VisionFrame, opts: WriteOptions = {}): Promise<void> {
  const ext = extname(path).toLowerCase();

  switch (ext) {
    case '.jpg':
    case '.jpeg':
      return writeJPEG(path, frame, opts.quality ?? 85, {
        progressive: opts.progressive ?? true,
      });

    case '.png':
      return writePNG(path, frame, {
        level: opts.compressionLevel ?? 1,
        filter: opts.pngFilter ?? 'none',
      });

    case '.ppm':
      return writePPM(path, frame);

    case '.bmp':
      return writeBMP(path, frame);

    default:
      throw new Error(`Unsupported image format: ${ext}`);
  }
}
