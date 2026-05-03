import type { VisionFrame } from '../core/VisionFrame.js';
import { readPPM, writePPM } from './ppm.js';
import { readPNG, writePNG } from './png.js';
import { readJPEG, writeJPEG } from './jpg.js';

export interface WriteOptions {
  /** JPEG quality 1–100 (default 85). Ignored for PNG and PPM. */
  quality?: number;
  /** Retained for API compatibility. */
  preserveAlpha?: boolean;
}

function ext(path: string): string {
  return path.split('.').pop()?.toLowerCase() ?? '';
}

export async function readFrame(path: string, _opts: WriteOptions = {}): Promise<VisionFrame> {
  switch (ext(path)) {
    case 'ppm':
      return readPPM(path);
    case 'png':
      return readPNG(path);
    case 'jpg':
    case 'jpeg':
      return readJPEG(path);
    default:
      throw new Error(`Unsupported format ".${ext(path)}". Supported (zero deps): ppm, png, jpg/jpeg.`);
  }
}

export async function writeFrame(path: string, frame: VisionFrame, opts: WriteOptions = {}): Promise<void> {
  switch (ext(path)) {
    case 'ppm':
      if (frame.channels !== 3) {
        throw new Error('PPM requires a 3-channel RGB frame');
      }
      return writePPM(path, frame);
    case 'png':
      return writePNG(path, frame);
    case 'jpg':
    case 'jpeg':
      return writeJPEG(path, frame, opts.quality ?? 85);
    default:
      throw new Error(`Unsupported format ".${ext(path)}". Supported (zero deps): ppm, png, jpg/jpeg.`);
  }
}
