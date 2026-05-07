/**
 * BMP codec — zero dependencies.
 * Read:  24-bit (BGR) and 32-bit (BGRA) uncompressed BMP.
 * Write: 24-bit (BGR) uncompressed BMP. Bottom-up row order.
 */

import { promises as fs } from 'node:fs';
import { VisionFrame } from '../core/VisionFrame.js';

// ── Read helpers ──────────────────────────────────────────────────────────────

function r16le(buf: Uint8Array, off: number): number {
  return buf[off] | (buf[off + 1] << 8);
}

function r32le(buf: Uint8Array, off: number): number {
  return (buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16) | (buf[off + 3] << 24)) >>> 0;
}

function r32leS(buf: Uint8Array, off: number): number {
  return buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16) | (buf[off + 3] << 24);
}

// ── Write helpers ─────────────────────────────────────────────────────────────

function w16le(buf: Uint8Array, off: number, v: number): void {
  buf[off] = v & 0xff;
  buf[off + 1] = (v >> 8) & 0xff;
}

function w32le(buf: Uint8Array, off: number, v: number): void {
  buf[off] = v & 0xff;
  buf[off + 1] = (v >> 8) & 0xff;
  buf[off + 2] = (v >> 16) & 0xff;
  buf[off + 3] = (v >> 24) & 0xff;
}

// ── Reader ────────────────────────────────────────────────────────────────────

export async function readBMP(path: string): Promise<VisionFrame> {
  const file = await fs.readFile(path);
  const buf = new Uint8Array(file.buffer, file.byteOffset, file.byteLength);

  // File header (14 bytes)
  if (buf[0] !== 0x42 || buf[1] !== 0x4d) {
    throw new Error('BMP: invalid signature');
  }

  const dataOffset = r32le(buf, 10);

  // DIB header
  const width = r32leS(buf, 18); // signed
  const height = r32leS(buf, 22); // negative = top-down
  const bitCount = r16le(buf, 28);
  const compression = r32le(buf, 30);

  if (width <= 0) {
    throw new Error('BMP: invalid width');
  }
  if (height === 0) {
    throw new Error('BMP: invalid height');
  }
  if (compression !== 0) {
    throw new Error(`BMP: compression type ${compression} not supported (only BI_RGB)`);
  }
  if (bitCount !== 24 && bitCount !== 32) {
    throw new Error(`BMP: ${bitCount}-bit not supported (only 24 and 32-bit)`);
  }

  const topDown = height < 0;
  const absH = Math.abs(height);
  const bytesPerPixel = bitCount >> 3;
  const rowStride = (width * bytesPerPixel + 3) & ~3; // 4-byte aligned

  const outChannels: 3 | 4 = bitCount === 32 ? 4 : 3;
  const frame = new VisionFrame(width, absH, outChannels);
  const dst = frame.data;

  for (let y = 0; y < absH; y++) {
    // BMP rows are stored bottom-up unless topDown flag
    const srcRow = topDown ? y : absH - 1 - y;
    const srcOff = dataOffset + srcRow * rowStride;
    const dstOff = y * width * outChannels;

    for (let x = 0; x < width; x++) {
      const sp = srcOff + x * bytesPerPixel;
      const dp = dstOff + x * outChannels;
      // BMP stores BGR(A), output RGB(A)
      dst[dp] = buf[sp + 2]; // R
      dst[dp + 1] = buf[sp + 1]; // G
      dst[dp + 2] = buf[sp]; // B
      if (outChannels === 4) {
        dst[dp + 3] = buf[sp + 3];
      } // A
    }
  }

  return frame;
}

// ── Writer ────────────────────────────────────────────────────────────────────

/**
 * Write a VisionFrame to a 24-bit BMP file.
 * Input must be 3-channel RGB. For grayscale, convert with grayscaleToRGB first.
 */
export async function writeBMP(path: string, frame: VisionFrame): Promise<void> {
  if (frame.channels !== 3) {
    throw new Error('writeBMP: requires 3-channel RGB frame. Use grayscaleToRGB() first for grayscale.');
  }

  const { width, height } = frame;
  const src = frame.data;

  const bytesPerPixel = 3;
  const rowStride = (width * bytesPerPixel + 3) & ~3; // 4-byte aligned
  const pixelDataSize = rowStride * height;
  const fileSize = 14 + 40 + pixelDataSize;
  const dataOffset = 14 + 40;

  const out = new Uint8Array(fileSize);

  // ── File header (14 bytes) ──
  out[0] = 0x42;
  out[1] = 0x4d; // 'BM'
  w32le(out, 2, fileSize); // file size
  w32le(out, 6, 0); // reserved
  w32le(out, 10, dataOffset); // pixel data offset

  // ── DIB header — BITMAPINFOHEADER (40 bytes) ──
  w32le(out, 14, 40); // header size
  w32le(out, 18, width); // width
  w32le(out, 22, height); // height (positive = bottom-up)
  w16le(out, 26, 1); // planes
  w16le(out, 28, 24); // bit count
  w32le(out, 30, 0); // compression = BI_RGB
  w32le(out, 34, pixelDataSize); // image size
  w32le(out, 38, 2835); // X pixels per meter (~72 DPI)
  w32le(out, 42, 2835); // Y pixels per meter
  w32le(out, 46, 0); // colors in table
  w32le(out, 50, 0); // important colors

  // ── Pixel data (bottom-up) ──
  for (let y = 0; y < height; y++) {
    const srcRow = (height - 1 - y) * width * 3; // flip vertically
    const dstRow = dataOffset + y * rowStride;

    for (let x = 0; x < width; x++) {
      const sp = srcRow + x * 3;
      const dp = dstRow + x * 3;
      out[dp] = src[sp + 2]; // B
      out[dp + 1] = src[sp + 1]; // G
      out[dp + 2] = src[sp]; // R
    }
    // Padding bytes are already 0 from Uint8Array initialization
  }

  await fs.writeFile(path, Buffer.from(out.buffer, out.byteOffset, out.byteLength));
}
