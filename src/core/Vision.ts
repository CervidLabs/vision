import type { VisionFrame } from './VisionFrame.js';
import { readFrame, writeFrame, type WriteOptions } from '../codecs/index.js';

import {
    type ResizeOptions,
    // basic
    grayscale,
    grayscaleToRGB,
    invert,
    threshold,
    // convolve
    gaussianBlur,
    boxBlur,
    sobel,
    sharpen,
    convolve,
    // geometry
    crop,
    resize,
    flipH,
    flipV,
    rotate90,
    resolveResizeSize,
    // color
    brightnessContrast,
    gamma,
    extractChannel,
    histogram,
    equalizeHistogram,
    normalize,
    ResizeMethod,
} from '../kernels/index.js';

export class VisionImage {
    constructor(readonly frame: VisionFrame) { }

    // ── Basic ────────────────────────────────────────────────────────────────

    grayscale(): VisionImage {
        return new VisionImage(grayscale(this.frame));
    }

    threshold(value = 128): VisionImage {
        return new VisionImage(threshold(this.frame, value));
    }

    invert(): VisionImage {
        return new VisionImage(invert(this.frame));
    }

    toRGB(): VisionImage {
        if (this.frame.channels === 3) {
            return this;
        }
        return new VisionImage(grayscaleToRGB(this.frame));
    }

    // ── Convolution ──────────────────────────────────────────────────────────

    /** Gaussian blur. `radius` in pixels (kernel = 2r+1). */
    blur(radius = 1, sigma?: number): VisionImage {
        return new VisionImage(gaussianBlur(this.frame, radius, sigma));
    }

    boxBlur(radius = 1): VisionImage {
        return new VisionImage(boxBlur(this.frame, radius));
    }

    /**
     * Sobel edge detection.
     * Input must be grayscale; chain `.grayscale().edges()`.
     */
    edges(): VisionImage {
        return new VisionImage(sobel(this.frame));
    }
    async edgesParallel(workers?: number): Promise<VisionImage> {
        const { sobelParallel } = await import('../kernels/parallel.js');
        return new VisionImage(await sobelParallel(this.frame, workers));
    }
    sharpen(strength = 1): VisionImage {
        return new VisionImage(sharpen(this.frame, strength));
    }

    /** Apply an arbitrary flat convolution kernel (kw × kh, must be odd). */
    convolve(kernel: number[], kw: number, kh: number): VisionImage {
        return new VisionImage(convolve(this.frame, kernel, kw, kh));
    }

    // ── Geometry ─────────────────────────────────────────────────────────────

    crop(x: number, y: number, w: number, h: number): VisionImage {
        return new VisionImage(crop(this.frame, x, y, w, h));
    }

    resize(
        width: number,
        height: number,
        method?: ResizeMethod,
    ): VisionImage;

    resize(options: ResizeOptions): VisionImage;

    resize(
        arg1: number | ResizeOptions,
        arg2?: number,
        arg3: "nearest" | "bilinear" = "bilinear"
    ): VisionImage {
        if (typeof arg1 === "number") {
            if (typeof arg2 !== "number") {
                throw new Error("resize(width, height) requires both width and height");
            }

            return new VisionImage(resize(this.frame, arg1, arg2, arg3));
        }

        const size = resolveResizeSize(this.frame, arg1);
        return new VisionImage(resize(this.frame, size.width, size.height, size.method));
    }
    /** Scale by a factor, e.g. 0.5 for half size. */
    scale(factor: number, method: 'nearest' | 'bilinear' = 'bilinear'): VisionImage {
        return this.resize(Math.max(1, Math.round(this.frame.width * factor)), Math.max(1, Math.round(this.frame.height * factor)), method);
    }

    flipH(): VisionImage {
        return new VisionImage(flipH(this.frame));
    }
    flipV(): VisionImage {
        return new VisionImage(flipV(this.frame));
    }

    /** Rotate 90° clockwise. Chain twice for 180°, three times for 270°. */
    rotate90(): VisionImage {
        return new VisionImage(rotate90(this.frame));
    }

    // ── Color / tone ─────────────────────────────────────────────────────────

    /**
     * Adjust brightness and contrast.
     * @param brightness  -255 to +255
     * @param contrast    0.0 (flat grey) … 1.0 (unchanged) … 2.0+ (high contrast)
     */
    brightnessContrast(brightness = 0, contrast = 1): VisionImage {
        return new VisionImage(brightnessContrast(this.frame, brightness, contrast));
    }

    /** Gamma correction. `g < 1` brightens, `g > 1` darkens. */
    gamma(g: number): VisionImage {
        return new VisionImage(gamma(this.frame, g));
    }

    /** Normalise pixel values to span the full 0–255 range. */
    normalize(): VisionImage {
        return new VisionImage(normalize(this.frame));
    }

    /** Histogram equalisation (grayscale frames only). */
    equalizeHistogram(): VisionImage {
        return new VisionImage(equalizeHistogram(this.frame));
    }

    /** Extract one channel as a grayscale frame (0=R,1=G,2=B,3=A). */
    channel(c: 0 | 1 | 2 | 3): VisionImage {
        return new VisionImage(extractChannel(this.frame, c));
    }

    // ── Analysis ─────────────────────────────────────────────────────────────

    /**
     * Compute per-channel histograms.
     * Returns `Uint32Array[256]` per channel.
     */
    histogram(): Uint32Array[] {
        return histogram(this.frame);
    }

    get width(): number {
        return this.frame.width;
    }
    get height(): number {
        return this.frame.height;
    }
    get channels(): number {
        return this.frame.channels;
    }

    // ── I/O ──────────────────────────────────────────────────────────────────

    /**
     * Save to disk. Format is inferred from the file extension.
     * Supported: `.ppm`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.avif`
     */
    async save(path: string, opts: WriteOptions = {}): Promise<void> {
        const frame = this.frame.channels === 1 ? grayscaleToRGB(this.frame) : this.frame;
        await writeFrame(path, frame, opts);
    }
}

export const Vision = {
    /**
     * Read an image from disk.
     * Supported formats: .ppm (no deps), .png, .jpg, .jpeg, .webp, .tiff, .avif (requires sharp).
     */
    async read(path: string, opts: WriteOptions = {}): Promise<VisionImage> {
        return new VisionImage(await readFrame(path, opts));
    },

    /** Wrap an existing VisionFrame. */
    fromFrame(frame: VisionFrame): VisionImage {
        return new VisionImage(frame);
    },
};
