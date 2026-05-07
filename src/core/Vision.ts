import type { VisionFrame } from './VisionFrame.js';
import { readFrame, writeFrame, type WriteOptions, type ReadOptions } from '../codecs/index.js';
import * as Kernels from '../kernels/index.js';
import { VisionPipeline } from './VisionPipeline.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type KernelFn = (frame: VisionFrame, ...args: any[]) => any;

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
export class VisionImage {
  constructor(public frame: VisionFrame) {}

  // ── Pipeline ─────────────────────────────────────────────────────────────

  /**
   * Returns a VisionPipeline for deferred, optimized execution.
   *
   * IMPORTANT: methods called directly on VisionImage (e.g. `.grayscale()`)
   * execute immediately on the main thread and mutate this instance.
   * Use `.pipeline().grayscale()...run()` for fusion + worker offloading.
   */
  pipeline(): VisionPipeline<this> {
    return new VisionPipeline(this);
  }

  // ── Clone ────────────────────────────────────────────────────────────────

  /**
   * Return a deep copy of this VisionImage.
   * Useful when you need to branch the image without mutating the original.
   *
   * @example
   * const copy = img.clone();
   * const edges = copy.grayscale().edges();
   * // img is unchanged
   */
  clone(): VisionImage {
    return new VisionImage(this.frame.clone());
  }

  // ── Accessors ────────────────────────────────────────────────────────────

  get width(): number {
    return this.frame.width;
  }
  get height(): number {
    return this.frame.height;
  }
  get channels(): number {
    return this.frame.channels;
  }

  integral(): ReturnType<typeof Kernels.integralImage> {
    return Kernels.integralImage(this.frame);
  }

  // ── Color space ──────────────────────────────────────────────────────────

  toRGB(): VisionImage {
    if (this.frame.channels === 3) {
      return this;
    }
    return new VisionImage(Kernels.grayscaleToRGB(this.frame));
  }

  // ── Convolution / filters ─────────────────────────────────────────────────

  /** Gaussian blur. `radius` in pixels (kernel = 2r+1). Supports 1, 3, 4 channels. */
  blur(radius = 1, sigma?: number): VisionImage {
    this.frame = Kernels.gaussianBlur(this.frame, radius, sigma);
    return this;
  }

  /**
   * Sobel edge detection.
   * Input must be grayscale; chain `.grayscale().edges()`.
   */
  edges(): VisionImage {
    this.frame = Kernels.sobel(this.frame);
    return this;
  }

  async edgesParallel(workers?: number): Promise<VisionImage> {
    const { sobelParallel } = await import('../kernels/parallel.js');
    return new VisionImage(await sobelParallel(this.frame, workers));
  }

  // ── Geometry ──────────────────────────────────────────────────────────────

  resize(width: number, height: number, method?: Kernels.ResizeMethod): VisionImage;
  resize(options: Kernels.ResizeOptions): VisionImage;
  resize(arg1: number | Kernels.ResizeOptions, arg2?: number, arg3: Kernels.ResizeMethod = 'bilinear'): VisionImage {
    if (typeof arg1 === 'number') {
      if (typeof arg2 !== 'number') {
        throw new Error('resize(width, height) requires both width and height');
      }
      this.frame = Kernels.resize(this.frame, arg1, arg2, arg3);
      return this;
    }
    const size = Kernels.resolveResizeSize(this.frame, arg1);
    this.frame = Kernels.resize(this.frame, size.width, size.height, size.method);
    return this;
  }

  /** Scale by a factor, e.g. 0.5 for half size. */
  scale(factor: number, method: 'nearest' | 'bilinear' = 'bilinear'): VisionImage {
    this.frame = Kernels.resize(
      this.frame,
      Math.max(1, Math.round(this.frame.width * factor)),
      Math.max(1, Math.round(this.frame.height * factor)),
      method,
    );
    return this;
  }

  /** Extract one channel as a grayscale frame (0=R,1=G,2=B,3=A). */
  channel(c: 0 | 1 | 2 | 3): VisionImage {
    this.frame = Kernels.extractChannel(this.frame, c);
    return this;
  }

  // ── Save ──────────────────────────────────────────────────────────────────

  /**
   * Save to disk. Format is inferred from the file extension.
   * Supported: `.ppm`, `.png`, `.jpg`, `.jpeg`, `.bmp`
   */
  async save(path: string, opts: WriteOptions = {}): Promise<void> {
    const lower = path.toLowerCase();
    const needsRGB = lower.endsWith('.jpg') || lower.endsWith('.jpeg') || lower.endsWith('.ppm') || lower.endsWith('.bmp');
    const frame = needsRGB && this.frame.channels === 1 ? Kernels.grayscaleToRGB(this.frame) : this.frame;
    await writeFrame(path, frame, opts);
  }
}

function isVisionFrame(value: unknown): value is VisionFrame {
  return typeof value === 'object' && value !== null && 'width' in value && 'height' in value && 'channels' in value && 'data' in value;
}

// Methods manually implemented above that should NOT be auto-attached
const RESERVED_METHODS = new Set<string>([
  'constructor',
  'frame',
  'pipeline',
  'clone',
  'blur',
  'edges',
  'edgesParallel',
  'resize',
  'scale',
  'toRGB',
  'save',
  'width',
  'height',
  'channels',
  'integral',
  'channel',
  'boxBlur', // avoid conflict if exported from kernels
  'gaussianBlur', // mapped as 'blur'
  'sobel', // mapped as 'edges'
  'resolveResizeSize',
]);

function attachKernelMethods(): void {
  for (const [name, fn] of Object.entries(Kernels)) {
    if (RESERVED_METHODS.has(name)) {
      continue;
    }
    if (typeof fn !== 'function') {
      continue;
    }
    if (name.startsWith('resolve')) {
      continue;
    }
    if (name.startsWith('type')) {
      continue;
    }
    if (name in VisionImage.prototype) {
      continue;
    }

    Object.defineProperty(VisionImage.prototype, name, {
      configurable: true,
      writable: true,
      value: function dynamicKernelMethod(this: VisionImage, ...args: unknown[]): unknown {
        const result = (fn as KernelFn)(this.frame, ...args);
        if (isVisionFrame(result)) {
          this.frame = result;
          return this;
        }
        return result;
      },
    });
  }
}

attachKernelMethods();

export const Vision = {
  /** Read an image from disk. Supported: .ppm, .png, .jpg, .jpeg, .bmp */
  async read(path: string, opts: ReadOptions = {}): Promise<VisionImage> {
    return new VisionImage(await readFrame(path, opts));
  },

  /** Wrap an existing VisionFrame. */
  fromFrame(frame: VisionFrame): VisionImage {
    return new VisionImage(frame);
  },
};

// ── Interface declaration merging (for IDE autocomplete) ──────────────────────

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
export interface VisionImage {
  grayscale(): VisionImage;
  threshold(value?: number): VisionImage;
  invert(): VisionImage;

  sharpen(strength?: number): VisionImage;
  convolve(kernel: number[], kw: number, kh: number): VisionImage;
  boxBlur(radius?: number): VisionImage;

  crop(x: number, y: number, w: number, h: number): VisionImage;
  flipH(): VisionImage;
  flipV(): VisionImage;
  rotate90(): VisionImage;
  rotate180(): VisionImage;
  rotate270(): VisionImage;

  brightnessContrast(brightness?: number, contrast?: number): VisionImage;
  gamma(g: number): VisionImage;
  normalize(): VisionImage;
  equalizeHistogram(): VisionImage;
  rgbToHSV(): VisionImage;
  hsvToRGB(): VisionImage;

  erode(radius?: number): VisionImage;
  dilate(radius?: number): VisionImage;
  open(radius?: number): VisionImage;
  close(radius?: number): VisionImage;
  morphGradient(radius?: number): VisionImage;
  topHat(radius?: number): VisionImage;
  blackHat(radius?: number): VisionImage;

  adaptiveThreshold(options?: Kernels.AdaptiveThresholdOptions): VisionImage;

  canny(options?: Kernels.CannyOptions): VisionImage;

  warpAffine(matrix: Kernels.AffineMatrix, outWidth?: number, outHeight?: number, borderValue?: number): VisionImage;

  inRangeGray(min: number, max: number): VisionImage;
  inRangeRGB(range: Kernels.RGBRange): VisionImage;
  inRangeHSV(range: Kernels.HSVRange): VisionImage;

  drawPoint(x: number, y: number, color?: Kernels.DrawColor, thickness?: number): VisionImage;
  drawLine(x0: number, y0: number, x1: number, y1: number, color?: Kernels.DrawColor, thickness?: number): VisionImage;
  drawRect(x: number, y: number, width: number, height: number, color?: Kernels.DrawColor, thickness?: number): VisionImage;
  drawFilledRect(x: number, y: number, width: number, height: number, color?: Kernels.DrawColor): VisionImage;
  drawCircle(cx: number, cy: number, radius: number, color?: Kernels.DrawColor, thickness?: number): VisionImage;
  drawFilledCircle(cx: number, cy: number, radius: number, color?: Kernels.DrawColor): VisionImage;
  drawBoxes(
    boxes: Array<Pick<Kernels.ConnectedComponent, 'x' | 'y' | 'width' | 'height'>>,
    color?: Kernels.DrawColor,
    thickness?: number,
  ): VisionImage;

  connectedComponents(options?: Kernels.ConnectedComponentsOptions): Kernels.ConnectedComponent[];
  histogram(): Uint32Array[];
}
