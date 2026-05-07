import { VisionFrame } from './VisionFrame.js';
import * as Kernels from '../kernels/index.js';
import { getPipelineWorkerPool } from './PipelineWorkerPool.js';
import { cpus } from 'node:os';

// ── Fix: was hardcoded to Math.min(8, 7) = 7 ─────────────────────────────────
const NUM_PIPELINE_WORKERS = Math.max(1, Math.min(8, cpus().length - 1));
const FUSED_WORKER_THRESHOLD_PX = 2_000_000;

interface ImageOwner {
  frame: VisionFrame;
}

type PipelineOp =
  | { name: 'grayscale' }
  | { name: 'blur'; radius: number; sigma?: number }
  | { name: 'edges' }
  | { name: 'threshold'; value: number }
  | { name: 'invert' }
  | { name: 'erode'; radius: number }
  | { name: 'dilate'; radius: number }
  | { name: 'open'; radius: number }
  | { name: 'close'; radius: number }
  | { name: 'normalize' }
  | { name: 'brightnessContrast'; brightness: number; contrast: number }
  | { name: 'drawCircle'; cx: number; cy: number; radius: number; color?: Kernels.DrawColor; thickness?: number }
  | { name: 'drawRect'; x: number; y: number; width: number; height: number; color?: Kernels.DrawColor; thickness?: number };

// ── Fusion detection ──────────────────────────────────────────────────────────
//
// A fusion collapses N sequential ops into a single optimized kernel, avoiding
// intermediate frames. Fused ops run in parallel workers when the frame is large.
//
// Current fusions:
//   gbe  → grayscale + blur(r=1) + edges           (3-op)
//   gt   → grayscale + threshold                   (2-op)
//   be   → blur(r=1) + edges   [grayscale input]   (2-op)
//   bte  → blur(any r) + edges [grayscale input]   (2-op, less tight but fuseable)
//   gte  → grayscale + threshold + erode           (3-op, common in binarization)

type FusionResult =
  | { fusion: 'gbe'; consumed: 3 }
  | { fusion: 'gt'; consumed: 2; threshold: number }
  | { fusion: 'be'; consumed: 2 }
  | { fusion: 'bte'; consumed: 2; radius: number; sigma?: number }
  | { fusion: 'gte'; consumed: 3; threshold: number; erodeRadius: number };

function detectFusion(ops: PipelineOp[], start: number): FusionResult | null {
  const a = ops[start];
  const b = ops[start + 1];
  const c = ops[start + 2];

  if (!a) {
    return null;
  }

  // grayscale + blur(r=1, no sigma) + edges → gbe
  if (a.name === 'grayscale' && b?.name === 'blur' && b.radius === 1 && b.sigma === undefined && c?.name === 'edges') {
    return { fusion: 'gbe', consumed: 3 };
  }

  // grayscale + threshold + erode → gte (common binarization + cleanup pattern)
  if (a.name === 'grayscale' && b?.name === 'threshold' && c?.name === 'erode') {
    return { fusion: 'gte', consumed: 3, threshold: b.value, erodeRadius: c.radius };
  }

  // grayscale + threshold → gt
  if (a.name === 'grayscale' && b?.name === 'threshold') {
    return { fusion: 'gt', consumed: 2, threshold: b.value };
  }

  // blur(r=1, no sigma) + edges [grayscale input] → be
  if (a.name === 'blur' && a.radius === 1 && a.sigma === undefined && b?.name === 'edges') {
    return { fusion: 'be', consumed: 2 };
  }

  // blur(any radius) + edges [grayscale input] → bte (blurred then edges, any radius)
  if (a.name === 'blur' && b?.name === 'edges') {
    return a.sigma !== undefined
      ? { fusion: 'bte', consumed: 2, radius: a.radius, sigma: a.sigma }
      : { fusion: 'bte', consumed: 2, radius: a.radius };
  }

  return null;
}

// ── Fused kernel implementations ──────────────────────────────────────────────

function clampU8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

function computeGray(frame: VisionFrame): Uint8Array {
  const { width, height, channels, data } = frame;
  if (channels !== 3 && channels !== 4) {
    throw new Error('computeGray requires RGB or RGBA input');
  }
  const gray = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += channels) {
    gray[i] = (77 * data[p] + 150 * data[p + 1] + 29 * data[p + 2]) >> 8;
  }
  return gray;
}

function computeGraySAB(frame: VisionFrame): SharedArrayBuffer {
  const { width, height, channels, data } = frame;
  if (channels !== 3 && channels !== 4) {
    throw new Error('computeGraySAB requires RGB or RGBA input');
  }
  const sab = new SharedArrayBuffer(width * height);
  const gray = new Uint8Array(sab);
  for (let i = 0, p = 0; i < gray.length; i++, p += channels) {
    gray[i] = (77 * data[p] + 150 * data[p + 1] + 29 * data[p + 2]) >> 8;
  }
  return sab;
}

function blurRow3x3(gray: Uint8Array, width: number, height: number, y: number, out: Uint8Array): void {
  const ym = Math.max(0, y - 1),
    yp = Math.min(height - 1, y + 1);
  const rowA = ym * width,
    rowB = y * width,
    rowC = yp * width;

  for (let x = 0; x < width; x++) {
    const xm = x > 0 ? x - 1 : 0,
      xp = x < width - 1 ? x + 1 : width - 1;
    out[x] =
      (gray[rowA + xm] +
        2 * gray[rowA + x] +
        gray[rowA + xp] +
        2 * gray[rowB + xm] +
        4 * gray[rowB + x] +
        2 * gray[rowB + xp] +
        gray[rowC + xm] +
        2 * gray[rowC + x] +
        gray[rowC + xp]) >>
      4;
  }
}

function fusedGBE(frame: VisionFrame, gray?: Uint8Array): VisionFrame {
  const { width, height } = frame;
  const g = gray ?? computeGray(frame);
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;

  if (width < 3 || height < 3) {
    return out;
  }

  let prev = new Uint8Array(width);
  let curr = new Uint8Array(width);
  let next = new Uint8Array(width);

  blurRow3x3(g, width, height, 0, prev);
  blurRow3x3(g, width, height, 1, curr);

  for (let y = 1; y < height - 1; y++) {
    blurRow3x3(g, width, height, y + 1, next);
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const tl = prev[x - 1],
        tc = prev[x],
        tr = prev[x + 1];
      const ml = curr[x - 1],
        mr = curr[x + 1];
      const bl = next[x - 1],
        bc = next[x],
        br = next[x + 1];
      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      dst[row + x] = clampU8(Math.abs(gx) + Math.abs(gy));
    }
    const tmp = prev;
    prev = curr;
    curr = next;
    next = tmp;
  }

  return out;
}

function fusedBE(frame: VisionFrame): VisionFrame {
  if (frame.channels !== 1) {
    throw new Error('fused blur(1)+edges requires grayscale input');
  }
  return fusedGBE(frame, frame.data);
}

function fusedGT(frame: VisionFrame, threshold: number): VisionFrame {
  const { width, height, channels, data } = frame;
  if (channels !== 3 && channels !== 4) {
    throw new Error('fused grayscale+threshold requires RGB or RGBA input');
  }
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;
  for (let i = 0, p = 0; i < width * height; i++, p += channels) {
    const gray = (77 * data[p] + 150 * data[p + 1] + 29 * data[p + 2]) >> 8;
    dst[i] = gray >= threshold ? 255 : 0;
  }
  return out;
}

/** GTE: grayscale → threshold → erode, fused to avoid 2 intermediate frames. */
function fusedGTE(frame: VisionFrame, threshold: number, erodeRadius: number): VisionFrame {
  // First fuse grayscale+threshold
  const binary = fusedGT(frame, threshold);
  // Then erode (can't easily fuse further without changing morphology internals)
  return Kernels.erode(binary, erodeRadius);
}

/** BTE: blur(any radius) → edges, separated. Avoids intermediate Sobel-only frame. */
function fusedBTE(frame: VisionFrame, radius: number, sigma?: number): VisionFrame {
  if (frame.channels !== 1) {
    throw new Error('fused blur+edges requires grayscale input');
  }
  const blurred = Kernels.gaussianBlur(frame, radius, sigma);
  return Kernels.sobel(blurred);
}

// ── Parallel fused execution ──────────────────────────────────────────────────

async function parallelFuse(fusion: 'gbe' | 'gt' | 'be', frame: VisionFrame, opts: { threshold?: number } = {}): Promise<VisionFrame> {
  const { width, height, channels } = frame;
  const out = new VisionFrame(width, height, 1);
  const pool = getPipelineWorkerPool();
  const workerCount = Math.max(1, Math.min(NUM_PIPELINE_WORKERS, height));
  const rowsPerWorker = Math.ceil(height / workerCount);
  const grayBuf = fusion === 'gbe' ? computeGraySAB(frame) : fusion === 'be' ? frame.buffer : undefined;

  const jobs: Promise<void>[] = [];
  for (let wid = 0; wid < workerCount; wid++) {
    const rowStart = wid * rowsPerWorker;
    const rowEnd = Math.min(height, rowStart + rowsPerWorker);
    if (rowStart >= rowEnd) {
      continue;
    }
    jobs.push(
      pool.run({
        fusion,
        srcBuf: frame.buffer,
        dstBuf: out.buffer,
        ...(grayBuf !== undefined ? { grayBuf } : {}),
        width,
        height,
        channels,
        rowStart,
        rowEnd,
        ...(opts.threshold !== undefined ? { threshold: opts.threshold } : {}),
      }),
    );
  }

  await Promise.all(jobs);
  return out;
}

// ── Single-op runner ──────────────────────────────────────────────────────────

function runSingleOp(frame: VisionFrame, op: PipelineOp): VisionFrame {
  switch (op.name) {
    case 'grayscale':
      return Kernels.grayscale(frame);
    case 'blur':
      return Kernels.gaussianBlur(frame, op.radius, op.sigma);
    case 'edges':
      return Kernels.sobel(frame);
    case 'threshold':
      return Kernels.threshold(frame, op.value);
    case 'invert':
      return Kernels.invert(frame);
    case 'erode':
      return Kernels.erode(frame, op.radius);
    case 'dilate':
      return Kernels.dilate(frame, op.radius);
    case 'open':
      return Kernels.open(frame, op.radius);
    case 'close':
      return Kernels.close(frame, op.radius);
    case 'normalize':
      return Kernels.normalize(frame);
    case 'brightnessContrast':
      return Kernels.brightnessContrast(frame, op.brightness, op.contrast);
    case 'drawCircle':
      return Kernels.drawCircle(frame, op.cx, op.cy, op.radius, op.color, op.thickness);
    case 'drawRect':
      return Kernels.drawRect(frame, op.x, op.y, op.width, op.height, op.color, op.thickness);
  }
}

function runFusedSync(fused: FusionResult, frame: VisionFrame): VisionFrame {
  switch (fused.fusion) {
    case 'gbe':
      return fusedGBE(frame);
    case 'gt':
      return fusedGT(frame, fused.threshold);
    case 'be':
      return fusedBE(frame);
    case 'bte':
      return fusedBTE(frame, fused.radius, fused.sigma);
    case 'gte':
      return fusedGTE(frame, fused.threshold, fused.erodeRadius);
  }
}

// ── VisionPipeline ────────────────────────────────────────────────────────────
//
// IMPORTANT CONTRACT:
//   .run()      → synchronous, main thread, no workers. Safe for small frames.
//   .runAsync() → async, uses Workers for fused ops on frames >= 2MP.
//
// The VisionImage fluent API (.grayscale(), .blur(), etc.) runs immediately on
// the main thread. Use .pipeline() when you want deferred + optimized execution.

export class VisionPipeline<T extends ImageOwner> {
  private readonly ops: PipelineOp[] = [];

  constructor(private readonly image: T) {}

  grayscale(): this {
    this.ops.push({ name: 'grayscale' });
    return this;
  }
  blur(radius = 1, sigma?: number): this {
    this.ops.push(sigma !== undefined ? { name: 'blur', radius, sigma } : { name: 'blur', radius });
    return this;
  }
  edges(): this {
    this.ops.push({ name: 'edges' });
    return this;
  }
  threshold(value = 128): this {
    this.ops.push({ name: 'threshold', value });
    return this;
  }
  invert(): this {
    this.ops.push({ name: 'invert' });
    return this;
  }
  erode(radius = 1): this {
    this.ops.push({ name: 'erode', radius });
    return this;
  }
  dilate(radius = 1): this {
    this.ops.push({ name: 'dilate', radius });
    return this;
  }
  open(radius = 1): this {
    this.ops.push({ name: 'open', radius });
    return this;
  }
  close(radius = 1): this {
    this.ops.push({ name: 'close', radius });
    return this;
  }
  normalize(): this {
    this.ops.push({ name: 'normalize' });
    return this;
  }
  brightnessContrast(brightness = 0, contrast = 1): this {
    this.ops.push({ name: 'brightnessContrast', brightness, contrast });
    return this;
  }

  drawCircle(cx: number, cy: number, radius: number, color?: Kernels.DrawColor, thickness?: number): this {
    const op: PipelineOp = { name: 'drawCircle', cx, cy, radius };
    if (color !== undefined) {
      op.color = color;
    }
    if (thickness !== undefined) {
      op.thickness = thickness;
    }
    this.ops.push(op);
    return this;
  }

  drawRect(x: number, y: number, width: number, height: number, color?: Kernels.DrawColor, thickness?: number): this {
    const op: PipelineOp = { name: 'drawRect', x, y, width, height };
    if (color !== undefined) {
      op.color = color;
    }
    if (thickness !== undefined) {
      op.thickness = thickness;
    }
    this.ops.push(op);
    return this;
  }

  /** Execute pipeline synchronously on the main thread. Always safe; no Workers. */
  run(): T {
    let frame = this.image.frame;
    let i = 0;

    while (i < this.ops.length) {
      const fused = detectFusion(this.ops, i);
      if (fused) {
        frame = runFusedSync(fused, frame);
        i += fused.consumed;
        continue;
      }
      frame = runSingleOp(frame, this.ops[i]);
      i++;
    }

    this.image.frame = frame;
    return this.image;
  }

  /**
   * Execute pipeline asynchronously.
   * Fused ops on frames >= 2MP are offloaded to Worker threads automatically.
   * Falls back to sync execution for small frames or unfused ops.
   */
  async runAsync(): Promise<T> {
    let frame = this.image.frame;
    let i = 0;

    while (i < this.ops.length) {
      const fused = detectFusion(this.ops, i);

      if (fused) {
        const useWorkers =
          frame.width * frame.height >= FUSED_WORKER_THRESHOLD_PX && (fused.fusion === 'gbe' || fused.fusion === 'gt' || fused.fusion === 'be');

        if (useWorkers) {
          switch (fused.fusion) {
            case 'gbe':
              frame = await parallelFuse('gbe', frame);
              break;
            case 'gt':
              frame = await parallelFuse('gt', frame, { threshold: fused.threshold });
              break;
            case 'be':
              frame = await parallelFuse('be', frame);
              break;
          }
        } else {
          frame = runFusedSync(fused, frame);
        }

        i += fused.consumed;
        continue;
      }

      frame = runSingleOp(frame, this.ops[i]);
      i++;
    }

    this.image.frame = frame;
    return this.image;
  }
}
