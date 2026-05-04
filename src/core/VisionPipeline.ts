import { VisionFrame } from './VisionFrame.js';
import * as Kernels from '../kernels/index.js';

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
  | { name: 'drawCircle'; cx: number; cy: number; radius: number; color?: Kernels.DrawColor; thickness?: number }
  | { name: 'drawRect'; x: number; y: number; width: number; height: number; color?: Kernels.DrawColor; thickness?: number };

function clampU8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

function computeGrayFull(frame: VisionFrame): Uint8Array {
  if (frame.channels !== 3 && frame.channels !== 4) {
    throw new Error('fused grayscale requires RGB or RGBA input');
  }

  const { width, height, channels } = frame;
  const src = frame.data;
  const gray = new Uint8Array(width * height);

  for (let i = 0, p = 0; i < gray.length; i++, p += channels) {
    const r = src[p];
    const g = src[p + 1];
    const b = src[p + 2];

    gray[i] = (77 * r + 150 * g + 29 * b) >> 8;
  }

  return gray;
}

function computeGaussianBlurRow3x3(gray: Uint8Array, width: number, height: number, y: number, out: Uint8Array): void {
  const ym1 = Math.max(0, y - 1);
  const yp1 = Math.min(height - 1, y + 1);

  const rowA = ym1 * width;
  const rowB = y * width;
  const rowC = yp1 * width;

  for (let x = 0; x < width; x++) {
    const xm1 = x > 0 ? x - 1 : 0;
    const xp1 = x < width - 1 ? x + 1 : width - 1;

    const v =
      gray[rowA + xm1] +
      2 * gray[rowA + x] +
      gray[rowA + xp1] +
      2 * gray[rowB + xm1] +
      4 * gray[rowB + x] +
      2 * gray[rowB + xp1] +
      gray[rowC + xm1] +
      2 * gray[rowC + x] +
      gray[rowC + xp1];

    out[x] = v >> 4;
  }
}

/**
 * Fused path:
 * RGB/RGBA → grayscale → blur(1) → sobel
 *
 * Evita crear VisionFrame grayscale y VisionFrame blur.
 * Usa gray temporal + 3 filas de blur.
 */
function fusedGrayscaleBlur1Edges(frame: VisionFrame): VisionFrame {
  const { width, height } = frame;

  const gray = computeGrayFull(frame);
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;

  if (width < 3 || height < 3) {
    return out;
  }

  let prev = new Uint8Array(width);
  let curr = new Uint8Array(width);
  let next = new Uint8Array(width);

  computeGaussianBlurRow3x3(gray, width, height, 0, prev);
  computeGaussianBlurRow3x3(gray, width, height, 1, curr);

  for (let y = 1; y < height - 1; y++) {
    computeGaussianBlurRow3x3(gray, width, height, y + 1, next);

    const outRow = y * width;

    for (let x = 1; x < width - 1; x++) {
      const tl = prev[x - 1];
      const tc = prev[x];
      const tr = prev[x + 1];

      const ml = curr[x - 1];
      const mr = curr[x + 1];

      const bl = next[x - 1];
      const bc = next[x];
      const br = next[x + 1];

      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;

      dst[outRow + x] = clampU8(Math.abs(gx) + Math.abs(gy));
    }

    const tmp = prev;
    prev = curr;
    curr = next;
    next = tmp;
  }

  return out;
}

function canFuseGrayscaleBlurEdges(ops: PipelineOp[]): boolean {
  if (ops.length < 3) {
    return false;
  }

  const a = ops[0];
  const b = ops[1];
  const c = ops[2];

  return a.name === 'grayscale' && b.name === 'blur' && b.radius === 1 && b.sigma === undefined && c.name === 'edges';
}

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

    case 'drawCircle':
      return Kernels.drawCircle(frame, op.cx, op.cy, op.radius, op.color, op.thickness);

    case 'drawRect':
      return Kernels.drawRect(frame, op.x, op.y, op.width, op.height, op.color, op.thickness);

    default:
      throw new Error(`Unknown pipeline operation`);
  }
}

export class VisionPipeline<T extends ImageOwner> {
  private readonly ops: PipelineOp[] = [];

  constructor(private readonly image: T) {}

  grayscale(): this {
    this.ops.push({ name: 'grayscale' });
    return this;
  }

  blur(radius = 1, sigma?: number): this {
    if (sigma === undefined) {
      this.ops.push({ name: 'blur', radius });
    } else {
      this.ops.push({ name: 'blur', radius, sigma });
    }

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

  drawCircle(cx: number, cy: number, radius: number, color?: Kernels.DrawColor, thickness?: number): this {
    const op: PipelineOp = {
      name: 'drawCircle',
      cx,
      cy,
      radius,
    };

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
    const op: PipelineOp = {
      name: 'drawRect',
      x,
      y,
      width,
      height,
    };

    if (color !== undefined) {
      op.color = color;
    }

    if (thickness !== undefined) {
      op.thickness = thickness;
    }

    this.ops.push(op);
    return this;
  }
  run(): T {
    let frame = this.image.frame;
    let start = 0;

    if (canFuseGrayscaleBlurEdges(this.ops)) {
      frame = fusedGrayscaleBlur1Edges(frame);
      start = 3;
    }

    for (let i = start; i < this.ops.length; i++) {
      frame = runSingleOp(frame, this.ops[i]);
    }

    this.image.frame = frame;
    return this.image;
  }
}
