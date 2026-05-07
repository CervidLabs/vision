import { VisionFrame } from '../core/VisionFrame.js';

function assertGray(frame: VisionFrame, name: string): void {
  if (frame.channels !== 1) {
    throw new Error(`${name}: expected 1-channel grayscale/binary frame`);
  }
}

// ── Sliding window min/max via monotone deque — O(n) per row/col ─────────────
//
// For radius r, window size = 2r+1.
// Input is padded by replicating edge values so output length == input length.

function slidingWindowMin(arr: Uint8Array, n: number, r: number): Uint8Array {
  if (r <= 0) {
    return arr.slice(0, n);
  }

  const W = 2 * r + 1;
  // Build padded array: [arr[0]×r | arr | arr[n-1]×r]
  const padded = new Uint8Array(n + 2 * r);
  padded.fill(arr[0], 0, r);
  padded.set(arr.subarray(0, n), r);
  padded.fill(arr[n - 1], r + n, r + n + r);

  const out = new Uint8Array(n);
  const dq = new Int32Array(n + 2 * r); // circular-ish indices
  let head = 0,
    tail = 0;

  for (let i = 0; i < n + 2 * r; i++) {
    const v = padded[i];
    // Pop from back while back value >= current (won't be minimum)
    while (tail > head && padded[dq[tail - 1]] >= v) {
      tail--;
    }
    dq[tail++] = i;
    // Pop from front if out of window
    if (dq[head] <= i - W) {
      head++;
    }
    // Output: output index is i - (W-1) = i - 2r
    if (i >= W - 1) {
      out[i - (W - 1)] = padded[dq[head]];
    }
  }

  return out;
}

function slidingWindowMax(arr: Uint8Array, n: number, r: number): Uint8Array {
  if (r <= 0) {
    return arr.slice(0, n);
  }

  const W = 2 * r + 1;
  const padded = new Uint8Array(n + 2 * r);
  padded.fill(arr[0], 0, r);
  padded.set(arr.subarray(0, n), r);
  padded.fill(arr[n - 1], r + n, r + n + r);

  const out = new Uint8Array(n);
  const dq = new Int32Array(n + 2 * r);
  let head = 0,
    tail = 0;

  for (let i = 0; i < n + 2 * r; i++) {
    const v = padded[i];
    // Pop from back while back value <= current (won't be maximum)
    while (tail > head && padded[dq[tail - 1]] <= v) {
      tail--;
    }
    dq[tail++] = i;
    if (dq[head] <= i - W) {
      head++;
    }
    if (i >= W - 1) {
      out[i - (W - 1)] = padded[dq[head]];
    }
  }

  return out;
}

// ── 2D separable morphology ───────────────────────────────────────────────────
//
// Both erode and dilate are separable: apply 1D op horizontally then vertically.

function morphology2D(frame: VisionFrame, radius: number, op: (arr: Uint8Array, n: number, r: number) => Uint8Array): VisionFrame {
  const { width, height } = frame;
  const src = frame.data;

  // ── Horizontal pass ──
  const hBuf = new Uint8Array(width * height);
  const rowScratch = new Uint8Array(width);

  for (let y = 0; y < height; y++) {
    const offset = y * width;
    rowScratch.set(src.subarray(offset, offset + width));
    hBuf.set(op(rowScratch, width, radius), offset);
  }

  // ── Vertical pass ──
  const out = new VisionFrame(width, height, 1);
  const dst = out.data;
  const colScratch = new Uint8Array(height);

  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      colScratch[y] = hBuf[y * width + x];
    }
    const colOut = op(colScratch, height, radius);
    for (let y = 0; y < height; y++) {
      dst[y * width + x] = colOut[y];
    }
  }

  return out;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Morphological erosion (sliding minimum).
 * O(width × height) regardless of radius — uses monotone deque.
 */
export function erode(frame: VisionFrame, radius = 1): VisionFrame {
  assertGray(frame, 'erode');
  if (radius < 1) {
    return frame.clone();
  }
  return morphology2D(frame, radius, slidingWindowMin);
}

/**
 * Morphological dilation (sliding maximum).
 * O(width × height) regardless of radius — uses monotone deque.
 */
export function dilate(frame: VisionFrame, radius = 1): VisionFrame {
  assertGray(frame, 'dilate');
  if (radius < 1) {
    return frame.clone();
  }
  return morphology2D(frame, radius, slidingWindowMax);
}

/** Morphological opening: erode then dilate. Removes small bright spots. */
export function open(frame: VisionFrame, radius = 1): VisionFrame {
  return dilate(erode(frame, radius), radius);
}

/** Morphological closing: dilate then erode. Fills small dark holes. */
export function close(frame: VisionFrame, radius = 1): VisionFrame {
  return erode(dilate(frame, radius), radius);
}

/** Morphological gradient: dilate − erode. Produces edge outline. */
export function morphGradient(frame: VisionFrame, radius = 1): VisionFrame {
  assertGray(frame, 'morphGradient');
  const d = dilate(frame, radius).data;
  const e = erode(frame, radius).data;
  const out = new VisionFrame(frame.width, frame.height, 1);
  const dst = out.data;
  for (let i = 0; i < dst.length; i++) {
    dst[i] = d[i] - e[i];
  }
  return out;
}

/** Top-hat transform: original − open. Extracts bright features smaller than the structuring element. */
export function topHat(frame: VisionFrame, radius = 1): VisionFrame {
  assertGray(frame, 'topHat');
  const src = frame.data;
  const opened = open(frame, radius).data;
  const out = new VisionFrame(frame.width, frame.height, 1);
  const dst = out.data;
  for (let i = 0; i < dst.length; i++) {
    dst[i] = Math.max(0, src[i] - opened[i]);
  }
  return out;
}

/** Black-hat transform: close − original. Extracts dark features smaller than the structuring element. */
export function blackHat(frame: VisionFrame, radius = 1): VisionFrame {
  assertGray(frame, 'blackHat');
  const src = frame.data;
  const closed = close(frame, radius).data;
  const out = new VisionFrame(frame.width, frame.height, 1);
  const dst = out.data;
  for (let i = 0; i < dst.length; i++) {
    dst[i] = Math.max(0, closed[i] - src[i]);
  }
  return out;
}
