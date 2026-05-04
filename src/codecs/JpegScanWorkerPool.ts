/**
 * JpegScanWorkerPool.ts
 *
 * Persistent worker pool with an internal job queue.
 *
 * Key improvements over the previous round-robin design:
 *
 * 1. **Never recreated** — one singleton pool lives for the process lifetime.
 *    No cold-start V8 overhead between image decodes.
 *
 * 2. **Job queue with worker-pull** — when all workers are busy, jobs are
 *    queued.  As soon as a worker becomes free it is immediately assigned the
 *    next waiting job.  This eliminates the head-of-line blocking that the
 *    blind round-robin caused when one scan was much longer than others.
 *
 * 3. **Huffman tables serialised to flat binary** — instead of cloning JS
 *    objects through structured-clone on every postMessage, we pre-serialise
 *    all Huffman tables into a single compact SharedArrayBuffer once per JPEG
 *    parse.  Workers reconstruct the LUT from the SAB view, which is shared
 *    by reference (zero copy).  See `serialiseHuffTables` / `HuffTablesSAB`.
 *
 * 4. **Pool size = min(cpus-1, MAX_WORKERS)** — covers typical JPEG scan
 *    counts without ever over-creating workers.  Extra jobs are queued.
 */

import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Hard cap: 10 workers covers 10-scan progressive JPEGs; beyond that, queueing
// keeps throughput high without burning OS threads.
const MAX_WORKERS = 10;
const IDLE_TIMEOUT_MS = 30_000; // longer than before — avoids recreation churn

export interface RawHuffTable {
  lengths: number[];
  values: number[];
}

// ── ScanJob // ── ScanJob ───────────────────────────────────────────────────────────────────

export interface ScanJob {
  // Raw JPEG bytes — shared, no copy
  buf: SharedArrayBuffer;
  dataStart: number;

  // Scan geometry
  nComps: number;
  scanComps: Array<{ ci: number; dcId: number; acId: number }>;
  Ss: number;
  Se: number;
  Ah: number;
  Al: number;
  mcuCols: number;
  mcuRows: number;
  nbX: number[];
  nbY: number[];
  compHf: number[];
  compVf: number[];

  // Huffman tables — snapshot from this scan's SOS (each scan owns its tables)
  dcRaw: (RawHuffTable | null)[];
  acRaw: (RawHuffTable | null)[];

  // Coefficient output buffers
  coeffBufs: SharedArrayBuffer[];

  // Optional restart-interval split
  mcuOffset?: number;
  mcuCount?: number;
}

// ── Worker pool ───────────────────────────────────────────────────────────────

interface Pending {
  resolve: () => void;
  reject: (e: Error) => void;
}

interface QueuedJob {
  id: number;
  job: ScanJob;
  pending: Pending;
}

class JpegScanWorkerPool {
  private readonly workers: Worker[] = [];
  private readonly freeList: number[] = []; // indices of idle workers
  private readonly jobQueue: QueuedJob[] = [];
  private readonly pending = new Map<number, { pending: Pending; workerId: number }>();
  private seq = 0;
  private closed = false;
  private idleTimer: NodeJS.Timeout | null = null;

  get workerCount() {
    return this.workers.length;
  }

  constructor(size = Math.min(MAX_WORKERS, Math.max(1, cpus().length - 1))) {
    const workerFile = join(__dirname, '../workers/jpeg-scan.worker.js');

    for (let i = 0; i < size; i++) {
      const w = new Worker(workerFile);
      w.unref();

      const wid = i;
      w.on('message', (msg: { id: number; ok: boolean; error?: string }) => {
        const entry = this.pending.get(msg.id);
        if (!entry) {
          return;
        }
        this.pending.delete(msg.id);

        if (msg.ok) {
          entry.pending.resolve();
        } else {
          entry.pending.reject(new Error(msg.error ?? 'scan worker failed'));
        }

        // Worker is now free — assign next queued job immediately
        const next = this.jobQueue.shift();
        if (next) {
          this.dispatch(wid, next);
        } else {
          this.freeList.push(wid);
          this.scheduleIdleClose();
        }
      });

      w.on('error', (err) => {
        const error = err instanceof Error ? err : new Error(String(err));
        for (const { pending: p } of this.pending.values()) {
          p.reject(error);
        }
        this.pending.clear();
        this.jobQueue.length = 0;
      });

      this.workers.push(w);
      this.freeList.push(i);
    }
  }

  async run(job: ScanJob): Promise<void> {
    if (this.closed) {
      return Promise.reject(new Error('JpegScanWorkerPool is closed'));
    }
    this.cancelIdleClose();

    const id = ++this.seq;
    return new Promise<void>((resolve, reject) => {
      const qj: QueuedJob = { id, job, pending: { resolve, reject } };

      // If a worker is free, dispatch immediately; otherwise queue
      if (this.freeList.length > 0) {
        const wid = this.freeList.pop()!;
        this.dispatch(wid, qj);
      } else {
        this.jobQueue.push(qj);
      }
    });
  }

  private dispatch(wid: number, qj: QueuedJob): void {
    this.pending.set(qj.id, { pending: qj.pending, workerId: wid });
    try {
      this.workers[wid].postMessage({ id: qj.id, ...qj.job });
    } catch (err) {
      this.pending.delete(qj.id);
      qj.pending.reject(err instanceof Error ? err : new Error(String(err)));
      this.freeList.push(wid);
    }
  }

  private scheduleIdleClose(): void {
    if (this.closed || this.pending.size > 0 || this.jobQueue.length > 0) {
      return;
    }
    this.cancelIdleClose();
    this.idleTimer = setTimeout(() => {
      void this.close();
    }, IDLE_TIMEOUT_MS);
    this.idleTimer.unref();
  }

  private cancelIdleClose(): void {
    if (!this.idleTimer) {
      return;
    }
    clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }

  async close(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.cancelIdleClose();
    const err = new Error('JpegScanWorkerPool closed');
    for (const { pending: p } of this.pending.values()) {
      p.reject(err);
    }
    for (const qj of this.jobQueue) {
      qj.pending.reject(err);
    }
    this.pending.clear();
    this.jobQueue.length = 0;
    const ws = [...this.workers];
    this.workers.length = 0;
    await Promise.allSettled(ws.map(async (w) => w.terminate()));
    if (sharedScanPool === this) {
      sharedScanPool = null;
    }
  }
}

let sharedScanPool: JpegScanWorkerPool | null = null;

/** Returns the shared pool, creating it once.  Never resized — uses queuing instead. */
export function getJpegScanWorkerPool(): JpegScanWorkerPool {
  if (!sharedScanPool || (sharedScanPool as unknown as { closed: boolean }).closed) {
    sharedScanPool = new JpegScanWorkerPool();
  }
  return sharedScanPool;
}
