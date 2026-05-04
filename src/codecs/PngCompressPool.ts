/**
 * PngCompressPool.ts
 *
 * Persistent pool of png-compress worker threads.
 *
 * Workers stay alive between PNG encodes (idle-timeout = 30 s), eliminating
 * the ~30 ms startup cost per worker.  A job queue with worker-pull ensures
 * all N threads stay saturated when more chunks arrive than there are workers.
 */

import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MAX_WORKERS = 8;
const IDLE_TIMEOUT_MS = 30_000;

export interface CompressJob {
  dataSAB: SharedArrayBuffer;
  offset: number;
  length: number;
  isFinal: boolean;
  level: number;
}

interface Pending {
  resolve: (data: ArrayBuffer) => void;
  reject: (e: Error) => void;
}
interface QueuedJob {
  id: number;
  job: CompressJob;
  pending: Pending;
}

class PngCompressPool {
  private readonly workers: Worker[] = [];
  private readonly freeList: number[] = [];
  private readonly jobQueue: QueuedJob[] = [];
  private readonly pending = new Map<number, { pending: Pending; wid: number }>();
  private seq = 0;
  private closed = false;
  private idleTimer: NodeJS.Timeout | null = null;

  get workerCount() {
    return this.workers.length;
  }

  constructor(size = Math.min(MAX_WORKERS, Math.max(1, cpus().length - 1))) {
    const workerFile = join(__dirname, '../workers/png-compress.worker.js');

    for (let i = 0; i < size; i++) {
      const w = new Worker(workerFile);
      w.unref();

      const wid = i;
      w.on('message', (msg: { id: number; ok: boolean; data?: ArrayBuffer; error?: string }) => {
        const entry = this.pending.get(msg.id);
        if (!entry) {
          return;
        }
        this.pending.delete(msg.id);

        if (msg.ok && msg.data !== undefined) {
          entry.pending.resolve(msg.data);
        } else {
          entry.pending.reject(new Error(msg.error ?? 'png compress worker failed'));
        }

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

  async run(job: CompressJob): Promise<ArrayBuffer> {
    if (this.closed) {
      return Promise.reject(new Error('PngCompressPool is closed'));
    }
    this.cancelIdleClose();

    const id = ++this.seq;
    return new Promise<ArrayBuffer>((resolve, reject) => {
      const qj: QueuedJob = { id, job, pending: { resolve, reject } };
      if (this.freeList.length > 0) {
        this.dispatch(this.freeList.pop()!, qj);
      } else {
        this.jobQueue.push(qj);
      }
    });
  }

  private dispatch(wid: number, qj: QueuedJob): void {
    this.pending.set(qj.id, { pending: qj.pending, wid });
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
    const err = new Error('PngCompressPool closed');
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
    if (sharedPool === this) {
      sharedPool = null;
    }
  }
}

let sharedPool: PngCompressPool | null = null;

export function getPngCompressPool(): PngCompressPool {
  if (!sharedPool || (sharedPool as unknown as { closed: boolean }).closed) {
    sharedPool = new PngCompressPool();
  }
  return sharedPool;
}
