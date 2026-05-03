import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DEFAULT_IDLE_TIMEOUT_MS = 10_000;

interface Job {
  coeffBuffer: SharedArrayBuffer;
  planeBuffer: SharedArrayBuffer;
  qt: Uint8Array;
  nbX: number;
  rowStart: number;
  rowEnd: number;
  planeWidth: number;
}

interface Pending {
  resolve: () => void;
  reject: (err: Error) => void;
}

class JpegIdctWorkerPool {
  private workers: Worker[] = [];
  private cursor = 0;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private closed = false;
  private idleTimer: NodeJS.Timeout | null = null;

  constructor(
    size = Math.max(1, Math.min(4, cpus().length - 1)),
    private readonly idleTimeoutMs = DEFAULT_IDLE_TIMEOUT_MS,
  ) {
    const workerFile = join(__dirname, '../workers/jpeg-idct.worker.js');

    for (let i = 0; i < size; i++) {
      const worker = new Worker(workerFile);
      worker.unref();

      worker.on('message', (msg: { id: number }) => {
        const item = this.pending.get(msg.id);
        if (!item) {
          return;
        }

        this.pending.delete(msg.id);
        item.resolve();

        this.scheduleIdleClose();
      });

      worker.on('error', (err) => {
        const error = err instanceof Error ? err : new Error(String(err));

        for (const item of this.pending.values()) {
          item.reject(error);
        }

        this.pending.clear();
        this.closeSoon();
      });

      worker.on('exit', (code) => {
        if (this.closed) {
          return;
        }

        if (code !== 0) {
          const error = new Error(`JPEG IDCT worker exited with code ${code}`);

          for (const item of this.pending.values()) {
            item.reject(error);
          }

          this.pending.clear();
          this.closeSoon();
        }
      });

      this.workers.push(worker);
    }

    this.scheduleIdleClose();
  }

  async run(job: Job): Promise<void> {
    if (this.closed) {
      return Promise.reject(new Error('JPEG IDCT worker pool is closed'));
    }

    if (this.workers.length === 0) {
      return Promise.reject(new Error('JPEG IDCT worker pool has no workers'));
    }

    this.cancelIdleClose();

    const id = ++this.seq;
    const worker = this.workers[this.cursor];
    this.cursor = (this.cursor + 1) % this.workers.length;

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });

      try {
        worker.postMessage({ id, ...job });
      } catch (err) {
        this.pending.delete(id);
        reject(err instanceof Error ? err : new Error(String(err)));
        this.scheduleIdleClose();
      }
    });
  }

  private scheduleIdleClose(): void {
    if (this.closed) {
      return;
    }
    if (this.pending.size > 0) {
      return;
    }

    this.cancelIdleClose();

    this.idleTimer = setTimeout(() => {
      void this.close();
    }, this.idleTimeoutMs);

    this.idleTimer.unref();
  }

  private cancelIdleClose(): void {
    if (!this.idleTimer) {
      return;
    }

    clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }

  private closeSoon(): void {
    this.cancelIdleClose();
    this.idleTimer = setTimeout(() => {
      void this.close();
    }, 0);
    this.idleTimer.unref();
  }

  async close(): Promise<void> {
    if (this.closed) {
      return;
    }

    this.closed = true;

    const error = new Error('JPEG IDCT worker pool closed');

    for (const item of this.pending.values()) {
      item.reject(error);
    }

    this.pending.clear();

    const workers = this.workers;
    this.workers = [];

    await Promise.allSettled(workers.map(async (worker) => worker.terminate()));

    if (sharedPool === this) {
      sharedPool = null;
    }
  }
}

let sharedPool: JpegIdctWorkerPool | null = null;

export function getJpegIdctWorkerPool(size?: number): JpegIdctWorkerPool {
  sharedPool ??= new JpegIdctWorkerPool(size);
  return sharedPool;
}
