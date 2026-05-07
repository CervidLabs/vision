import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MAX_WORKERS = 8;
const IDLE_TIMEOUT_MS = 30_000;

export interface PipelineWorkerJob {
  fusion: 'gbe' | 'gt' | 'be';
  srcBuf: SharedArrayBuffer;
  dstBuf: SharedArrayBuffer;
  grayBuf?: SharedArrayBuffer;
  width: number;
  height: number;
  channels: number;
  rowStart: number;
  rowEnd: number;
  threshold?: number;
}

interface WorkerResponse {
  id: number;
  ok: boolean;
  error?: string;
}

function isWorkerResponse(value: unknown): value is WorkerResponse {
  return typeof value === 'object' && value !== null && 'id' in value && typeof value.id === 'number';
}

interface Pending {
  resolve: () => void;
  reject: (error: Error) => void;
}

interface QueuedJob {
  id: number;
  job: PipelineWorkerJob;
  pending: Pending;
}

class PipelineWorkerPool {
  private readonly workers: Worker[] = [];
  private readonly freeList: number[] = [];
  private readonly jobQueue: QueuedJob[] = [];
  private readonly pending = new Map<number, { pending: Pending; wid: number }>();

  private seq = 0;
  private closed = false;
  private idleTimer: NodeJS.Timeout | null = null;

  constructor(size = Math.min(MAX_WORKERS, Math.max(1, cpus().length - 1))) {
    const workerFile = join(__dirname, '../workers/pipeline-fused.worker.js');

    for (let i = 0; i < size; i++) {
      const worker = new Worker(workerFile);
      worker.unref();

      const wid = i;

      worker.on('message', (msg: unknown) => {
        if (!isWorkerResponse(msg)) {
          return;
        }

        const entry = this.pending.get(msg.id);

        if (!entry) {
          return;
        }

        this.pending.delete(msg.id);

        if (msg.ok) {
          entry.pending.resolve();
        } else {
          entry.pending.reject(new Error(msg.error ?? 'Pipeline worker failed'));
        }

        this.releaseOrDispatch(wid);
      });

      worker.on('error', (err) => {
        const error = err instanceof Error ? err : new Error(String(err));

        for (const { pending } of this.pending.values()) {
          pending.reject(error);
        }

        for (const queued of this.jobQueue) {
          queued.pending.reject(error);
        }

        this.pending.clear();
        this.jobQueue.length = 0;
      });

      worker.on('exit', (code) => {
        if (this.closed || code === 0) {
          return;
        }

        const error = new Error(`Pipeline worker exited with code ${code}`);

        for (const { pending } of this.pending.values()) {
          pending.reject(error);
        }

        for (const queued of this.jobQueue) {
          queued.pending.reject(error);
        }

        this.pending.clear();
        this.jobQueue.length = 0;
      });

      this.workers.push(worker);
      this.freeList.push(i);
    }
  }

  async run(job: PipelineWorkerJob): Promise<void> {
    if (this.closed) {
      return Promise.reject(new Error('PipelineWorkerPool is closed'));
    }

    this.cancelIdleClose();

    const id = ++this.seq;

    return new Promise<void>((resolve, reject) => {
      const queued: QueuedJob = {
        id,
        job,
        pending: { resolve, reject },
      };

      if (this.freeList.length > 0) {
        this.dispatch(this.freeList.pop()!, queued);
      } else {
        this.jobQueue.push(queued);
      }
    });
  }

  private dispatch(wid: number, queued: QueuedJob): void {
    this.pending.set(queued.id, {
      pending: queued.pending,
      wid,
    });

    try {
      this.workers[wid].postMessage({
        id: queued.id,
        ...queued.job,
      });
    } catch (err) {
      this.pending.delete(queued.id);

      queued.pending.reject(err instanceof Error ? err : new Error(String(err)));

      this.freeList.push(wid);
    }
  }

  private releaseOrDispatch(wid: number): void {
    const next = this.jobQueue.shift();

    if (next) {
      this.dispatch(wid, next);
      return;
    }

    this.freeList.push(wid);
    this.scheduleIdleClose();
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

    const error = new Error('PipelineWorkerPool closed');

    for (const { pending } of this.pending.values()) {
      pending.reject(error);
    }

    for (const queued of this.jobQueue) {
      queued.pending.reject(error);
    }

    this.pending.clear();
    this.jobQueue.length = 0;

    const workers = [...this.workers];
    this.workers.length = 0;
    this.freeList.length = 0;

    await Promise.allSettled(workers.map(async (worker) => worker.terminate()));

    if (sharedPool === this) {
      sharedPool = null;
    }
  }
}

let sharedPool: PipelineWorkerPool | null = null;

export function getPipelineWorkerPool(): PipelineWorkerPool {
  sharedPool ??= new PipelineWorkerPool();

  return sharedPool;
}

export async function closePipelineWorkerPool(): Promise<void> {
  return sharedPool?.close();
}
