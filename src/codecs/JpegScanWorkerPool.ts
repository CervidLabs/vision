/**
 * JpegScanWorkerPool.ts
 *
 * Manages a fixed pool of JPEG scan-parser workers.  Each worker can parse
 * one full scan (Huffman decode + coefficient accumulation) independently.
 *
 * Because scans in a progressive JPEG do NOT share bitstream state, all five
 * scans can be dispatched simultaneously and run in true parallel.
 */

import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DEFAULT_IDLE_TIMEOUT_MS = 10_000;

interface Pending {
  resolve: () => void;
  reject: (err: Error) => void;
}

export interface RawHuffTable {
  lengths: number[];
  values: number[];
}

export interface ScanJob {
  buf: SharedArrayBuffer;
  dataStart: number;
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
  dcRaw: (RawHuffTable | null)[];
  acRaw: (RawHuffTable | null)[];
  coeffBufs: SharedArrayBuffer[];
}

class JpegScanWorkerPool {
  private workers: Worker[] = [];
  private cursor = 0;
  get workerCount() {
    return this.workers.length;
  }
  private seq = 0;
  private pending = new Map<number, Pending>();
  private closed = false;
  private idleTimer: NodeJS.Timeout | null = null;

  constructor(
    size = Math.max(1, Math.min(8, cpus().length - 1)),
    private readonly idleTimeoutMs = DEFAULT_IDLE_TIMEOUT_MS,
  ) {
    const workerFile = join(__dirname, '../workers/jpeg-scan.worker.js');

    for (let i = 0; i < size; i++) {
      const w = new Worker(workerFile);
      w.unref();

      w.on('message', (msg: { id: number; ok: boolean; error?: string }) => {
        const item = this.pending.get(msg.id);
        if (!item) {
          return;
        }
        this.pending.delete(msg.id);
        if (msg.ok) {
          item.resolve();
        } else {
          item.reject(new Error(msg.error ?? 'scan worker failed'));
        }
        this.scheduleIdleClose();
      });

      w.on('error', (err) => {
        const error = err instanceof Error ? err : new Error(String(err));
        for (const item of this.pending.values()) {
          item.reject(error);
        }
        this.pending.clear();
        this.closeSoon();
      });

      this.workers.push(w);
    }

    this.scheduleIdleClose();
  }

  async run(job: ScanJob): Promise<void> {
    if (this.closed) {
      return Promise.reject(new Error('JpegScanWorkerPool is closed'));
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
    if (this.closed || this.pending.size > 0) {
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
    const err = new Error('JpegScanWorkerPool closed');
    for (const item of this.pending.values()) {
      item.reject(err);
    }
    this.pending.clear();
    const ws = this.workers;
    this.workers = [];
    await Promise.allSettled(ws.map(async (w) => w.terminate()));
    if (sharedScanPool === this) {
      sharedScanPool = null;
    }
  }
}

let sharedScanPool: JpegScanWorkerPool | null = null;

export function getJpegScanWorkerPool(size?: number): JpegScanWorkerPool {
  if (!sharedScanPool || (size !== undefined && size > sharedScanPool.workerCount)) {
    if (sharedScanPool) {
      void sharedScanPool.close();
    }
    sharedScanPool = new JpegScanWorkerPool(size);
  }
  return sharedScanPool;
}
