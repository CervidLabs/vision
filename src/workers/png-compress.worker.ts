/**
 * png-compress.worker.ts
 *
 * Compresses one chunk of PNG scanline data using deflateRawSync.
 * Runs in a real OS thread (Worker), NOT in the libuv thread pool,
 * allowing true N-way parallelism beyond libuv's 4-thread limit.
 *
 * Concatenation strategy
 * ──────────────────────
 * Non-final chunks use Z_SYNC_FLUSH.  This appends a 5-byte zero-length
 * stored block:  00 00 00 FF FF
 *   byte 1: 0x00 = BFINAL(0) | BTYPE(00) | padding(00000)
 *   bytes 2-3: LEN  = 0x0000
 *   bytes 4-5: NLEN = 0xFFFF  (~LEN ✓)
 *
 * We keep these bytes intact — they are valid DEFLATE.  When the decoder
 * hits this block it reads LEN=0, verifies NLEN, copies zero bytes, and
 * continues to the next chunk's compressed blocks.  No stripping needed.
 *
 * The final chunk uses Z_FINISH (default), producing BFINAL=1 in its
 * last block — the end-of-stream signal for the decoder.
 *
 * ── Why finishFlush and not flush ────────────────────────────────────────────
 * In Node.js zlib, `flush` controls intermediate flush points in streaming
 * mode. `finishFlush` controls what happens at the END of the stream — which
 * is what we need here since deflateRawSync is a single one-shot call.
 * Using `flush` instead of `finishFlush` causes every chunk to be finalized
 * with Z_FINISH (BFINAL=1), making concatenation invalid: the PNG decoder
 * stops at the first BFINAL=1 and ignores all subsequent chunks → black PNG.
 */

import { parentPort } from 'node:worker_threads';
import { deflateRawSync, constants } from 'node:zlib';

interface CompressJob {
  id: number;
  dataSAB: SharedArrayBuffer;
  offset: number;
  length: number;
  isFinal: boolean;
  level: number;
}

parentPort!.on('message', (job: CompressJob) => {
  try {
    const input = Buffer.from(job.dataSAB, job.offset, job.length);

    const finishFlush = job.isFinal
      ? constants.Z_FINISH // final block: BFINAL=1, end-of-stream
      : constants.Z_SYNC_FLUSH; // non-final: BFINAL=0, appends 00 00 00 FF FF

    // ── KEY FIX: use `finishFlush`, not `flush` ───────────────────────────
    // `flush`       = intermediate flush mode (used in streaming, ignored here)
    // `finishFlush` = final flush mode (used by deflateRawSync at end of input)
    const compressed = deflateRawSync(input, { level: job.level, finishFlush });

    const owned = compressed.buffer.slice(compressed.byteOffset, compressed.byteOffset + compressed.byteLength);

    parentPort!.postMessage({ id: job.id, ok: true, data: owned }, [owned]);
  } catch (err) {
    parentPort!.postMessage({
      id: job.id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
});
