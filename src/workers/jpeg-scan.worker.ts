/**
 * jpeg-scan.worker.ts
 *
 * Worker thread that parses ONE JPEG scan (Huffman decode → coefficient
 * accumulation).  Receives the raw file buffer as SharedArrayBuffer, writes
 * quantised Int16 levels directly into per-component coefficient SABs.
 *
 * All five scan sub-types are supported:
 *   DC first  (Ss=0, Se=0, Ah=0)
 *   DC refine (Ss=0, Se=0, Ah>0)
 *   AC first  (Ss>0, Ah=0)
 *   AC refine (Ss>0, Ah>0)
 */

import { parentPort } from 'node:worker_threads';

// ── BitReader (self-contained copy — workers cannot import from main thread) ──

class BitReader {
  private acc = 0;
  private avail = 0;
  private _rst = false;

  constructor(
    private readonly buf: Uint8Array,
    public pos: number,
  ) {}

  get restartSeen(): boolean {
    const v = this._rst;
    this._rst = false;
    return v;
  }

  private fill(): void {
    while (this.avail < 16) {
      if (this.pos >= this.buf.length) {
        this.acc = ((this.acc << 8) | 0xff) >>> 0;
        this.avail += 8;
        continue;
      }
      const byte = this.buf[this.pos++];
      if (byte === 0xff) {
        const nxt = this.buf[this.pos];
        if (nxt === 0x00) {
          this.pos++;
        } else if (nxt >= 0xd0 && nxt <= 0xd7) {
          this.pos++;
          this._rst = true;
          this.acc = 0;
          this.avail = 0;
          continue; // keep filling from post-RST bytes
        }
      }
      this.acc = ((this.acc << 8) | byte) >>> 0;
      this.avail += 8;
    }
  }

  peek(n: number): number {
    if (this.avail < n) {
      this.fill();
    }
    return (this.acc >>> (this.avail - n)) & ((1 << n) - 1);
  }

  skip(n: number): void {
    this.avail -= n;
  }

  read(n: number): number {
    if (n === 0) {
      return 0;
    }
    if (this.avail < n) {
      this.fill();
    }
    this.avail -= n;
    return (this.acc >>> this.avail) & ((1 << n) - 1);
  }
}

// ── Huffman decode table with 9-bit LUT ──────────────────────────────────────

const LUT_BITS = 9;

class HuffDec {
  private readonly lut = new Uint16Array(1 << LUT_BITS);
  private readonly minCode = new Int32Array(17).fill(-1);
  private readonly maxCode = new Int32Array(17).fill(-2);
  private readonly valOff = new Int32Array(17);
  private readonly values: Uint8Array;

  constructor(lengths: number[], values: number[]) {
    this.values = new Uint8Array(values);
    let code = 0,
      vi = 0;
    for (let b = 1; b <= 16; b++) {
      const cnt = lengths[b - 1];
      if (cnt > 0) {
        this.minCode[b] = code;
        this.maxCode[b] = code + cnt - 1;
        this.valOff[b] = vi - code;
        if (b <= LUT_BITS) {
          const fill = 1 << (LUT_BITS - b);
          for (let j = 0; j < cnt; j++) {
            const entry = (values[vi + j] << 4) | b;
            const base = (code + j) << (LUT_BITS - b);
            for (let f = 0; f < fill; f++) {
              this.lut[base + f] = entry;
            }
          }
        }
        vi += cnt;
        code += cnt;
      }
      code <<= 1;
    }
  }

  decode(r: BitReader): number {
    const peek = r.peek(LUT_BITS);
    const entry = this.lut[peek];
    if (entry !== 0) {
      r.skip(entry & 0xf);
      return entry >> 4;
    }
    r.skip(LUT_BITS);
    let code = peek;
    for (let b = LUT_BITS + 1; b <= 16; b++) {
      code = (code << 1) | r.read(1);
      if (code >= this.minCode[b] && code <= this.maxCode[b]) {
        return this.values[code + this.valOff[b]];
      }
    }
    throw new Error('JPEG scan worker: bad Huffman code');
  }
}

function signExtend(v: number, cat: number): number {
  return v < 1 << (cat - 1) ? v - ((1 << cat) - 1) : v;
}

// ── Message types ─────────────────────────────────────────────────────────────

interface RawHuffTable {
  lengths: number[];
  values: number[];
}

interface ScanJob {
  id: number;
  // Raw JPEG bytes (SharedArrayBuffer so no copy)
  buf: SharedArrayBuffer;
  dataStart: number;
  // Scan parameters
  nComps: number;
  scanComps: Array<{ ci: number; dcId: number; acId: number }>;
  Ss: number;
  Se: number;
  Ah: number;
  Al: number;
  // MCU grid (for interleaved scans)
  mcuCols: number;
  mcuRows: number;
  // Per-component block grid
  nbX: number[];
  nbY: number[];
  compHf: number[];
  compVf: number[];
  // Huffman tables (raw, to rebuild HuffDec in this worker)
  dcRaw: (RawHuffTable | null)[]; // indexed 0-3
  acRaw: (RawHuffTable | null)[];
  // Coefficient output buffers (Int16, zigzag order, one SAB per component)
  coeffBufs: SharedArrayBuffer[];
}

// ── Scan parser ───────────────────────────────────────────────────────────────

function parseScan(job: ScanJob): void {
  const buf = new Uint8Array(job.buf);
  const reader = new BitReader(buf, job.dataStart);

  // Rebuild Huffman tables
  const dc = job.dcRaw.map((r) => (r ? new HuffDec(r.lengths, r.values) : null));
  const ac = job.acRaw.map((r) => (r ? new HuffDec(r.lengths, r.values) : null));

  // Attach Int16Array views onto the shared coefficient buffers
  const coeffs = job.coeffBufs.map((sab) => new Int16Array(sab));

  const { nComps, scanComps, Ss, Se, Ah, Al } = job;
  const { mcuCols, mcuRows, nbX, nbY, compHf, compVf } = job;

  const dcPrev = new Int32Array(Math.max(...scanComps.map((s) => s.ci)) + 1);
  let eobRun = 0;

  const interleaved = nComps > 1;
  const gridCols = interleaved ? mcuCols : nbX[scanComps[0].ci];
  const gridRows = interleaved ? mcuRows : nbY[scanComps[0].ci];

  for (let mr = 0; mr < gridRows; mr++) {
    for (let mc = 0; mc < gridCols; mc++) {
      for (const { ci, dcId, acId } of scanComps) {
        const dcHuff = dc[dcId];
        const acHuff = ac[acId];
        const cb = coeffs[ci];
        const bRows = interleaved ? compVf[ci] : 1;
        const bCols = interleaved ? compHf[ci] : 1;

        for (let brow = 0; brow < bRows; brow++) {
          for (let bcol = 0; bcol < bCols; bcol++) {
            const col = interleaved ? mc * compHf[ci] + bcol : mc;
            const row = interleaved ? mr * compVf[ci] + brow : mr;
            const bo = (row * nbX[ci] + col) * 64;

            // ── DC first pass ─────────────────────────────────────
            if (Ss === 0 && Ah === 0) {
              const cat = dcHuff!.decode(reader);
              dcPrev[ci] += cat === 0 ? 0 : signExtend(reader.read(cat), cat);
              cb[bo] = dcPrev[ci] << Al;
            }
            // ── DC refinement ─────────────────────────────────────
            else if (Ss === 0 && Ah > 0) {
              if (reader.read(1)) {
                cb[bo] |= 1 << Al;
              }
            }
            // ── AC first pass ─────────────────────────────────────
            else if (Ss > 0 && Ah === 0) {
              if (eobRun > 0) {
                eobRun--;
              } else {
                let k = Ss;
                while (k <= Se) {
                  const sym = acHuff!.decode(reader);
                  const ssss = sym & 0xf,
                    rrrr = (sym >> 4) & 0xf;
                  if (ssss === 0) {
                    if (rrrr === 15) {
                      k += 16;
                    } else if (rrrr === 0) {
                      break;
                    } else {
                      eobRun = (1 << rrrr) + reader.read(rrrr) - 1;
                      break;
                    }
                  } else {
                    k += rrrr;
                    if (k > Se) {
                      break;
                    }
                    cb[bo + k] = signExtend(reader.read(ssss), ssss) << Al;
                    k++;
                  }
                }
              }
            }
            // ── AC refinement ─────────────────────────────────────
            else {
              const bit1 = 1 << Al;
              const refine = (v: number) => (v > 0 ? v + bit1 : v - bit1);
              if (eobRun > 0) {
                for (let k = Ss; k <= Se; k++) {
                  if (cb[bo + k] !== 0 && reader.read(1)) {
                    cb[bo + k] = refine(cb[bo + k]);
                  }
                }
                eobRun--;
              } else {
                let k = Ss;
                outer: while (k <= Se) {
                  const sym = acHuff!.decode(reader);
                  const ssss = sym & 0xf;
                  let rrrr = (sym >> 4) & 0xf;
                  if (ssss === 0) {
                    if (rrrr === 15) {
                      let zeros = 16;
                      while (k <= Se && zeros > 0) {
                        if (cb[bo + k] !== 0) {
                          if (reader.read(1)) {
                            cb[bo + k] = refine(cb[bo + k]);
                          }
                        } else {
                          zeros--;
                        }
                        k++;
                      }
                    } else {
                      if (rrrr > 0) {
                        eobRun = (1 << rrrr) + reader.read(rrrr) - 1;
                      }
                      for (; k <= Se; k++) {
                        if (cb[bo + k] !== 0 && reader.read(1)) {
                          cb[bo + k] = refine(cb[bo + k]);
                        }
                      }
                      break outer;
                    }
                  } else {
                    while (k <= Se) {
                      if (cb[bo + k] !== 0) {
                        if (reader.read(1)) {
                          cb[bo + k] = refine(cb[bo + k]);
                        }
                        k++;
                      } else {
                        if (rrrr === 0) {
                          break;
                        }
                        rrrr--;
                        k++;
                      }
                    }
                    if (k <= Se) {
                      cb[bo + k] = signExtend(reader.read(1), 1) << Al;
                      k++;
                    }
                  }
                }
              }
            }
          }
        }
      }

      if (reader.restartSeen) {
        dcPrev.fill(0);
        eobRun = 0;
      }
    }
  }
}

// ── Entry point ───────────────────────────────────────────────────────────────

parentPort!.on('message', (job: ScanJob) => {
  try {
    parseScan(job);
    parentPort!.postMessage({ id: job.id, ok: true });
  } catch (err) {
    parentPort!.postMessage({ id: job.id, ok: false, error: String(err) });
  }
});
