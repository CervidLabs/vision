/**
 * jpeg-scan.worker.ts
 *
 * Worker thread that parses ONE JPEG scan:
 * Huffman decode → coefficient accumulation.
 *
 * Receives:
 * - raw JPEG bytes as SharedArrayBuffer
 * - scan parameters
 * - Huffman tables
 * - shared coefficient buffers
 *
 * Supports:
 * - DC first  (Ss=0, Se=0, Ah=0)
 * - DC refine (Ss=0, Se=0, Ah>0)
 * - AC first  (Ss>0, Ah=0)
 * - AC refine (Ss>0, Ah>0)
 */

import { parentPort } from "node:worker_threads";

// ── BitReader ────────────────────────────────────────────────────────────────

class BitReader {
  private acc = 0;
  private avail = 0;
  private _rst = false;

  constructor(
    private readonly buf: Uint8Array,
    public pos: number
  ) { }

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
          continue;
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
  read1(): number {
    if (this.avail < 1) {
      this.fill();
    }

    this.avail--;
    return (this.acc >>> this.avail) & 1;
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

// ── Huffman decoder ──────────────────────────────────────────────────────────

const LUT_BITS = 9;

class HuffDec {
  private readonly lut = new Uint16Array(1 << LUT_BITS);
  private readonly minCode = new Int32Array(17).fill(-1);
  private readonly maxCode = new Int32Array(17).fill(-2);
  private readonly valOff = new Int32Array(17);
  private readonly values: Uint8Array;

  constructor(lengths: number[], values: number[]) {
    this.values = new Uint8Array(values);

    let code = 0;
    let vi = 0;

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

    throw new Error("JPEG scan worker: bad Huffman code");
  }
}

function signExtend(v: number, cat: number): number {
  return v < 1 << (cat - 1) ? v - ((1 << cat) - 1) : v;
}

// ── Message types ────────────────────────────────────────────────────────────

interface RawHuffTable {
  lengths: number[];
  values: number[];
}

interface ScanJob {
  id: number;

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

  // Restart-interval split: only process MCUs [mcuOffset, mcuOffset+mcuCount).
  // When set, dataStart points to the start of the restart interval byte.
  // dcPrev and eobRun reset to 0 at each restart boundary, so sub-jobs are independent.
  mcuOffset?: number;
  mcuCount?: number;
}

// ── Fast path: single component scan ─────────────────────────────────────────

function parseSingleComponentScan(
  reader: BitReader,
  coeffsAll: Int16Array[],
  dcAll: (HuffDec | null)[],
  acAll: (HuffDec | null)[],
  ci: number,
  dcId: number,
  acId: number,
  Ss: number,
  Se: number,
  Ah: number,
  Al: number,
  gridCols: number,
  gridRows: number,
  dcPrev: Int32Array,
  mcuOffset = 0,
  mcuCount = gridCols * gridRows,
): void {
  // For restart-interval sub-jobs: iterate only over [mcuOffset, mcuOffset+mcuCount).
  // dataStart already points to the correct restart interval byte, so the BitReader
  // is positioned correctly. dcPrev and eobRun are fresh (reset at every RST boundary).
  const mcuEnd = mcuOffset + mcuCount;
  const startRow = (mcuOffset / gridCols) | 0;
  const startCol = mcuOffset % gridCols;
  const cb = coeffsAll[ci];
  const dcHuff = dcAll[dcId];
  const acHuff = acAll[acId];

  let eobRun = 0;

  // ── DC first ──────────────────────────────────────────────
  if (Ss === 0 && Ah === 0) {
    let mcu = mcuOffset;
    for (let row = startRow; row < gridRows && mcu < mcuEnd; row++) {
      const colStart = row === startRow ? startCol : 0;
      let bo = (row * gridCols + colStart) * 64;

      for (let col = colStart; col < gridCols && mcu < mcuEnd; col++, bo += 64, mcu++) {
        const cat = dcHuff!.decode(reader);
        dcPrev[ci] += cat === 0 ? 0 : signExtend(reader.read(cat), cat);
        cb[bo] = dcPrev[ci] << Al;

        if (reader.restartSeen) {
          dcPrev.fill(0);
          eobRun = 0;
        }
      }
    }

    return;
  }

  // ── DC refinement ─────────────────────────────────────────
  if (Ss === 0 && Ah > 0) {
    const bit = 1 << Al;
    let mcu = mcuOffset;
    for (let row = startRow; row < gridRows && mcu < mcuEnd; row++) {
      const colStart = row === startRow ? startCol : 0;
      let bo = (row * gridCols + colStart) * 64;

      for (let col = colStart; col < gridCols && mcu < mcuEnd; col++, bo += 64, mcu++) {
        if (reader.read1()) {
          cb[bo] |= bit;
        }

        if (reader.restartSeen) {
          dcPrev.fill(0);
          eobRun = 0;
        }
      }
    }

    return;
  }

  // ── AC first ──────────────────────────────────────────────
  if (Ss > 0 && Ah === 0) {
    let mcu = mcuOffset;
    for (let row = startRow; row < gridRows && mcu < mcuEnd; row++) {
      const colStart = row === startRow ? startCol : 0;
      let bo = (row * gridCols + colStart) * 64;

      for (let col = colStart; col < gridCols && mcu < mcuEnd; col++, bo += 64, mcu++) {
        if (eobRun > 0) {
          eobRun--;
        } else {
          let k = Ss;

          while (k <= Se) {
            const sym = acHuff!.decode(reader);
            const ssss = sym & 0xf;
            const rrrr = (sym >> 4) & 0xf;

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

        if (reader.restartSeen) {
          dcPrev.fill(0);
          eobRun = 0;
        }
      }
    }

    return;
  }

  // ── AC refinement ─────────────────────────────────────────
  {
    const bit1 = 1 << Al;
    let mcu = mcuOffset;
    for (let row = startRow; row < gridRows && mcu < mcuEnd; row++) {
      const colStart = row === startRow ? startCol : 0;
      let bo = (row * gridCols + colStart) * 64;

      for (let col = colStart; col < gridCols && mcu < mcuEnd; col++, bo += 64, mcu++) {
        if (eobRun > 0) {
          for (let k = Ss; k <= Se; k++) {
            const p = bo + k;
            const v = cb[p];

            if (v !== 0 && reader.read1()) {
              cb[p] = v > 0 ? v + bit1 : v - bit1;
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
                  const p = bo + k;
                  const v = cb[p];

                  if (v !== 0) {
                    if (reader.read1()) {
                      cb[p] = v > 0 ? v + bit1 : v - bit1;
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
                  const p = bo + k;
                  const v = cb[p];

                  if (v !== 0 && reader.read1()) {
                    cb[p] = v > 0 ? v + bit1 : v - bit1;
                  }
                }

                break outer;
              }
            } else {
              while (k <= Se) {
                const p = bo + k;
                const v = cb[p];

                if (v !== 0) {
                  if (reader.read1()) {
                    cb[p] = v > 0 ? v + bit1 : v - bit1;
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
                cb[bo + k] = signExtend(reader.read1(), 1) << Al;
                k++;
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
}

// ── Generic path: interleaved/multi-component scan ───────────────────────────

function parseGenericScan(
  reader: BitReader,
  coeffs: Int16Array[],
  dc: (HuffDec | null)[],
  ac: (HuffDec | null)[],
  job: ScanJob
): void {
  const { nComps, scanComps, Ss, Se, Ah, Al } = job;
  const { mcuCols, mcuRows, nbX, compHf, compVf } = job;

  const dcPrev = new Int32Array(4);
  let eobRun = 0;

  const interleaved = nComps > 1;
  const gridCols = interleaved ? mcuCols : nbX[scanComps[0].ci];
  const gridRows = interleaved ? mcuRows : job.nbY[scanComps[0].ci];

  for (let mr = 0; mr < gridRows; mr++) {
    for (let mc = 0; mc < gridCols; mc++) {
      for (let si = 0; si < scanComps.length; si++) {
        const sc = scanComps[si];

        const ci = sc.ci;
        const dcHuff = dc[sc.dcId];
        const acHuff = ac[sc.acId];
        const cb = coeffs[ci];

        const bRows = interleaved ? compVf[ci] : 1;
        const bCols = interleaved ? compHf[ci] : 1;
        const nbXci = nbX[ci];

        for (let brow = 0; brow < bRows; brow++) {
          for (let bcol = 0; bcol < bCols; bcol++) {
            const col = interleaved ? mc * compHf[ci] + bcol : mc;
            const row = interleaved ? mr * compVf[ci] + brow : mr;
            const bo = (row * nbXci + col) * 64;

            // DC first
            if (Ss === 0 && Ah === 0) {
              const cat = dcHuff!.decode(reader);
              dcPrev[ci] += cat === 0 ? 0 : signExtend(reader.read(cat), cat);
              cb[bo] = dcPrev[ci] << Al;
            }

            // DC refinement
            else if (Ss === 0 && Ah > 0) {
              if (reader.read1()) {
                cb[bo] |= 1 << Al;
              }
            }

            // AC first
            else if (Ss > 0 && Ah === 0) {
              if (eobRun > 0) {
                eobRun--;
              } else {
                let k = Ss;

                while (k <= Se) {
                  const sym = acHuff!.decode(reader);
                  const ssss = sym & 0xf;
                  const rrrr = (sym >> 4) & 0xf;

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

            // AC refinement
            else {
              const bit1 = 1 << Al;

              if (eobRun > 0) {
                for (let k = Ss; k <= Se; k++) {
                  const p = bo + k;
                  const v = cb[p];

                  if (v !== 0 && reader.read1()) {
                    cb[p] = v > 0 ? v + bit1 : v - bit1;
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
                        const p = bo + k;
                        const v = cb[p];

                        if (v !== 0) {
                          if (reader.read1()) {
                            cb[p] = v > 0 ? v + bit1 : v - bit1;
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
                        const p = bo + k;
                        const v = cb[p];

                        if (v !== 0 && reader.read1()) {
                          cb[p] = v > 0 ? v + bit1 : v - bit1;
                        }
                      }

                      break outer;
                    }
                  } else {
                    while (k <= Se) {
                      const p = bo + k;
                      const v = cb[p];

                      if (v !== 0) {
                        if (reader.read1()) {
                          cb[p] = v > 0 ? v + bit1 : v - bit1;
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
                      cb[bo + k] = signExtend(reader.read1(), 1) << Al;
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

// ── Main parser ──────────────────────────────────────────────────────────────

function parseScan(job: ScanJob): void {
  const buf = new Uint8Array(job.buf);
  const reader = new BitReader(buf, job.dataStart);

  const dc = job.dcRaw.map((r) => (r ? new HuffDec(r.lengths, r.values) : null));
  const ac = job.acRaw.map((r) => (r ? new HuffDec(r.lengths, r.values) : null));
  const coeffs = job.coeffBufs.map((sab) => new Int16Array(sab));

  const { nComps, scanComps, Ss, Se, Ah, Al, nbX, nbY } = job;

  const interleaved = nComps > 1;

  if (!interleaved) {
    const sc = scanComps[0];
    const dcPrev = new Int32Array(4);

    parseSingleComponentScan(
      reader,
      coeffs,
      dc,
      ac,
      sc.ci,
      sc.dcId,
      sc.acId,
      Ss,
      Se,
      Ah,
      Al,
      nbX[sc.ci],
      nbY[sc.ci],
      dcPrev,
      job.mcuOffset ?? 0,
      job.mcuCount ?? nbX[sc.ci] * nbY[sc.ci],
    );

    return;
  }

  parseGenericScan(reader, coeffs, dc, ac, job);
}

// ── Entry point ──────────────────────────────────────────────────────────────

parentPort!.on("message", (job: ScanJob) => {
  try {
    parseScan(job);
    parentPort!.postMessage({ id: job.id, ok: true });
  } catch (err) {
    parentPort!.postMessage({
      id: job.id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
});