# Cervid Vision

**High-Performance Image Processing Engine for Node.js**

> Decode, process, and transform images at hardware-level speeds — purely in Node.js.

[![Node.js Version](https://img.shields.io/node/v/@cervid/vision.svg?style=flat-square)](https://nodejs.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

## The Bare-Metal Vision Engine

**Cervid/vision** is a high-performance, multithreaded image processing library designed to push Node.js beyond traditional limits. By leveraging `SharedArrayBuffer` and worker pools, Cervid processes large images with **extremely low latency** and a **predictable memory footprint**.

Unlike traditional pure-JS libraries that block the main thread, or native wrappers that incur heavy FFI (Foreign Function Interface) overhead, Cervid operates directly on raw memory using advanced multithreading.

---

## Why Cervid?

- ⚡ **Built for Node.js** — pure TypeScript/JS, zero native dependencies or build tools  
- 🧠 **Zero-Copy Architecture** — no memory duplication across the processing pipeline  
- 🧵 **True Multithreading** — parallel image processing using shared memory worker pools  
- 🚀 **Deferred Pipeline** — operation fusion to minimize memory traversals  
- 🎯 **OpenCV Alternative** — provides powerful, low-level computer vision kernels natively  

---

## Performance Beyond the Canvas

Traditional pure-Node.js image processing suffers from:
- Blocking the main Event Loop 
- Heavy Garbage Collection (GC) pressure
- Single-threaded execution bottlenecks

Cervid solves this by using **TypedArrays** for pixel storage, **SharedArrayBuffer** for zero-copy memory access across threads, and a **Deferred Pipeline** to fuse operations (like grayscale, blur, and edge detection) into a single optimized pass.

---

## Benchmark: Cervid vs Others

**Task:** Decode 4K Image + Grayscale + Gaussian Blur + Sobel Edge Detection  
**Environment:** Node.js 22.x | 8 Workers | Local Machine  

| Metric | Cervid | Jimp (Pure JS) | Sharp (Native/C++) |
| :--- | :---: | :---: | :---: |
| **Main Thread Blocked** | **< 2ms** | ~1200ms | ~5ms |
| **Total Execution Time** | **~45ms** | ~1200ms | ~30ms |
| **Peak Memory Usage** | **~60MB** | ~400MB+ | ~55MB |
| **Dependencies** | **0** | Many | Native Libvips |

> Cervid/vision achieves near-native performance by staying close to the metal, distributing work across CPU cores without the installation complexity of native binaries.

---

## Installation

```bash
npm install @cervid/vision
```

---

## Quick Start

```javascript
import { Vision } from '@cervid/vision';

async function main() {
    // Reads directly into SharedArrayBuffer for zero-copy processing
    const img = await Vision.read('./large_input.jpg');

    // Deferred pipeline: operations are queued and fused for optimal cache hits
    await img.pipeline()
        .grayscale()
        .blur(2)           // Gaussian blur with radius 2
        .edgesParallel(8)  // Multi-threaded Sobel edge detection using 8 workers
        .run();            // Executes the pipeline

    await img.save('./output_edges.jpg');
    console.log('Image processing complete!');
}

main().catch(console.error);
```

---

## Architecture Overview

### Pixel Storage Engine
Image data is stored in contiguous memory using **TypedArrays** (Uint8Array / Float32Array). This structure mimics low-level C++ structures, minimizing memory fragmentation and maximizing CPU cache locality during convolution operations.

### Parallel Execution Engine
Heavy workloads (like edge detection or resizing) are split across persistent **Workers**. By avoiding the traditional main thread bottleneck, Cervid can process gigapixel images without blocking your Node.js server.

### Zero-Copy Pipeline Model
All workers operate on shared memory via **SharedArrayBuffer**, eliminating pixel duplication. The **Deferred Pipeline** analyzes queued transformations and fuses them when possible, reducing the number of times the image data needs to be read from and written to memory.

---

## Key Features

* **Native Decoders:** High-performance, zero-copy binary decoding for JPEG, PNG, and PPM formats directly into shared memory.
* **Computer Vision Kernels:** Built-in low-level operations including Convolution, Adaptive Thresholding, Canny Edge Detection, and Morphological operations (Erode/Dilate).
* **Deferred Execution:** Chain multiple operations and compile them into a single memory pass for maximum throughput.

---

## Roadmap

* **WASM / SIMD Acceleration:** Leveraging hardware-level vectorization via WebAssembly for ultra-fast convolution and matrix operations.
* **WebGPU Support:** Offloading massively parallel pixel operations directly to the GPU for real-time video stream processing.
* **Advanced Streaming Codecs:** Out-of-core image decoding for medical and satellite imaging formats (TIFF) that exceed physical RAM.

---

## License

MIT © 2026 [Villager/Github](https://github.com/villager)
