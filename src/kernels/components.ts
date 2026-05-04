import { VisionFrame } from "../core/VisionFrame.js";

export interface ConnectedComponent {
    id: number;
    x: number;
    y: number;
    width: number;
    height: number;
    area: number;
    centroidX: number;
    centroidY: number;
}

export interface ConnectedComponentsOptions {
    minArea?: number;
    maxArea?: number;
    connectivity?: 4 | 8;
}

function assertGray(frame: VisionFrame, name: string): void {
    if (frame.channels !== 1) {
        throw new Error(`${name}: expected 1-channel grayscale/binary frame`);
    }
}

function active(v: number): boolean {
    return v > 0;
}

export function connectedComponents(
    frame: VisionFrame,
    options: ConnectedComponentsOptions = {},
): ConnectedComponent[] {
    assertGray(frame, "connectedComponents");

    const { width, height } = frame;
    const src = frame.data;

    const minArea = options.minArea ?? 1;
    const maxArea = options.maxArea ?? Number.POSITIVE_INFINITY;
    const connectivity = options.connectivity ?? 8;

    const labels = new Int32Array(width * height);
    const stack = new Int32Array(width * height);

    const components: ConnectedComponent[] = [];
    let nextId = 1;

    const dx4 = [1, -1, 0, 0];
    const dy4 = [0, 0, 1, -1];

    const dx8 = [1, -1, 0, 0, 1, 1, -1, -1];
    const dy8 = [0, 0, 1, -1, 1, -1, 1, -1];

    const dx = connectivity === 4 ? dx4 : dx8;
    const dy = connectivity === 4 ? dy4 : dy8;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const start = y * width + x;

            if (labels[start] !== 0 || !active(src[start])) {
                continue;
            }

            const id = nextId++;
            let top = 0;

            labels[start] = id;
            stack[top++] = start;

            let minX = x;
            let maxX = x;
            let minY = y;
            let maxY = y;
            let area = 0;
            let sumX = 0;
            let sumY = 0;

            while (top > 0) {
                const p = stack[--top];
                const py = (p / width) | 0;
                const px = p - py * width;

                area++;
                sumX += px;
                sumY += py;

                if (px < minX) minX = px;
                if (px > maxX) maxX = px;
                if (py < minY) minY = py;
                if (py > maxY) maxY = py;

                for (let i = 0; i < dx.length; i++) {
                    const nx = px + dx[i];
                    const ny = py + dy[i];

                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                        continue;
                    }

                    const ni = ny * width + nx;

                    if (labels[ni] !== 0 || !active(src[ni])) {
                        continue;
                    }

                    labels[ni] = id;
                    stack[top++] = ni;
                }
            }

            if (area < minArea || area > maxArea) {
                continue;
            }

            components.push({
                id,
                x: minX,
                y: minY,
                width: maxX - minX + 1,
                height: maxY - minY + 1,
                area,
                centroidX: sumX / area,
                centroidY: sumY / area,
            });
        }
    }

    return components;
}

export function filterComponents(
    components: ConnectedComponent[],
    options: ConnectedComponentsOptions = {},
): ConnectedComponent[] {
    const minArea = options.minArea ?? 1;
    const maxArea = options.maxArea ?? Number.POSITIVE_INFINITY;

    return components.filter((c) => c.area >= minArea && c.area <= maxArea);
}