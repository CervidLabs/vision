import { VisionFrame } from "../core/VisionFrame.js";
import type { ConnectedComponent } from "./components.js";

export interface DrawColor {
    r?: number;
    g?: number;
    b?: number;
    a?: number;
    gray?: number;
}

function colorFor(frame: VisionFrame, color: DrawColor): number[] {
    if (frame.channels === 1) {
        return [color.gray ?? color.r ?? 255];
    }

    if (frame.channels === 3) {
        return [
            color.r ?? 255,
            color.g ?? 0,
            color.b ?? 0,
        ];
    }

    return [
        color.r ?? 255,
        color.g ?? 0,
        color.b ?? 0,
        color.a ?? 255,
    ];
}

function setPixelUnsafe(
    frame: VisionFrame,
    x: number,
    y: number,
    values: number[],
): void {
    const off = (y * frame.width + x) * frame.channels;

    for (let c = 0; c < frame.channels; c++) {
        frame.data[off + c] = values[c] ?? values[0] ?? 255;
    }
}

function setPixel(
    frame: VisionFrame,
    x: number,
    y: number,
    values: number[],
): void {
    if (x < 0 || y < 0 || x >= frame.width || y >= frame.height) {
        return;
    }

    setPixelUnsafe(frame, x, y, values);
}

function drawThickPoint(
    frame: VisionFrame,
    x: number,
    y: number,
    values: number[],
    thickness: number,
): void {
    const r = Math.max(0, (thickness / 2) | 0);

    if (r <= 0) {
        setPixel(frame, x, y, values);
        return;
    }

    for (let yy = y - r; yy <= y + r; yy++) {
        for (let xx = x - r; xx <= x + r; xx++) {
            setPixel(frame, xx, yy, values);
        }
    }
}

export function drawPoint(
    frame: VisionFrame,
    x: number,
    y: number,
    color: DrawColor = { r: 255, g: 0, b: 0 },
    thickness = 1,
): VisionFrame {
    const out = frame.clone();
    const values = colorFor(out, color);

    drawThickPoint(
        out,
        Math.round(x),
        Math.round(y),
        values,
        Math.max(1, thickness),
    );

    return out;
}

export function drawLine(
    frame: VisionFrame,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    color: DrawColor = { r: 255, g: 0, b: 0 },
    thickness = 1,
): VisionFrame {
    const out = frame.clone();
    const values = colorFor(out, color);

    let x = Math.round(x0);
    let y = Math.round(y0);
    const tx = Math.round(x1);
    const ty = Math.round(y1);

    const dx = Math.abs(tx - x);
    const dy = Math.abs(ty - y);
    const sx = x < tx ? 1 : -1;
    const sy = y < ty ? 1 : -1;

    let err = dx - dy;

    while (true) {
        drawThickPoint(out, x, y, values, Math.max(1, thickness));

        if (x === tx && y === ty) {
            break;
        }

        const e2 = err << 1;

        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }

        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }

    return out;
}

export function drawRect(
    frame: VisionFrame,
    x: number,
    y: number,
    width: number,
    height: number,
    color: DrawColor = { r: 255, g: 0, b: 0 },
    thickness = 2,
): VisionFrame {
    let out = frame;

    const x0 = Math.round(x);
    const y0 = Math.round(y);
    const x1 = Math.round(x + width - 1);
    const y1 = Math.round(y + height - 1);

    out = drawLine(out, x0, y0, x1, y0, color, thickness);
    out = drawLine(out, x1, y0, x1, y1, color, thickness);
    out = drawLine(out, x1, y1, x0, y1, color, thickness);
    out = drawLine(out, x0, y1, x0, y0, color, thickness);

    return out;
}

export function drawFilledRect(
    frame: VisionFrame,
    x: number,
    y: number,
    width: number,
    height: number,
    color: DrawColor = { r: 255, g: 0, b: 0 },
): VisionFrame {
    const out = frame.clone();
    const values = colorFor(out, color);

    const x0 = Math.max(0, Math.round(x));
    const y0 = Math.max(0, Math.round(y));
    const x1 = Math.min(out.width - 1, Math.round(x + width - 1));
    const y1 = Math.min(out.height - 1, Math.round(y + height - 1));

    for (let yy = y0; yy <= y1; yy++) {
        for (let xx = x0; xx <= x1; xx++) {
            setPixelUnsafe(out, xx, yy, values);
        }
    }

    return out;
}

export function drawCircle(
    frame: VisionFrame,
    cx: number,
    cy: number,
    radius: number,
    color: DrawColor = { r: 255, g: 0, b: 0 },
    thickness = 1,
): VisionFrame {
    const out = frame.clone();
    const values = colorFor(out, color);

    const x0 = Math.round(cx);
    const y0 = Math.round(cy);
    let x = Math.max(0, Math.round(radius));
    let y = 0;
    let err = 0;

    const t = Math.max(1, thickness);

    while (x >= y) {
        drawThickPoint(out, x0 + x, y0 + y, values, t);
        drawThickPoint(out, x0 + y, y0 + x, values, t);
        drawThickPoint(out, x0 - y, y0 + x, values, t);
        drawThickPoint(out, x0 - x, y0 + y, values, t);
        drawThickPoint(out, x0 - x, y0 - y, values, t);
        drawThickPoint(out, x0 - y, y0 - x, values, t);
        drawThickPoint(out, x0 + y, y0 - x, values, t);
        drawThickPoint(out, x0 + x, y0 - y, values, t);

        y++;

        if (err <= 0) {
            err += 2 * y + 1;
        }

        if (err > 0) {
            x--;
            err -= 2 * x + 1;
        }
    }

    return out;
}

export function drawFilledCircle(
    frame: VisionFrame,
    cx: number,
    cy: number,
    radius: number,
    color: DrawColor = { r: 255, g: 0, b: 0 },
): VisionFrame {
    const out = frame.clone();
    const values = colorFor(out, color);

    const x0 = Math.round(cx);
    const y0 = Math.round(cy);
    const r = Math.max(0, Math.round(radius));
    const r2 = r * r;

    const yStart = Math.max(0, y0 - r);
    const yEnd = Math.min(out.height - 1, y0 + r);

    for (let y = yStart; y <= yEnd; y++) {
        const dy = y - y0;
        const dxMax = Math.floor(Math.sqrt(r2 - dy * dy));

        const xStart = Math.max(0, x0 - dxMax);
        const xEnd = Math.min(out.width - 1, x0 + dxMax);

        for (let x = xStart; x <= xEnd; x++) {
            setPixelUnsafe(out, x, y, values);
        }
    }

    return out;
}

export function drawBoxes(
    frame: VisionFrame,
    boxes: Array<Pick<ConnectedComponent, "x" | "y" | "width" | "height">>,
    color: DrawColor = { r: 255, g: 0, b: 0 },
    thickness = 2,
): VisionFrame {
    let out = frame.clone();

    for (const box of boxes) {
        out = drawRect(out, box.x, box.y, box.width, box.height, color, thickness);
    }

    return out;
}