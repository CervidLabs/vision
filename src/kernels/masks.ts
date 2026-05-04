import { VisionFrame } from "../core/VisionFrame.js";

export interface Range {
    min: number;
    max: number;
}

export interface RGBRange {
    r?: [number, number];
    g?: [number, number];
    b?: [number, number];
}

function inside(v: number, range?: [number, number]): boolean {
    if (!range) {
        return true;
    }

    return v >= range[0] && v <= range[1];
}

export function inRangeGray(
    frame: VisionFrame,
    min: number,
    max: number,
): VisionFrame {
    if (frame.channels !== 1) {
        throw new Error("inRangeGray: expected 1-channel frame");
    }

    const out = new VisionFrame(frame.width, frame.height, 1);
    const src = frame.data;
    const dst = out.data;

    for (let i = 0; i < src.length; i++) {
        const v = src[i];
        dst[i] = v >= min && v <= max ? 255 : 0;
    }

    return out;
}

export function inRangeRGB(frame: VisionFrame, range: RGBRange): VisionFrame {
    if (frame.channels !== 3 && frame.channels !== 4) {
        throw new Error("inRangeRGB: expected 3 or 4-channel RGB/RGBA frame");
    }

    const { width, height, channels } = frame;
    const out = new VisionFrame(width, height, 1);
    const src = frame.data;
    const dst = out.data;

    for (let i = 0, p = 0; i < width * height; i++, p += channels) {
        const r = src[p];
        const g = src[p + 1];
        const b = src[p + 2];

        dst[i] =
            inside(r, range.r) &&
                inside(g, range.g) &&
                inside(b, range.b)
                ? 255
                : 0;
    }

    return out;
}