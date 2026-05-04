import * as Kernels from "./kernels/index.js";

export interface VisionImage {
    grayscale(): VisionImage;
    threshold(value?: number): VisionImage;
    invert(): VisionImage;

    sharpen(strength?: number): VisionImage;
    convolve(kernel: number[], kw: number, kh: number): VisionImage;

    crop(x: number, y: number, w: number, h: number): VisionImage;
    flipH(): VisionImage;
    flipV(): VisionImage;
    rotate90(): VisionImage;

    brightnessContrast(brightness?: number, contrast?: number): VisionImage;
    gamma(g: number): VisionImage;
    normalize(): VisionImage;
    equalizeHistogram(): VisionImage;

    erode(radius?: number): VisionImage;
    dilate(radius?: number): VisionImage;
    open(radius?: number): VisionImage;
    close(radius?: number): VisionImage;

    adaptiveThreshold(options?: Kernels.AdaptiveThresholdOptions): VisionImage;

    inRangeGray(min: number, max: number): VisionImage;
    inRangeRGB(range: Kernels.RGBRange): VisionImage;

    drawPoint(
        x: number,
        y: number,
        color?: Kernels.DrawColor,
        thickness?: number,
    ): VisionImage;

    drawLine(
        x0: number,
        y0: number,
        x1: number,
        y1: number,
        color?: Kernels.DrawColor,
        thickness?: number,
    ): VisionImage;

    drawRect(
        x: number,
        y: number,
        width: number,
        height: number,
        color?: Kernels.DrawColor,
        thickness?: number,
    ): VisionImage;

    drawFilledRect(
        x: number,
        y: number,
        width: number,
        height: number,
        color?: Kernels.DrawColor,
    ): VisionImage;

    drawCircle(
        cx: number,
        cy: number,
        radius: number,
        color?: Kernels.DrawColor,
        thickness?: number,
    ): VisionImage;

    drawFilledCircle(
        cx: number,
        cy: number,
        radius: number,
        color?: Kernels.DrawColor,
    ): VisionImage;

    drawBoxes(
        boxes: Array<Pick<Kernels.ConnectedComponent, "x" | "y" | "width" | "height">>,
        color?: Kernels.DrawColor,
        thickness?: number,
    ): VisionImage;

    connectedComponents(
        options?: Kernels.ConnectedComponentsOptions,
    ): Kernels.ConnectedComponent[];

    histogram(): Uint32Array[];
}