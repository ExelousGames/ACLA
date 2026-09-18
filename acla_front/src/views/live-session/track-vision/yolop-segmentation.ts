import { FloatTensor, rgbaToChw } from './yolo-segmentation';

/** YOLOP uses RGB ImageNet normalization, unlike Ultralytics segmentation. */
export function rgbaToYolopInput(rgba: Uint8ClampedArray): Float32Array {
    const input = rgbaToChw(rgba);
    const pixels = rgba.length / 4;
    const mean = [0.485, 0.456, 0.406];
    const deviation = [0.229, 0.224, 0.225];
    for (let channel = 0; channel < 3; channel++) {
        for (let i = 0; i < pixels; i++) {
            const index = channel * pixels + i;
            input[index] = (input[index] - mean[channel]) / deviation[channel];
        }
    }
    return input;
}

/** The official drive_area_seg head already applies sigmoid to both channels. */
export function decodeYolopMask(output: FloatTensor, threshold: number) {
    const [batch, channels, height, width] = output.dims;
    if (output.dims.length !== 4 || batch !== 1 || channels !== 2 || height < 1 || width < 1
        || output.data.length !== 2 * height * width) {
        throw new Error('YOLOP must return a [1, 2, height, width] drivable-area tensor.');
    }
    if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) throw new Error('Invalid confidence threshold.');
    const pixels = height * width;
    const mask = new Uint8Array(pixels);
    let total = 0;
    let count = 0;
    for (let i = 0; i < pixels; i++) {
        const background = output.data[i];
        const road = output.data[pixels + i];
        if (Number.isFinite(background) && background >= 0 && background <= 1
            && Number.isFinite(road) && road > background && road >= threshold && road <= 1) {
            mask[i] = 1;
            total += road;
            count++;
        }
    }
    return { mask, width, height, confidence: count ? total / count : 0 };
}
