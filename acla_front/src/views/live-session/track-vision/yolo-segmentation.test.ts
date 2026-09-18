import { decodeTrackMask, letterbox, rgbaToChw, traceTrackBoundaries } from './yolo-segmentation';

function fixture(classes = 1) {
    const count = 21; // 32px input, strides 8/16/32.
    const predictions = { dims: [1, 5 + classes, count], data: new Float32Array((5 + classes) * count) };
    const prototypes = { dims: [1, 1, 4, 4], data: new Float32Array(16).fill(1) };
    const box = (index: number, values: number[]) => values.forEach((value, channel) => { predictions.data[channel * count + index] = value; });
    return { predictions, prototypes, box };
}

it('letterboxes wide frames and converts RGBA to normalized RGB planes', () => {
    expect(letterbox(1280, 720, 640)).toEqual({ resizedWidth: 640, resizedHeight: 360, padX: 0, padY: 140 });
    expect(Array.from(rgbaToChw(new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 255, 0])))).toEqual([1, 0, 0, 1, 0, 1]);
});

it('reconstructs a selected mask and crops it to its detection box', () => {
    const { predictions, prototypes, box } = fixture();
    box(0, [16, 16, 32, 16, 0.9, 1]);
    const result = decodeTrackMask(predictions, prototypes, 32, 0, 0.5);
    expect(Array.from(result.mask)).toEqual([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]);
    expect(result.confidence).toBeCloseTo(0.9);
});

it('filters confidence and competing classes before decoding', () => {
    const { predictions, prototypes, box } = fixture(2);
    box(0, [16, 16, 32, 32, 0.8, 0.9, 1]);
    box(1, [16, 16, 32, 32, 0.3, 0.1, 1]);
    expect(decodeTrackMask(predictions, prototypes, 32, 0, 0.5).mask.some(Boolean)).toBe(false);
    expect(decodeTrackMask(predictions, prototypes, 32, 1, 0.5).mask.every(Boolean)).toBe(true);
});

it('suppresses duplicate boxes and does not report a confidence for an empty mask', () => {
    const { predictions, prototypes, box } = fixture();
    box(0, [16, 16, 32, 32, 0.9, -1]);
    box(1, [16, 16, 32, 32, 0.8, 1]);
    expect(decodeTrackMask(predictions, prototypes, 32, 0, 0.5).confidence).toBe(0);
});

it('rejects unsupported output layouts and invalid track classes', () => {
    const { predictions, prototypes } = fixture();
    expect(() => decodeTrackMask({ ...predictions, dims: [1, 300, 38] }, prototypes, 32, 0, 0.5)).toThrow('Unsupported model outputs');
    expect(() => decodeTrackMask(predictions, prototypes, 32, 1, 0.5)).toThrow('Track class');
    expect(() => decodeTrackMask(predictions, prototypes, 32, NaN, 0.5)).toThrow('Track class');
});

it('removes letterbox padding and maps boundaries back to the captured frame', () => {
    const mask = new Uint8Array(64);
    for (let y = 0; y < 8; y++) for (let x = 1; x < 7; x++) mask[y * 8 + x] = 1;
    const [segment] = traceTrackBoundaries(mask, 8, 8, 64, 32, 32);
    expect(segment.left).toHaveLength(4);
    expect(segment.left[0]).toEqual({ x: 0.125, y: 0.125 });
    expect(segment.right[3]).toEqual({ x: 0.875, y: 0.875 });
});

it('keeps missing rows as gaps and ignores isolated one-pixel noise', () => {
    const mask = new Uint8Array(64);
    [1, 2, 5, 6].forEach((y) => { mask.fill(1, y * 8 + 2, y * 8 + 6); });
    mask[3 * 8] = 1;
    expect(traceTrackBoundaries(mask, 8, 8, 32, 32, 32)).toHaveLength(2);
    expect(traceTrackBoundaries(new Uint8Array(64), 8, 8, 32, 32, 32)).toEqual([]);
});
