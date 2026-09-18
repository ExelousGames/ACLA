import { drawVisionOverlay } from './vision-overlay';
import { DepthRange, TrackVisionDetection } from './track-vision-types';

const renderDepth = (values: number[], range: DepthRange) => {
    const pixels = new Uint8ClampedArray(values.length * 4);
    const layerContext = { createImageData: () => ({ data: pixels }), putImageData: jest.fn() };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(layerContext as any);
    const context = { save: jest.fn(), restore: jest.fn(), drawImage: jest.fn() } as unknown as CanvasRenderingContext2D;
    const result: TrackVisionDetection = {
        capturedAt: 1, width: values.length, height: 1,
        detections: { depth: { task: 'depth', width: values.length, height: 1, values: new Float32Array(values), inferenceMs: 1 } },
    };
    drawVisionOverlay(context, result, range);
    return Array.from({ length: values.length }, (_, pixel) => Array.from(pixels.slice(pixel * 4, pixel * 4 + 4)));
};

afterEach(() => { jest.restoreAllMocks(); });

it('clamps close and far colors to the selected distances and blends between them', () => {
    expect(renderDepth([1, 5, 15, 25, 100], { near: 5, far: 25 })).toEqual([
        [255, 80, 0, 100], [255, 80, 0, 100], [128, 80, 128, 100], [0, 80, 255, 100], [0, 80, 255, 100],
    ]);
});

it('keeps the same distance the same color across frames with different extremes', () => {
    const range = { near: 5, far: 25 };
    expect(renderDepth([1, 15, 100], range)[1]).toEqual(renderDepth([14, 15, 16], range)[1]);
    expect(renderDepth([15], range)[0]).not.toEqual(renderDepth([15], { near: 15, far: 25 })[0]);
});

it('leaves invalid depths transparent and handles a zero close cutoff and the smallest range', () => {
    expect(renderDepth([NaN, Infinity, -1, 0, 0.25, 0.5], { near: 0, far: 0.5 })).toEqual([
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [128, 80, 128, 100], [0, 80, 255, 100],
    ]);
});
