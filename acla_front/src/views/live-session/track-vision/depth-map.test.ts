import { createDepthMap, depthAtMouse, drawDepthMap } from './depth-map';
import type { TrackVisionFrame } from './track-vision-types';

const frame = (values: number[], width = 640, height = 320): TrackVisionFrame => ({
    capturedAt: 0, width, height, detections: {
        depth: { task: 'depth', width: 4, height: 4, values: Float32Array.from(values), classNames: [], inferenceMs: 1 },
    },
});

afterEach(() => jest.restoreAllMocks());

it('paints full-frame depth without label masks and excludes model padding from its scale', () => {
    const source = frame([1, 1, 1, 1, 10, 20, 30, 40, 50, 60, 70, 250, 999, 999, 999, 999]);
    const original = (source.detections.depth as { values: Float32Array }).values.slice();
    const map = createDepthMap(source)!;
    expect(map).toMatchObject({ near: 10, far: 250, crop: { x: 0, y: 1, width: 4, height: 2 } });
    const pixels = new Uint8ClampedArray(64);
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        createImageData: () => ({ data: pixels }), putImageData: jest.fn(),
    } as any);
    const context = { save: jest.fn(), restore: jest.fn(), fillRect: jest.fn(), drawImage: jest.fn() } as unknown as CanvasRenderingContext2D;
    drawDepthMap(context, map);
    expect(context.drawImage).toHaveBeenCalledWith(expect.any(HTMLCanvasElement), 0, 1, 4, 2, 0, 0, 640, 320);
    expect(Array.from(pixels.slice(4 * 4, 5 * 4))).toEqual([239, 68, 68, 255]);
    expect(Array.from(pixels.slice(11 * 4, 12 * 4))).toEqual([59, 130, 246, 255]);
    for (let i = 4; i < 12; i++) expect(pixels[i * 4 + 3]).toBe(255);
    expect(map.depth.values).toEqual(original);
});

it('samples the exact depth pixel through both model padding and a contained, resized preview', () => {
    const map = createDepthMap(frame(Array.from({ length: 16 }, (_, i) => i + 1)))!;
    const rect = { left: 10, top: 20, width: 400, height: 400 };
    expect(depthAtMouse(map, rect, 10, 120)?.depth).toBe(5);
    expect(depthAtMouse(map, rect, 409, 319)?.depth).toBe(12);
    expect(depthAtMouse(map, rect, 310, 220)?.depth).toBe(12);
    expect(depthAtMouse(map, rect, 210, 119)).toBeNull();
    expect(depthAtMouse(map, rect, 210, 320)).toBeNull();
    expect(depthAtMouse(map, rect, 410, 220)).toBeNull();
    expect(depthAtMouse(map, { ...rect, width: 0 }, 10, 20)).toBeNull();
    const portrait = createDepthMap(frame(Array.from({ length: 16 }, (_, i) => i + 1), 320, 640))!;
    expect(depthAtMouse(portrait, rect, 110, 20)?.depth).toBe(2);
    expect(depthAtMouse(portrait, rect, 309, 419)?.depth).toBe(15);
    expect(depthAtMouse(portrait, rect, 109, 220)).toBeNull();
    expect(depthAtMouse(portrait, rect, 310, 220)).toBeNull();
});

it('keeps missing and invalid depths unavailable and renders constant depth without division by zero', () => {
    const map = createDepthMap(frame([0, -1, NaN, Infinity, 12, 12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0], 640, 640))!;
    const rect = { left: 0, top: 0, width: 640, height: 640 };
    for (const x of [0, 160, 320, 480]) expect(depthAtMouse(map, rect, x, 0)?.depth).toBeNull();
    expect(map).toMatchObject({ near: 12, far: 12 });
    const pixels = new Uint8ClampedArray(64);
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        createImageData: () => ({ data: pixels }), putImageData: jest.fn(),
    } as any);
    const context = { save: jest.fn(), restore: jest.fn(), fillRect: jest.fn(), drawImage: jest.fn() } as unknown as CanvasRenderingContext2D;
    drawDepthMap(context, map);
    expect(Array.from(pixels.slice(0, 16))).toEqual(new Array(16).fill(0));
    expect(Array.from(pixels.slice(16, 20))).toEqual([239, 68, 68, 255]);
    expect(createDepthMap(frame(new Array(16).fill(0)))).toMatchObject({ near: null, far: null });
    expect(createDepthMap(null)).toBeNull();
    expect(createDepthMap({ capturedAt: 0, width: 640, height: 320, detections: {} })).toBeNull();
});
