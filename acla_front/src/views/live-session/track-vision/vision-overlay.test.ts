import { drawVisionOverlay } from './vision-overlay';
import { TrackVisionFrame } from './track-vision-types';

afterEach(() => { jest.restoreAllMocks(); });

it('does not paint depth data over the captured frame', () => {
    const getContext = jest.spyOn(HTMLCanvasElement.prototype, 'getContext');
    const context = { save: jest.fn(), restore: jest.fn(), drawImage: jest.fn() } as unknown as CanvasRenderingContext2D;
    const values = new Float32Array([1, 10, 30, 60]);
    const result: TrackVisionFrame = {
        capturedAt: 1, width: 1280, height: 720,
        detections: { depth: { task: 'depth', width: 2, height: 2, values, inferenceMs: 1, classNames: [] } },
    };
    drawVisionOverlay(context, result);
    expect(getContext).not.toHaveBeenCalled();
    expect(context.drawImage).not.toHaveBeenCalled();
    expect(context.save).not.toHaveBeenCalled();
    expect(Array.from(values)).toEqual([1, 10, 30, 60]);
});

it.each([false, true])('draws segmentation labels and masks with letterbox padding removed (depth enabled: %s)', (withDepth) => {
    const pixels = new Uint8ClampedArray(16);
    const layerContext = { createImageData: () => ({ data: pixels }), putImageData: jest.fn() };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(layerContext as any);
    const context = {
        save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), strokeRect: jest.fn(), fillText: jest.fn(),
    } as unknown as CanvasRenderingContext2D;
    const result: TrackVisionFrame = {
        capturedAt: 1, width: 1280, height: 720, detections: { segment: {
            task: 'segment', width: 2, height: 2, inferenceMs: 1, classNames: ['curb', 'track surface'],
            instances: [{ classId: 1, confidence: 0.9, box: [0, 0, 1, 1], mask: new Uint8Array([1, 0, 1, 0]) }],
        } },
    };
    if (withDepth) result.detections.depth = {
        task: 'depth', width: 2, height: 2, values: new Float32Array([1, 10, 30, 60]), inferenceMs: 1, classNames: [],
    };
    drawVisionOverlay(context, result);
    expect(context.drawImage).toHaveBeenCalledTimes(1);
    expect(context.drawImage).toHaveBeenCalledWith(expect.any(HTMLCanvasElement), 0, 0.4375, 2, 1.125, 0, 0, 1280, 720);
    expect(context.fillText).toHaveBeenCalledWith('track surface · 90%', 4, 16);
    expect(context.strokeRect).toHaveBeenCalledWith(0, 0, 1280, 720);
    expect(Array.from(pixels.slice(0, 4))).toEqual([87, 185, 255, 110]);
    expect(Array.from(pixels.slice(4, 8))).toEqual([0, 0, 0, 0]);
    jest.restoreAllMocks();
});

it.each(['car', 'car pack', 'curb', 'grass', 'other', 'fence', 'sand', 'Outfield asphalt road'])
('keeps %s visible where it overlaps a track mask, regardless of detection order', (label) => {
    const pixels = new Uint8ClampedArray(16);
    const layerContext = { createImageData: () => ({ data: pixels }), putImageData: jest.fn() };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(layerContext as any);
    const context = {
        save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), strokeRect: jest.fn(), fillText: jest.fn(),
    } as unknown as CanvasRenderingContext2D;
    for (const trackLabel of ['track', ' ROAD ', '\tAsPhAlT ', 'tarmac']) {
        for (const trackFirst of [true, false]) {
            pixels.fill(0);
            const foreground = { classId: 0, confidence: 0.95, box: [0, 0, 1, 1] as [number, number, number, number],
                mask: new Uint8Array([1, 1, 0, 0]) };
            const track = { ...foreground, classId: 1, confidence: 0.8, mask: new Uint8Array([1, 0, 1, 0]) };
            const instances = trackFirst ? [track, foreground] : [foreground, track];
            const originalOrder = instances.slice();
            drawVisionOverlay(context, {
                capturedAt: 1, width: 640, height: 640, detections: { segment: {
                    task: 'segment', width: 2, height: 2, inferenceMs: 1, classNames: [label, trackLabel], instances,
                } },
            });
            expect(Array.from(pixels)).toEqual([
                67, 219, 202, 173, 55, 239, 172, 110, 87, 185, 255, 110, 0, 0, 0, 0,
            ]);
            expect(instances).toEqual(originalOrder);
            expect(track.mask).toEqual(new Uint8Array([1, 0, 1, 0]));
            expect(foreground.mask).toEqual(new Uint8Array([1, 1, 0, 0]));
        }
    }
});
