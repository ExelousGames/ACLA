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

it.each(['track', 'car', 'grass'])('shows only the selected %s label while preserving track cleanup and raw detections', (label) => {
    const pixels = new Uint8ClampedArray(16);
    const layerContext = { createImageData: () => ({ data: pixels }), putImageData: jest.fn() };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(layerContext as any);
    const context = {
        save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), strokeRect: jest.fn(), fillText: jest.fn(),
    } as unknown as CanvasRenderingContext2D;
    const instances = [
        { classId: 0, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array([1, 1, 1, 0]) },
        { classId: 1, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array([1, 0, 0, 0]) },
        { classId: 2, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array([0, 1, 0, 0]) },
    ];
    const original = instances.map((instance) => ({ ...instance, mask: instance.mask.slice() }));
    drawVisionOverlay(context, {
        capturedAt: 1, width: 640, height: 640, detections: { segment: {
            task: 'segment', width: 2, height: 2, inferenceMs: 1, classNames: ['track', 'curb', 'car', 'grass'], instances,
        } },
    }, label);
    const expected = label === 'track'
        ? [0, 0, 0, 0, 55, 239, 172, 110, 55, 239, 172, 110, 0, 0, 0, 0]
        : label === 'car' ? [0, 0, 0, 0, 255, 190, 87, 110, 0, 0, 0, 0, 0, 0, 0, 0] : Array(16).fill(0);
    expect(Array.from(pixels)).toEqual(expected);
    expect(context.strokeRect).toHaveBeenCalledTimes(label === 'grass' ? 0 : 1);
    expect(context.fillText).toHaveBeenCalledTimes(label === 'grass' ? 0 : 1);
    if (label !== 'grass') expect(context.fillText).toHaveBeenCalledWith(`${label} · 90%`, 4, 16);
    expect(instances).toEqual(original);
});

it.each(['car', 'car pack', 'curb', 'grass', 'other', 'fence', 'sand', 'Outfield asphalt road', 'car interior'])
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
            const overlapping = ['car', 'car pack'].includes(label) ? [67, 219, 202, 173] : [55, 239, 172, 110];
            expect(Array.from(pixels)).toEqual([
                ...overlapping, 55, 239, 172, 110, 87, 185, 255, 110, 0, 0, 0, 0,
            ]);
            expect(instances).toEqual(originalOrder);
            expect(track.mask).toEqual(new Uint8Array([1, 0, 1, 0]));
            expect(foreground.mask).toEqual(new Uint8Array([1, 1, 0, 0]));
        }
    }
});

it('shows the cleaned corridor without bonnet flares or track underneath the windshield mask', () => {
    const width = 20, height = 7, pixels = new Uint8ClampedArray(width * height * 4);
    const layerContext = { createImageData: () => ({ data: pixels }), putImageData: jest.fn() };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(layerContext as any);
    const context = { save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), strokeRect: jest.fn(), fillText: jest.fn(),
    } as unknown as CanvasRenderingContext2D;
    const track = { classId: 0, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number],
        mask: Uint8Array.from({ length: width * height }, (_, i) => Number(Math.floor(i / width) >= 3
            || (i % width >= 6 && i % width < 14))) };
    const cockpit = { ...track, classId: 1, mask: Uint8Array.from({ length: width * height }, (_, i) =>
        Number(Math.floor(i / width) === 1 && i % width >= 8 && i % width < 12)) };
    const original = track.mask.slice();
    for (const instances of [[track, cockpit], [cockpit, track]]) {
        pixels.fill(0);
        drawVisionOverlay(context, { capturedAt: 0, width, height, detections: { segment: {
            task: 'segment', width, height, inferenceMs: 1, classNames: ['track', 'car interior'], instances,
        } } });
        for (let i = 0; i < width * height; i++) {
            const expected = cockpit.mask[i] ? [87, 185, 255, 110]
                : Math.floor(i / width) < 3 && track.mask[i] ? [55, 239, 172, 110] : [0, 0, 0, 0];
            expect(Array.from(pixels.slice(i * 4, i * 4 + 4))).toEqual(expected);
        }
        expect(track.mask).toEqual(original);
    }
});
