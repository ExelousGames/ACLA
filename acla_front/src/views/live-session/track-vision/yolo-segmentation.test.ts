import { letterbox, rgbaToChw } from './yolo-segmentation';

it('letterboxes wide frames and converts RGBA to normalized RGB planes', () => {
    expect(letterbox(1280, 720, 640)).toEqual({ resizedWidth: 640, resizedHeight: 360, padX: 0, padY: 140 });
    expect(Array.from(rgbaToChw(new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 255, 0])))).toEqual([1, 0, 0, 1, 0, 1]);
});
