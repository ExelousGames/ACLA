import { capturePolicyCameraImage, POLICY_CAMERA_INPUTS } from './policy-camera';

afterEach(() => jest.restoreAllMocks());

it('copies spatial RGB pixels from the clean capture at the policy resolution', () => {
    const rgba = new Uint8ClampedArray(384 * 216 * 4);
    rgba.set([255, 128, 0, 255, 0, 64, 255, 255]);
    rgba.set([32, 16, 8, 255], rgba.length - 4);
    const drawImage = jest.fn();
    const getImageData = jest.fn(() => ({ data: rgba }));
    const fillRect = jest.fn();
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage, getImageData, fillRect } as any);
    const source = document.createElement('canvas');
    source.width = 1920;
    source.height = 1080;
    const image = capturePolicyCameraImage(source);
    expect(fillRect).toHaveBeenCalledWith(0, 0, 384, 216);
    expect(drawImage).toHaveBeenCalledWith(source, 0, 0, 384, 216);
    expect(getImageData).toHaveBeenCalledWith(0, 0, 384, 216);
    expect(image).toMatchObject({ width: 384, height: 216 });
    expect(image.rgb).toHaveLength(POLICY_CAMERA_INPUTS);
    expect(Array.from(image.rgb.slice(0, 6))).toEqual([255, 128, 0, 0, 64, 255]);
    expect(Array.from(image.rgb.slice(-3))).toEqual([32, 16, 8]);
    rgba.fill(0);
    expect(Array.from(image.rgb.slice(0, 3))).toEqual([255, 128, 0]);
});

it.each([
    [1440, 1080, 48, 0, 288, 216],
    [2560, 1080, 0, 27, 384, 162],
    [1080, 1920, 131.25, 0, 121.5, 216],
])('preserves the entire %s × %s capture with centered black letterboxing', (width, height, x, y, w, h) => {
    const context = { drawImage: jest.fn(), fillRect: jest.fn(), fillStyle: '',
        getImageData: jest.fn(() => ({ data: new Uint8ClampedArray(384 * 216 * 4) })) };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as any);
    const source = document.createElement('canvas'); source.width = width; source.height = height;
    capturePolicyCameraImage(source);
    expect(context.fillStyle).toBe('#000');
    expect(context.fillRect).toHaveBeenCalledWith(0, 0, 384, 216);
    expect(context.drawImage).toHaveBeenCalledWith(source, x, y, w, h);
});
