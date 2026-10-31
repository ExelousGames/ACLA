import { capturePolicyCameraImage, POLICY_CAMERA_INPUTS } from './policy-camera';

afterEach(() => jest.restoreAllMocks());

it('copies spatial RGB pixels from the clean capture at the policy resolution', () => {
    const rgba = new Uint8ClampedArray(768 * 768 * 4);
    rgba.set([255, 128, 0, 255, 0, 64, 255, 255]);
    rgba.set([32, 16, 8, 255], rgba.length - 4);
    const drawImage = jest.fn();
    const getImageData = jest.fn(() => ({ data: rgba }));
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage, getImageData } as any);
    const source = document.createElement('canvas');
    source.width = 1920;
    source.height = 1080;
    const image = capturePolicyCameraImage(source);
    expect(drawImage).toHaveBeenCalledWith(source, 0, 0, 768, 768);
    expect(getImageData).toHaveBeenCalledWith(0, 0, 768, 768);
    expect(image).toMatchObject({ width: 768, height: 768 });
    expect(image.rgb).toHaveLength(POLICY_CAMERA_INPUTS);
    expect(Array.from(image.rgb.slice(0, 6))).toEqual([255, 128, 0, 0, 64, 255]);
    expect(Array.from(image.rgb.slice(-3))).toEqual([32, 16, 8]);
    rgba.fill(0);
    expect(Array.from(image.rgb.slice(0, 3))).toEqual([255, 128, 0]);
});
