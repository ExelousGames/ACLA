import { decodeYolopMask, rgbaToYolopInput } from './yolop-segmentation';

it('uses the trained RGB mean and standard deviation in CHW order', () => {
    const input = rgbaToYolopInput(new Uint8ClampedArray([255, 0, 128, 255, 0, 255, 64, 0]));
    const expected = [(1 - 0.485) / 0.229, -0.485 / 0.229, -0.456 / 0.224,
        (1 - 0.456) / 0.224, (128 / 255 - 0.406) / 0.225, (64 / 255 - 0.406) / 0.225];
    input.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 5));
});

it('selects confident road pixels only when road wins against background', () => {
    const output = { dims: [1, 2, 2, 3], data: new Float32Array([
        0.1, 0.9, 0.1, 0.6, 0.2, 0.1,
        0.9, 0.8, 0.4, 0.6, 0.7, NaN,
    ]) };
    const result = decodeYolopMask(output, 0.5);
    expect(Array.from(result.mask)).toEqual([1, 0, 0, 0, 1, 0]);
    expect(result.confidence).toBeCloseTo(0.8);
    expect(Array.from(decodeYolopMask(output, 0.85).mask)).toEqual([1, 0, 0, 0, 0, 0]);
});

it('returns no confidence for empty masks and rejects malformed output', () => {
    expect(decodeYolopMask({ dims: [1, 2, 1, 2], data: new Float32Array([1, 1, 0, 0]) }, 0.5).confidence).toBe(0);
    expect(() => decodeYolopMask({ dims: [1, 1, 2, 2], data: new Float32Array(4) }, 0.5)).toThrow('drivable-area tensor');
    expect(() => decodeYolopMask({ dims: [1, 2, 1, 2], data: new Float32Array(4) }, NaN)).toThrow('confidence');
});
