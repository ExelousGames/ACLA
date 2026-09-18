import { decodeDepth, decodeSegments, decodeSemantic } from './vision-decoding';

it('decodes Ultralytics baked semantic maps, multiclass logits, and binary logits', () => {
    expect(decodeSemantic({ dims: [1, 2, 2], data: new Uint8Array([0, 7, 2, 18]) }).classes).toEqual(new Uint16Array([0, 7, 2, 18]));
    expect(decodeSemantic({ dims: [1, 2, 1, 2], data: new Float32Array([2, -1, 1, 3]) }).classes).toEqual(new Uint16Array([0, 1]));
    expect(decodeSemantic({ dims: [1, 1, 1, 2], data: new Float32Array([-1, 3]) }).classes).toEqual(new Uint16Array([0, 1]));
    expect(() => decodeSemantic({ dims: [1, 1, 1], data: new Float32Array([-1]) })).toThrow('Invalid semantic class');
});

it('owns its depth data after tensors are released and rejects multichannel depth', () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const result = decodeDepth({ dims: [1, 1, 2, 2], data });
    data.fill(0);
    expect(result.values).toEqual(new Float32Array([1, 2, 3, 4]));
    expect(() => decodeDepth({ dims: [1, 2, 1, 2], data })).toThrow('Depth requires');
});

it('keeps same-position objects of different classes separate and suppresses duplicate same-class boxes', () => {
    const size = 32;
    const count = 21;
    const data = new Float32Array(7 * count);
    const put = (index: number, values: number[]) => values.forEach((value, channel) => { data[channel * count + index] = value; });
    put(0, [16, 16, 32, 32, 0.9, 0.1, 1]);
    put(1, [16, 16, 32, 32, 0.1, 0.8, -1]);
    put(2, [16, 16, 32, 32, 0.7, 0.1, 1]);
    const result = decodeSegments({ dims: [1, 7, count], data }, { dims: [1, 1, 2, 2], data: new Float32Array([1, -1, 1, -1]) }, size, 0.5);
    expect(result.instances.map(({ classId }) => classId)).toEqual([0, 1]);
    expect(result.instances[0].mask).toEqual(new Uint8Array([1, 0, 1, 0]));
    expect(result.instances[1].mask).toEqual(new Uint8Array([0, 1, 0, 1]));
    expect(decodeSegments({ dims: [1, 7, count], data }, { dims: [1, 1, 2, 2], data: new Float32Array(4) }, size, 0.95).instances).toHaveLength(0);
});
