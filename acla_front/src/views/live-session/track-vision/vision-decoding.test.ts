import { decodeDepth, decodeSegments } from './vision-decoding';

it('keeps same-position objects of different classes separate and suppresses duplicate same-class boxes', () => {
    const size = 32;
    const count = 21;
    const data = new Float32Array(7 * count);
    const put = (index: number, values: number[]) => values.forEach((value, channel) => { data[channel * count + index] = value; });
    put(0, [16, 16, 32, 32, 0.9, 0.1, 1]);
    put(1, [16, 16, 32, 32, 0.1, 0.8, -1]);
    put(2, [16, 16, 32, 32, 0.7, 0.1, 1]);
    const result = decodeSegments({ dims: [1, 7, count], data }, { dims: [1, 1, 2, 2], data: new Float32Array([1, -1, 1, -1]) }, size, 0.5, 2);
    expect(result.instances.map(({ classId }) => classId)).toEqual([0, 1]);
    expect(result.instances[0].mask).toEqual(new Uint8Array([1, 0, 1, 0]));
    expect(result.instances[1].mask).toEqual(new Uint8Array([0, 1, 0, 1]));
    expect(decodeSegments({ dims: [1, 7, count], data }, { dims: [1, 1, 2, 2], data: new Float32Array(4) }, size, 0.95, 2).instances).toHaveLength(0);
});

it('rejects mismatched class counts so output IDs cannot use the wrong labels', () => {
    expect(() => decodeSegments(
        { dims: [1, 7, 21], data: new Float32Array(147) },
        { dims: [1, 1, 2, 2], data: new Float32Array(4) }, 32, 0.5, 3,
    )).toThrow('backend labels');
});

it('owns its depth data after tensors are released and rejects multichannel depth', () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const result = decodeDepth({ dims: [1, 1, 2, 2], data });
    data.fill(0);
    expect(result.values).toEqual(new Float32Array([1, 2, 3, 4]));
    expect(() => decodeDepth({ dims: [1, 2, 1, 2], data })).toThrow('Depth requires');
});
