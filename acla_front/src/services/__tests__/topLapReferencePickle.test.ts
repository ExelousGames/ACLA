import { decodeTopLapPickle } from '../topLapReferencePickle';
import fixtures from './fixtures/top-lap-reference.json';

it.each(['protocol4C', 'protocol4F', 'protocol5C', 'protocol5F', 'bigEndian'] as const)(
    'reads actual Python/NumPy %s model data', (format) => {
        expect(decodeTopLapPickle(fixtures[format])).toEqual(fixtures.expected);
    },
);

it('reads legacy ACC entries and empty feature matrices', () => {
    const { game, ...legacy } = fixtures.expected;
    expect(decodeTopLapPickle(fixtures.legacy)).toEqual(legacy);
    expect(decodeTopLapPickle(fixtures.noCoordinates)).toMatchObject({ y: [[], [], []], target_features: [] });
});

it('rejects executable globals and incomplete pickles', () => {
    expect(() => decodeTopLapPickle(fixtures.forbiddenGlobal)).toThrow();
    expect(() => decodeTopLapPickle(fixtures.protocol5C.slice(0, -8))).toThrow();
    expect(() => decodeTopLapPickle(btoa('\x80\x05\x68\xff.'))).toThrow();
});
