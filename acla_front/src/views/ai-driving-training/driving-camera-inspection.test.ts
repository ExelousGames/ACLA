import { cameraChannelStats, DrivingCameraLayer } from './driving-camera-inspection';

const layer: DrivingCameraLayer = { index: 0, width: 2, height: 2, channels: 2,
    values: new Float32Array([1, -4, 3, -1, 2, -3, 6, -2]) };

it('scores the selected channel in HWC order using raw activations', () => {
    expect(cameraChannelStats(layer, 0)).toEqual({ mean: 3, peak: 6, peakIndex: 3 });
    expect(cameraChannelStats(layer, 1)).toEqual({ mean: -2.5, peak: -1, peakIndex: 1 });
});

it('keeps raw scores comparable when the same pattern has a stronger response in another frame', () => {
    const stronger = { ...layer, values: Float32Array.from(layer.values, (value) => value * 2) };
    expect(cameraChannelStats(stronger, 0).peak).toBe(2 * cameraChannelStats(layer, 0).peak);
    expect(cameraChannelStats(stronger, 0).mean).toBe(2 * cameraChannelStats(layer, 0).mean);
});

it('rejects invalid channels, malformed maps and non-finite activations', () => {
    for (const channel of [-1, 2, 0.5, NaN]) expect(() => cameraChannelStats(layer, channel)).toThrow(/valid camera channel/);
    expect(() => cameraChannelStats({ ...layer, values: new Float32Array(3) }, 0)).toThrow(/shape/);
    expect(() => cameraChannelStats({ ...layer, values: new Float32Array([NaN, 0, 0, 0, 0, 0, 0, 0]) }, 0)).toThrow(/invalid activations/);
});
