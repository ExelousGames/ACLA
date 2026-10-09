import { DEFAULT_CAMERA } from './camera-projection';
import { readTrackVisionSettings, saveTrackVisionSettings, TRACK_VISION_SETTINGS_KEY } from './track-vision-settings';

beforeEach(() => window.localStorage.clear());
afterEach(() => jest.restoreAllMocks());

it.each(['{broken', 'null', '[]', '{"version":2}'])('uses defaults for invalid local settings: %s', (raw) => {
    const defaults = readTrackVisionSettings();
    window.localStorage.setItem(TRACK_VISION_SETTINGS_KEY, raw);
    expect(readTrackVisionSettings()).toEqual(defaults);
});

it('rejects invalid saved values without losing the other settings', () => {
    const defaults = readTrackVisionSettings();
    window.localStorage.setItem(TRACK_VISION_SETTINGS_KEY, JSON.stringify({
        version: 1, inputSizes: { segment: 17, depth: 392 }, confidence: 1, filterConfidence: 0.8,
        displayLabel: 42, cameraDraft: { ...DEFAULT_CAMERA, heightM: null },
        calibration: { ...DEFAULT_CAMERA, imageWidth: 0, imageHeight: 720 }, showCalibrationOnCapture: 'true',
    }));
    expect(readTrackVisionSettings()).toEqual({
        ...defaults, inputSizes: { ...defaults.inputSizes, depth: 392 }, filterConfidence: 0.8, calibration: undefined,
    });
});

it('keeps settings usable when local storage is unavailable', () => {
    const defaults = readTrackVisionSettings();
    jest.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('Storage blocked'); });
    jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('Storage full'); });
    jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(readTrackVisionSettings()).toEqual(defaults);
    expect(() => saveTrackVisionSettings({ ...defaults, confidence: 0.7 })).not.toThrow();
    expect(console.warn).toHaveBeenCalledWith('Unable to save Track Vision settings locally', expect.any(Error));
});
