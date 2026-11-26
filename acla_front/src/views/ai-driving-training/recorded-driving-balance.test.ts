import { cloneConfig, emptyRecordedSamplingWeights } from './driving-model-config';
import { RecordedDrivingBalance } from './recorded-driving-balance';
import { trainingFixture } from './training-fixtures';

function setup() {
    const base = trainingFixture().observation;
    const config = cloneConfig();
    config.recordedSamplingWeights = { ...emptyRecordedSamplingWeights(), Unlabeled: 30, 'Shift down': 30, 'Shift up': 15, Hairpin: 25 };
    const balance = new RecordedDrivingBalance(config);
    const count = (at: number, { label, startsRange, steering = 0, gear = 3 }:
        { label?: string; startsRange?: boolean; steering?: number; gear?: number } = {}) => {
        balance.count({ ...base, capturedAt: at, measuredAt: at, gear, controls: { throttle: 0.4, steering, brake: 0 } },
            label, startsRange);
    };
    return { balance, config, count };
}

it('uses only the origin annotation to categorize examples, even during gear changes and steering', () => {
    const { balance, count } = setup();
    count(1000, { label: ' Hairpin ' });
    count(1100, { label: 'SHIFT UP' });
    count(1200, { label: 'Shift down' });
    count(1300, { steering: 0.3, gear: 4 });
    count(1400, { label: 'Race pace', steering: -0.3, gear: 4 });
    count(1500, { label: 'Shifting', gear: 5 });
    count(1600, { label: 'Cornering', gear: 5 });
    count(1700, { gear: 5 });

    expect(balance.prepare()).toEqual({ eligible: { ...emptyRecordedSamplingWeights(), Unlabeled: 4, 'Shift down': 1,
        'Shift up': 1, Hairpin: 1 } });
    expect(balance.weights).toEqual({ ...emptyRecordedSamplingWeights(), Unlabeled: 30, 'Shift down': 30,
        'Shift up': 15, Hairpin: 25 });
    expect(() => balance.select('Shift down')).toThrow('The aligned samples changed');
    expect(balance.select('Hairpin')).toBe('Hairpin');
    expect(balance.select('shift up')).toBe('Shift up');
    expect(balance.select('Shift down')).toBe('Shift down');
    expect(balance.select()).toBe('Unlabeled');
    expect(balance.select('Race pace')).toBe('Unlabeled');
    expect(balance.select('Shifting')).toBe('Unlabeled');
    expect(balance.select('Cornering')).toBe('Unlabeled');
    expect(balance.complete).toBe(true);
});

it.each(['clip', 'gap', 'invalid'])('does not count origins across a %s boundary', (boundary) => {
    const { balance, count } = setup();
    count(1000);
    count(1100);
    if (boundary === 'invalid') balance.count(undefined);
    const at = boundary === 'gap' ? 1500 : 1200;
    count(at, { startsRange: boundary === 'clip', gear: 4 });
    count(at + 100, { gear: 4 });
    expect(balance.prepare()).toEqual({ eligible: { ...emptyRecordedSamplingWeights(), Unlabeled: 2 } });
    expect(balance.weights).toEqual({ ...emptyRecordedSamplingWeights(), Unlabeled: 30 });
    expect(balance.select()).toBe('Unlabeled');
    expect(balance.select()).toBe('Unlabeled');
    expect(balance.complete).toBe(true);
});

it('explains when none of the available categories has a positive weight', () => {
    const { balance, config, count } = setup();
    config.recordedSamplingWeights = { ...emptyRecordedSamplingWeights(), Hairpin: 100 };
    count(1000);
    count(1100, { steering: 0.3 });
    expect(() => balance.prepare()).toThrow('Adjust the sampling weights or select different annotations.');
});
