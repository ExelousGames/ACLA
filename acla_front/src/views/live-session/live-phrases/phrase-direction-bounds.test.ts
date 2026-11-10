import { createPhraseMapContext } from './phrase-map-context';
import { copyPhraseReference } from './phrase-driving-reference';
import { referenceDirectionChanges, referenceRangeSegments } from './phrase-reference-guidance';
import { steeringCornerFixture, variableBrakingReference } from './test-fixtures';

it.each(['acc', 'iracing'] as const)('excludes entry positioning and exit adjustments from shared ranges in %s', (game) => {
    const { map, reference } = steeringCornerFixture(game);
    map.centerline_segments![0].start_position = 0.275;
    map.centerline_segments![0].end_position = 0.355;
    const corners = createPhraseMapContext(map).cornerBounds;
    const ranges = referenceDirectionChanges(reference, corners);
    expect(ranges).toHaveLength(1);
    expect(ranges[0].start).toBeCloseTo(275);
    expect(ranges[0].end).toBeCloseTo(355);
    const segments = referenceRangeSegments(reference, corners)!.segments.filter(({ changesDirection }) => changesDirection);
    const first = reference.y[27], next = reference.y[28], last = reference.y[35], after = reference.y[36];
    const ground = game === 'acc' ? 1 : 2;
    expect(segments[0].start.x).toBeCloseTo((first[0] + next[0]) / 2);
    expect(segments[0].start.y).toBeCloseTo((first[ground] + next[ground]) / 2);
    expect(segments[segments.length - 1].end.x).toBeCloseTo((last[0] + after[0]) / 2);
    expect(segments[segments.length - 1].end.y).toBeCloseTo((last[ground] + after[ground]) / 2);
});

it.each([false, true])('keeps a bounded corner continuous across start/finish (duplicate: %s)', (duplicate) => {
    const lap = variableBrakingReference();
    if (duplicate) { lap.x.push(1); lap.y.push([...lap.y[0]]); }
    const length = referenceDirectionChanges(lap)[0].end;
    const ranges = referenceDirectionChanges(lap, [{ start: 0.975, end: 0.025 }]);
    expect(ranges).toHaveLength(1);
    expect(ranges[0].start).toBeCloseTo(length * 0.975);
    expect(ranges[0].end).toBeCloseTo(length * 1.025);
    expect(referenceDirectionChanges(lap, [{ start: 0, end: 1 }])).toEqual(referenceDirectionChanges(lap));
});

it('separates neighboring corners even when the reference keeps turning in the same direction', () => {
    const { reference } = steeringCornerFixture();
    const ranges = referenceDirectionChanges(reference, [{ start: 0.27, end: 0.3 }, { start: 0.32, end: 0.35 }]);
    expect(ranges).toHaveLength(2);
    expect(ranges[0].start).toBeCloseTo(270);
    expect(ranges[0].end).toBeCloseTo(300);
    expect(ranges[1].start).toBeCloseTo(320);
    expect(ranges[1].end).toBeCloseTo(350);
});

it('does not use turns outside mapped corners to adjust the braking reference', () => {
    const lap = variableBrakingReference();
    const corners = [{ start: 0.09, end: 0.095 }]; // Beyond this braking sequence's minimum.
    const bounded = copyPhraseReference(lap, 100, corners)!;
    expect(bounded).toEqual(lap);
    expect(copyPhraseReference(lap, 100)).not.toEqual(lap);
    expect(referenceDirectionChanges(lap, [])).toEqual([]);
    expect(referenceRangeSegments(lap, [])!.segments.every(({ changesDirection }) => !changesDirection)).toBe(true);
});
