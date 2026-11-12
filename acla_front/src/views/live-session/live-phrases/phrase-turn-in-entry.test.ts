import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { PHRASE_DEFINITIONS, PhraseEngine } from './phrase-engine';
import { createPhraseMapContext } from './phrase-map-context';
import { referenceDirectionChanges } from './phrase-reference-guidance';
import { getPhraseSpeechInstances } from './phrase-speech';
import { steeringCornerFixture } from './test-fixtures';

const frame = (position: number): LiveTelemetryFrameEvent => {
    const sample: StandardTelemetrySample = {
        Static_track: 'test', Static_car_model: 'car', Graphics_normalized_car_position: position,
        Physics_speed_kmh: 36, Physics_brake: 0, Physics_gas: 0,
    };
    return {
        type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
        committedSampleCount: 0,
        update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 },
    };
};

function multiArcCorner() {
    const { map, reference } = steeringCornerFixture();
    let x = 0, y = 0;
    reference.y = reference.x.map((_, index) => {
        const row = [x, y, 0, 0, 0, 0, 36];
        // Four separate same-direction arcs belong to one mapped corner; a fifth belongs to the next.
        const angle = [10, 22, 40, 60, 80].reduce((sum, start) =>
            sum + Math.max(0, Math.min(5, index - start)), 0) * 6 * Math.PI / 180;
        x += 10 * Math.cos(angle);
        y += 10 * Math.sin(angle);
        return row;
    });
    map.samples = { middle_line: reference.y.map((row, index) => ({
        bin: index, normalized_position: reference.x[index], x: row[0], y: row[1], z: 0,
        sample_count: 1, updated_at: '',
    })) };
    map.centerline_segments = [
        { id: 'turn-1', tags: ['corner'], start_position: 0.1, end_position: 0.7 },
        { id: 'turn-2', tags: ['corner'], start_position: 0.8, end_position: 0.9 },
    ];
    return { map, reference };
}

function setup(fixture = multiArcCorner()) {
    const phrases = PHRASE_DEFINITIONS.filter(({ id }) => id === 'reference-steering');
    const engine = new PhraseEngine(phrases);
    engine.setDrivingAggression(100, 0);
    engine.setSpeechDurations(new Map(getPhraseSpeechInstances(phrases).map((instance) => [instance.cacheKey, 200])));
    engine.receiveMap(fixture.map, 0, fixture.reference);
    const at = (position: number, now: number) => engine.receiveTelemetry(frame(position), now);
    return { engine, at };
}

it('announces only the entry arc when a mapped corner contains several detected turns', () => {
    const fixture = multiArcCorner();
    const context = createPhraseMapContext(fixture.map);
    expect(referenceDirectionChanges(fixture.reference, [context.cornerBounds[0]])).toHaveLength(4);
    expect(context.at(0.24).phase).toBe('entry');
    const { engine, at } = setup(fixture);
    const entry = at(0.12, 1000);
    expect(entry.events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    engine.speechFinished(entry.events[0].id, 1200);
    at(0.18, 2000); // The first detected arc has ended, but the mapped corner has not.
    for (const [position, now] of [[0.24, 3000], [0.42, 4000], [0.62, 5000]]) {
        expect(at(position, now).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    }
});

it.each([0.24, 0.42, 0.62])('does not announce a later arc on first telemetry at %s within a mapped corner', (position) => {
    const { at } = setup();
    expect(at(position, 1000).events).toEqual([]);
});

it.each([0.35, 0.39])('does not announce Turn in after entry even while the first reference arc is active at %s', (position) => {
    const { at } = setup(steeringCornerFixture());
    expect(at(position, 1000).events).toEqual([]);
});

it('gives the next mapped corner its own entry announcement', () => {
    const { engine, at } = setup();
    const entry = at(0.12, 1000);
    engine.speechFinished(entry.events[0].id, 1200);
    at(0.42, 2000);
    at(0.72, 3000);
    expect(at(0.82, 4000).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Turn in']);
});

it('retains the announcement through an arc gap until measured progress leaves the mapped corner', () => {
    const { engine, at } = setup();
    const entry = at(0.12, 1000);
    engine.speechFinished(entry.events[0].id, 1200);
    at(0.18, 2000);
    expect(at(0.12, 3000).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    at(0.42, 4000);
    at(0.72, 5000);
    expect(at(0.12, 6000).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Turn in']);
});

it('finds entry within the speech forecast when its endpoint has reached the corner middle', () => {
    const fixture = steeringCornerFixture();
    const context = createPhraseMapContext(fixture.map);
    expect(context.at(0.332).phase).toBe('entry');
    expect(context.at(0.334).phase).toBe('middle');
    const { at } = setup(fixture);
    expect(at(0.332, 1000).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    expect(setup(fixture).at(0.334, 1000).events).toEqual([]);
});

it('keeps a multi-arc corner consumed across start/finish until its mapped exit', () => {
    const fixture = multiArcCorner();
    const shift = (position: number) => (position + 0.7) % 1;
    const rows = fixture.reference.x.map((position, index) => ({ position: shift(position), row: fixture.reference.y[index] }))
        .sort((a, b) => a.position - b.position);
    fixture.reference.x = rows.map(({ position }) => position);
    fixture.reference.y = rows.map(({ row }) => row);
    fixture.map.samples.middle_line = fixture.map.samples.middle_line!.map((point) => ({
        ...point, normalized_position: shift(point.normalized_position),
    })).sort((a, b) => a.normalized_position - b.normalized_position);
    fixture.map.centerline_segments = fixture.map.centerline_segments!.map((corner) => ({
        ...corner, start_position: shift(corner.start_position), end_position: shift(corner.end_position),
    }));
    expect(setup(fixture).at(0.94, 1000).events).toEqual([]);
    const { engine, at } = setup(fixture);
    const entry = at(0.82, 1000);
    expect(entry.events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    engine.speechFinished(entry.events[0].id, 1200);
    for (const [position, now] of [[0.94, 2000], [0.12, 3000], [0.32, 4000]]) {
        expect(at(position, now).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    }
    at(0.42, 5000);
    expect(at(0.82, 6000).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Turn in']);
});
