import type { StandardTelemetrySample } from '../live-session-types';
import { Action, Closure } from './closure';
import { PhraseEngine, type PhraseContext } from './phrase-engine';
import { PhraseLookahead } from './phrase-lookahead';
import { referencePositionAhead } from './phrase-reference-guidance';
import { steeringCornerFixture } from './test-fixtures';

const frame = (sample: StandardTelemetrySample) => ({
    type: 'frame' as const, sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
    committedSampleCount: 0,
    update: { type: 'frame' as const, game: 'acc' as const, sample, sequence: 1, committedCount: 0, committedSequence: 0 },
});

it('retains every 100 ms prediction through three seconds with matching motion and pedal state', () => {
    const predictor = new PhraseLookahead();
    predictor.update({ Physics_speed_kmh: 7.2, Physics_brake: 0.4 }, 0, false);
    const sample = { Physics_speed_kmh: 3.6, Physics_brake: 0.3 };
    predictor.update(sample, 1000, true);
    const frames = predictor.getFrames(sample, null);
    expect(frames.map(({ offsetMs }) => offsetMs)).toEqual(Array.from({ length: 31 }, (_, i) => i * 100));
    frames.forEach((prediction) => {
        const seconds = prediction.offsetMs / 1000;
        expect(prediction.intendedTelemetry.Physics_speed_kmh).toBeCloseTo(Math.max(0, 3.6 - 3.6 * seconds));
        expect(prediction.intendedTelemetry.Physics_brake).toBeCloseTo(Math.max(0, 0.3 - 0.1 * seconds));
        expect(prediction.deceleration).toBe(seconds < 1 ? 1 : 0);
        expect(prediction.brakeReleasing).toBe(seconds < 3);
        expect(predictor.getFrame(sample, null, prediction.offsetMs)).toBe(prediction);
    });
    expect(predictor.getFrames(sample, null)).toBe(frames);
});

it('caches exact preparation boundaries and clamps access to the retained window', () => {
    const predictor = new PhraseLookahead();
    predictor.update({ Physics_speed_kmh: 36, Physics_steer_angle: 0 }, 0, false);
    const sample = { Physics_speed_kmh: 36, Physics_steer_angle: 0.1 };
    predictor.update(sample, 1000, true);
    const frames = predictor.getFrames(sample, null);
    const exact = predictor.getFrame(sample, null, 333);
    expect(exact.offsetMs).toBe(333);
    expect(exact.intendedTelemetry.Physics_steer_angle).toBeCloseTo(0.1333);
    expect(predictor.getFrame(sample, null, 333)).toBe(exact);
    expect(predictor.getFrame(sample, null, -100)).toBe(frames[0]);
    expect(predictor.getFrame(sample, null, 9000)).toBe(frames[30]);
    expect(predictor.getFrames(sample, null)).toBe(frames);
});

it('replaces the buffer on updates and resets without changing previously retained frames', () => {
    const predictor = new PhraseLookahead();
    const sample = { Physics_speed_kmh: 36, Physics_steer_angle: 0 };
    predictor.update(sample, 0, false);
    const first = predictor.getFrames(sample, null);
    sample.Physics_steer_angle = 0.1;
    predictor.update(sample, 1000, true);
    const next = predictor.getFrames(sample, null);
    expect(next).not.toBe(first);
    expect(first[30].intendedTelemetry.Physics_steer_angle).toBe(0);
    expect(next[30].intendedTelemetry.Physics_steer_angle).toBeCloseTo(0.4);
    predictor.reset();
    const reset = predictor.getFrames(sample, null);
    expect(reset).not.toBe(next);
    expect(reset[30].intendedTelemetry.Physics_steer_angle).toBe(0.1);
    expect(next[30].intendedTelemetry.Physics_steer_angle).toBeCloseTo(0.4);
});

it('rebuilds position predictions when reference geometry changes without new telemetry', () => {
    const predictor = new PhraseLookahead(), { reference } = steeringCornerFixture();
    const sample = { Physics_speed_kmh: 36, Graphics_normalized_car_position: 0.2 };
    predictor.update(sample, 1000, false, reference);
    const first = predictor.getFrames(sample, reference);
    const larger = { ...reference, y: reference.y.map((row) => row.map((value, i) => i < 3 ? value * 2 : value)) };
    const next = predictor.getFrames(sample, larger);
    expect(next).not.toBe(first);
    expect(next[30].intendedTelemetry.Graphics_normalized_car_position).toBe(referencePositionAhead(sample, larger));
    expect(next[30].intendedTelemetry.Graphics_normalized_car_position)
        .not.toBe(first[30].intendedTelemetry.Graphics_normalized_car_position);
});

it('lets closure and action conditions inspect any retained frame after their parent enters', () => {
    const run = jest.fn(), measure = jest.fn((_context: PhraseContext) => true);
    const engine = new PhraseEngine([], () => new Closure<PhraseContext>('root', '', () => true, [
        new Closure<PhraseContext>('inspect full forecast', '', (context) => context.predictionFrames.some(({ offsetMs, intendedTelemetry }) =>
            offsetMs === 2500 && (intendedTelemetry.Physics_steer_angle ?? 0) > 0.25), [
            new Action('child', '', run, measure),
        ]),
    ]));
    engine.receiveTelemetry(frame({ Physics_speed_kmh: 36, Physics_steer_angle: 0 }), 1000);
    expect(measure).not.toHaveBeenCalled();
    const result = engine.receiveTelemetry(frame({ Physics_speed_kmh: 36, Physics_steer_angle: 0.01 }), 1100);
    expect(result.state.path).toEqual(['root', 'inspect full forecast', 'child']);
    const context: PhraseContext = run.mock.calls[0][0];
    expect(context.predictionFrames).toHaveLength(31);
    expect(context.forPreparation!(2500).intendedTelemetry).toBe(context.predictionFrames[25].intendedTelemetry);
    expect(context.forPreparation!(333).intendedTelemetry.Physics_steer_angle).toBeCloseTo(0.0433);
    expect(measure.mock.calls[0][0].predictionFrames).toBe(context.predictionFrames);
});

it('reuses prediction frames on timer evaluations and hides them for stale or paused telemetry', () => {
    let context: PhraseContext;
    const project = jest.spyOn(PhraseLookahead.prototype, 'project');
    const engine = new PhraseEngine([], () => new Closure<PhraseContext>('root', '', () => true, [
        new Closure('inspect', '', (candidate) => { context = candidate; return false; }),
    ]));
    try {
        const sample = { Physics_speed_kmh: 36, Physics_steer_angle: 0 };
        engine.receiveTelemetry(frame(sample), 1000);
        const retained = context!.predictionFrames;
        expect(project).toHaveBeenCalledTimes(31);
        engine.evaluate(1100);
        expect(context!.predictionFrames).toBe(retained);
        expect(project).toHaveBeenCalledTimes(31);
        engine.evaluate(2501);
        expect(context!.predictionFrames).toEqual([]);
        expect(context!.intendedTelemetry).toEqual({});
        engine.receiveTelemetry(frame(sample), 2600);
        expect(context!.predictionFrames).toHaveLength(31);
        expect(context!.predictionFrames).not.toBe(retained);
        engine.receiveTelemetry({ ...frame(sample), telemetryStatus: 1 }, 2700);
        expect(context!.predictionFrames).toEqual([]);
        engine.reset(2800);
        engine.evaluate(2800);
        expect(context!.predictionFrames).toEqual([]);
    } finally {
        project.mockRestore();
    }
});

it('still checks an exact preparation endpoint between retained frames', () => {
    const run = jest.fn();
    const engine = new PhraseEngine([], () => new Closure<PhraseContext>('root', '', () => true, [
        new Closure('exact match', '', ({ intendedTelemetry }) => Math.abs((intendedTelemetry.Physics_steer_angle ?? 0) - 0.0433) < 1e-6,
            [new Action('child', '', run)], { preparationTimeMs: 333 }),
    ]));
    engine.receiveTelemetry(frame({ Physics_speed_kmh: 36, Physics_steer_angle: 0 }), 1000);
    expect(run).not.toHaveBeenCalled();
    const entered = engine.receiveTelemetry(frame({ Physics_speed_kmh: 36, Physics_steer_angle: 0.01 }), 1100);
    expect(entered.state.path).toEqual(['root', 'exact match', 'child']);
    expect(run.mock.calls[0][0].intendedTelemetry.Physics_steer_angle).toBeCloseTo(0.0433);
});
