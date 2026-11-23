import { DrivingDemonstrationBuffer } from './driving-demonstration-buffer';
import { cloneConfig } from './driving-model-config';
import { DrivingTelemetrySample, NEUTRAL } from './driving-observation';
import type { DrivingPolicyInput } from './driving-memory';

const shifts = { up: true, down: true };
const input = (id = 1): DrivingPolicyInput => ({ features: [id], cameras: [] });
const action = (value: number) => ({ throttle: value, brake: 0, steering: -value });
const telemetry = (measuredAt: number): DrivingTelemetrySample => ({ measuredAt, session: 'session',
    context: { game: 'acc', car: 'GT3', map: 'monza' }, normalizedPosition: 0.2, speed: 100,
    localVelocity: [0, 30, 0], gForce: [0, 0, 1], yawRate: 0, gear: 3, rpm: 5000, controls: action(0.3) });

it('collects all 51 timestamps before exposing one grouped demonstration, independent of training', () => {
    const buffer = new DrivingDemonstrationBuffer();
    const features = input();
    buffer.addObservation(features, 1000);
    features.features[0] = 2;
    for (let step = 0; step <= 50; step++) {
        const current = action(step / 50);
        buffer.recordAction(current, shifts, 1000 + step * 100);
        current.throttle = 0; // Collection owns a snapshot.
        buffer.advance(1000 + step * 100);
        expect(buffer.hasReady).toBe(false);
    }
    buffer.advance(6050);
    const sample = buffer.take()!;
    expect(sample.input.features).toEqual([1]);
    expect(sample.targets).toEqual(Array.from({ length: 51 }, (_, step) => ({ step,
        action: { pedal: step / 50, steering: -step / 50, shiftUp: false, shiftDown: false }, shifts })));
    expect(buffer.hasReady).toBe(false);
});

it('labels live demonstration targets with brake throughout pedal overlap', () => {
    const config = cloneConfig();
    config.actions.planTimesSeconds = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
    const buffer = new DrivingDemonstrationBuffer(config);
    buffer.addObservation(input(), 1000);
    [[0, 0], [0.8, 0], [0.8, 0.2], [0.8, 0.2], [0.9, 0.1], [0.9, 0]]
        .forEach(([throttle, brake], i) => buffer.recordAction({ throttle, brake, steering: 0 }, shifts, 1000 + i * 100));
    buffer.finish();
    expect(buffer.take()!.targets.map(({ action }) => action.pedal)).toEqual([0, 0.8, -0.2, -0.2, -0.1, 0.9]);

    buffer.addObservation(input(), 1600);
    buffer.recordAction({ throttle: 0.6, brake: 0.1, steering: 0 }, shifts, 1600);
    buffer.finish();
    expect(buffer.take()!.targets[0].action.pedal).toBe(-0.1);
});

it('retains overlapping origins until consumed and supplies only targets within timestamp tolerance', () => {
    const config = cloneConfig();
    config.actions.planTimesSeconds = [0, 0.1, 0.2, 0.3];
    const buffer = new DrivingDemonstrationBuffer(config);
    buffer.addObservation(input(1), 1000);
    buffer.addObservation(input(2), 1100);
    buffer.recordAction(action(0.1), shifts, 1000);
    buffer.recordAction(action(0.2), shifts, 1080);
    buffer.recordAction(action(0.3), shifts, 1110); // Closer to 1100 than 1080.
    buffer.recordAction(action(0.4), shifts, 1260); // Outside 1200's tolerance.
    buffer.recordAction(action(0.5), shifts, 1320);
    buffer.advance(1450);
    expect(buffer.queuedSamples).toBe(2);
    const first = buffer.take()!, second = buffer.take()!;
    expect(first.targets.map(({ step, action: value }) => [step, value.pedal])).toEqual([[0, 0.1], [1, 0.3], [3, 0.5]]);
    expect(second.targets.map(({ step, action: value }) => [step, value.pedal])).toEqual([[0, 0.3], [2, 0.5]]);
});

it('keeps distinct short paddle presses, assigns midpoint edges once, and never labels held pulses', () => {
    const config = cloneConfig();
    config.actions.planTimesSeconds = [0, 0.1, 0.2, 0.3];
    const buffer = new DrivingDemonstrationBuffer(config);
    buffer.addObservation(input(), 1000);
    for (let at = 1000; at <= 1300; at += 10) {
        buffer.recordAction({ ...NEUTRAL, shiftUp: at === 1050 || at === 1160, shiftDown: at === 1290 }, shifts, at);
    }
    buffer.advance(1350);
    expect(buffer.take()!.targets.map(({ action: value }) => [value.shiftUp, value.shiftDown]))
        .toEqual([[false, false], [true, false], [true, false], [false, true]]);
});

it('masks only shift supervision when multiple presses cannot fit one categorical output', () => {
    const config = cloneConfig();
    config.actions.planTimesSeconds = [0, 0.1];
    const buffer = new DrivingDemonstrationBuffer(config);
    buffer.addObservation(input(), 1000);
    buffer.recordAction(action(0.2), shifts, 1000);
    buffer.recordAction({ ...action(0.4), shiftUp: true }, shifts, 1080);
    buffer.recordAction({ ...action(0.6), shiftDown: true }, shifts, 1120);
    buffer.advance(1150);
    expect(buffer.take()!.targets[1]).toEqual({ step: 1,
        action: { pedal: 0.4, steering: -0.4, shiftUp: false, shiftDown: false }, shifts: { up: false, down: false } });
});

it('records telemetry and controls while waiting for a delayed camera, without duplicating telemetry timestamps', () => {
    const buffer = new DrivingDemonstrationBuffer();
    for (let at = 1000; at <= 6000; at += 100) {
        const live = telemetry(at);
        buffer.recordTelemetry(live);
        buffer.recordTelemetry(live);
        live.controls.throttle = 1;
        live.localVelocity[1] = 0;
        buffer.recordAction(action((at - 1000) / 5000), shifts, at);
        buffer.advance(at);
        if (at === 2400) buffer.addObservation(input(), 1000);
    }
    buffer.advance(6050);
    const sample = buffer.take()!;
    expect(sample.targets).toHaveLength(51);
    expect(sample.telemetry).toHaveLength(51);
    expect(sample.telemetry[0]).toEqual(telemetry(1000));
    expect(sample.telemetry[50].measuredAt).toBe(6000);
});

it('finishes partial futures at a boundary and reset prevents cross-episode targets', () => {
    const buffer = new DrivingDemonstrationBuffer();
    buffer.addObservation(input(), 1000);
    buffer.recordAction(action(0.2), shifts, 1000);
    buffer.recordAction(action(0.4), shifts, 1100);
    buffer.finish();
    expect(buffer.take()!.targets.map(({ step }) => step)).toEqual([0, 1]);
    buffer.addObservation(input(), 1200);
    buffer.recordAction(action(0.6), shifts, 1200);
    buffer.reset();
    buffer.advance(5000);
    expect(buffer.take()).toBeUndefined();
    buffer.addObservation(input(), 5000);
    buffer.finish();
    expect(buffer.take()).toBeUndefined();
});

it('keeps separate telemetry frames delivered in one millisecond and skips polling duplicates', () => {
    const buffer = new DrivingDemonstrationBuffer();
    const first = { ...telemetry(1000), sampleIndex: 1 };
    const second = { ...telemetry(1000), sampleIndex: 2, speed: 101 };
    buffer.addObservation(input(), 1000);
    buffer.recordAction(NEUTRAL, shifts, 1000);
    buffer.recordTelemetry(first);
    buffer.recordTelemetry({ ...first });
    buffer.recordTelemetry(second);
    buffer.recordTelemetry({ ...second });
    buffer.finish();
    expect(buffer.take()!.telemetry).toEqual([first, second]);
});

it('bounds a stalled learner backlog by dropping oldest completed samples first', () => {
    const config = cloneConfig();
    config.actions.planTimesSeconds = [0];
    const buffer = new DrivingDemonstrationBuffer(config);
    for (let index = 0; index < 520; index++) {
        const at = 1000 + index * 100;
        buffer.recordAction(NEUTRAL, shifts, at);
        buffer.addObservation(input(index), at);
        buffer.advance(at + 50);
    }
    expect(buffer.droppedSamples).toBe(8);
    expect(buffer.queuedSamples).toBe(512);
    expect(buffer.take()!.input.features).toEqual([8]);
});
