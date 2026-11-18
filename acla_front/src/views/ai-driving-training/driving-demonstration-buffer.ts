import { DrivingAction, DrivingPedalIntent, DrivingTelemetrySample } from './driving-observation';
import type { DrivingPolicyInput } from './driving-memory';
import type { PlannedActionTarget, ShiftAvailability } from './driving-policy';
import { DEFAULT_DRIVING_CONFIG } from './driving-model-config';
import type { ControllerAction } from './virtual-controller';

export interface DrivingDemonstrationSample {
    input: DrivingPolicyInput;
    capturedAt: number;
    targets: PlannedActionTarget[];
    telemetry: DrivingTelemetrySample[];
}

interface RecordedAction { action: DrivingAction; shifts: ShiftAvailability; appliedAt: number }
type Origin = Pick<DrivingDemonstrationSample, 'input' | 'capturedAt'>;

/** Collection owns these CPU snapshots; GPU completion never chooses a label.
 * A bounded backlog drops the oldest completed observation first, keeping current
 * collection running. Camera pixels are shared immutable copies from DrivingMemory. */
export class DrivingDemonstrationBuffer {
    private actions: RecordedAction[] = [];
    private pedals = new DrivingPedalIntent();
    private telemetry: DrivingTelemetrySample[] = [];
    private pending: Origin[] = [];
    private ready: DrivingDemonstrationSample[] = [];
    droppedSamples = 0;

    constructor(private config = DEFAULT_DRIVING_CONFIG) {}

    get hasReady() { return this.ready.length > 0; }
    get queuedSamples() { return this.ready.length; }

    reset() {
        this.actions = [];
        this.pedals.reset();
        this.telemetry = [];
        this.pending = [];
        this.ready = [];
        this.droppedSamples = 0;
    }

    /** Shift flags are fresh paddle edges, never the controller's held pulse. */
    recordAction(controls: ControllerAction, shifts: ShiftAvailability, appliedAt: number) {
        const previous = this.actions[this.actions.length - 1];
        if (previous && appliedAt - previous.appliedAt > this.config.runtime.maxTransitionSeconds * 1000) this.pedals.reset();
        this.actions.push({ action: this.pedals.action(controls), shifts: { ...shifts }, appliedAt });
        if (this.actions.length > 10000) this.actions.shift();
    }

    recordTelemetry(sample: DrivingTelemetrySample) {
        const previous = this.telemetry[this.telemetry.length - 1];
        if (previous && sample.measuredAt < previous.measuredAt) return;
        if (previous && sample.session === previous.session) {
            // IPC can deliver distinct frames in the same millisecond; deduplicate
            // polling/subscription copies by stream index rather than receipt time.
            if (sample.sampleIndex !== undefined && previous.sampleIndex !== undefined
                && sample.sampleIndex <= previous.sampleIndex) return;
            if (sample.sampleIndex === undefined && previous.sampleIndex === undefined
                && sample.measuredAt === previous.measuredAt && JSON.stringify(sample) === JSON.stringify(previous)) return;
        }
        this.telemetry.push({ ...sample, context: { ...sample.context }, controls: { ...sample.controls },
            localVelocity: [...sample.localVelocity], gForce: [...sample.gForce] });
        if (this.telemetry.length > 10000) this.telemetry.shift();
    }

    addObservation(input: DrivingPolicyInput, capturedAt: number) {
        this.pending.push({ capturedAt, input: { features: input.features.slice(), cameras: input.cameras.slice() } });
        if (this.pending.length + this.ready.length > 512) {
            if (this.ready.length) this.ready.shift();
            else this.pending.shift();
            this.droppedSamples++;
        }
    }

    advance(now: number) {
        const horizon = this.config.actions.planTimesSeconds.slice(-1)[0] * 1000;
        const tolerance = this.config.learning.planTargetToleranceMs;
        const remaining: Origin[] = [];
        for (const origin of this.pending) {
            if (now >= origin.capturedAt + horizon + tolerance) this.complete(origin);
            else remaining.push(origin);
        }
        this.pending = remaining;
        // A camera may arrive late, after control samples have already been recorded.
        const cutoff = Math.min(now - horizon - this.config.runtime.visionMaxAgeMs - tolerance,
            this.pending.length ? this.pending[0].capturedAt - tolerance : Infinity);
        this.actions = this.actions.filter(({ appliedAt }) => appliedAt >= cutoff);
        this.telemetry = this.telemetry.filter(({ measuredAt }) => measuredAt >= cutoff);
    }

    /** Close an ownership segment with only the targets actually demonstrated. */
    finish() {
        for (const origin of this.pending) this.complete(origin);
        this.pending = [];
        this.actions = [];
        this.pedals.reset();
        this.telemetry = [];
    }

    take() { return this.ready.shift(); }

    private complete(origin: Origin) {
        const times = this.config.actions.planTimesSeconds.map((seconds) => origin.capturedAt + seconds * 1000);
        const tolerance = this.config.learning.planTargetToleranceMs;
        const shifts = new Map<number, RecordedAction[]>();
        for (const event of this.actions) {
            if ((!event.action.shiftUp && !event.action.shiftDown) || event.appliedAt < origin.capturedAt) continue;
            let step = 0;
            // Midpoint ties belong to the later slot, so each edge labels one slot.
            for (let next = 1; next < times.length; next++) {
                if (Math.abs(times[next] - event.appliedAt) <= Math.abs(times[step] - event.appliedAt)) step = next;
            }
            if (Math.abs(times[step] - event.appliedAt) > tolerance) continue;
            shifts.set(step, [...(shifts.get(step) ?? []), event]);
        }
        const targets: PlannedActionTarget[] = [];
        times.forEach((at, step) => {
            let nearest: RecordedAction | undefined;
            for (const sample of this.actions) {
                const distance = Math.abs(sample.appliedAt - at);
                if (distance <= tolerance && (!nearest || distance < Math.abs(nearest.appliedAt - at)
                    || sample.appliedAt === nearest.appliedAt)) nearest = sample;
            }
            if (!nearest) return;
            const events = shifts.get(step) ?? [];
            const event = events.length === 1 ? events[0] : undefined;
            targets.push({ step, action: { ...nearest.action,
                shiftUp: Boolean(event?.action.shiftUp), shiftDown: Boolean(event?.action.shiftDown) },
                // Multiple presses in one slot cannot fit its categorical output.
                // Allowing only hold makes its shift gradient zero; analogs still train.
                shifts: events.length > 1 ? { up: false, down: false } : { ...(event?.shifts ?? nearest.shifts) } });
        });
        if (targets.length) this.ready.push({ ...origin, targets,
            telemetry: this.telemetry.filter(({ measuredAt }) => measuredAt >= origin.capturedAt && measuredAt <= times[times.length - 1]) });
    }
}
