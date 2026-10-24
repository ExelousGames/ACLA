import { DrivingObservation, MAX_TRAINING_TRANSITION_SECONDS } from './driving-observation';

export const DRIVING_MEMORY_MS = 3000;
export const DRIVING_MEMORY_INTERVAL_MS = 200;
const HISTORY_SLOTS = DRIVING_MEMORY_MS / DRIVING_MEMORY_INTERVAL_MS;
// Twenty changing telemetry/reference inputs, plus presence and normalized age.
const SLOT_INPUTS = 22;
export const DRIVING_MEMORY_INPUTS = HISTORY_SLOTS * SLOT_INPUTS;

/** A timestamped history shared by inference, the critic and human demonstrations.
 * Static categories and segmentation remain in the current observation only. */
export class DrivingMemory {
    private frames: { capturedAt: number; features: number[] }[] = [];
    private last?: DrivingObservation;

    reset() {
        this.frames = [];
        this.last = undefined;
    }

    encode(observation: DrivingObservation): number[] {
        const previous = this.last;
        if (previous && (observation.capturedAt < previous.capturedAt
            || observation.capturedAt - previous.capturedAt > MAX_TRAINING_TRANSITION_SECONDS * 1000
            || observation.session !== previous.session || observation.reference !== previous.reference
            || JSON.stringify(observation.context) !== JSON.stringify(previous.context))) this.reset();

        this.frames = this.frames.filter((frame) => observation.capturedAt - frame.capturedAt <= DRIVING_MEMORY_MS);
        const history = new Array(DRIVING_MEMORY_INPUTS).fill(0);
        // Newest sample per elapsed-time bin; missing bins stay explicitly absent.
        for (let i = this.frames.length - 1; i >= 0; i--) {
            const frame = this.frames[i];
            const age = observation.capturedAt - frame.capturedAt;
            if (age <= 0) continue;
            const offset = (Math.ceil(age / DRIVING_MEMORY_INTERVAL_MS) - 1) * SLOT_INPUTS;
            if (history[offset]) continue;
            history[offset] = 1;
            history[offset + 1] = age / DRIVING_MEMORY_MS;
            frame.features.forEach((value, j) => { history[offset + 2 + j] = value; });
        }
        if (!this.last || observation.capturedAt > this.last.capturedAt) {
            const features = observation.features;
            this.frames.push({ capturedAt: observation.capturedAt, features: [
                ...features.slice(0, 4), // Local velocity XYZ and speed.
                ...features.slice(52, 65), // Position, controls, reference offsets, gear/RPM and rotation rates.
                ...features.slice(-3), // Aggression and target pedals.
            ] });
            this.last = observation;
        }
        return [...history, ...observation.features];
    }
}
