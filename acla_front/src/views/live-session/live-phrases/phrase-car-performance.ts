/** Observations from the unadjusted reference, in metres, seconds and m/s².
 * Grade is rise / travelled distance (sin of the road angle), positive uphill.
 * Curvature is horizontal heading change / horizontal distance; zero is straight.
 */
export interface CarPerformanceSample {
    readonly distance: number;
    readonly speed: number;
    readonly curvature: number;
    readonly elevation: number;
    readonly grade: number;
    readonly deceleration: number;
    readonly brake: number;
    readonly throttle: number;
    readonly coasting: boolean;
}

interface BrakeResponse extends CarPerformanceSample {
    readonly brakeDeceleration: number;
    readonly combinedAcceleration: number;
}

export interface ReferenceCarPerformance {
    readonly samples: readonly CarPerformanceSample[];
    readonly coasting: readonly CarPerformanceSample[];
    readonly braking: readonly BrakeResponse[];
}

export interface CarMotionState { speed: number; curvature: number; grade: number }
const GRAVITY = 9.80665;
const lateralAcceleration = (state: CarMotionState) => state.speed ** 2 * (1 - state.grade ** 2) * Math.abs(state.curvature);
const stateDifference = (a: CarMotionState, b: CarMotionState) =>
    ((a.speed - b.speed) / 5) ** 2 + ((lateralAcceleration(a) - lateralAcceleration(b)) / 2) ** 2;

/** Do not infer free slowing from reference braking, throttle, or unsupported speeds/turn loads. */
function coastResistance(samples: readonly CarPerformanceSample[], state: CarMotionState): number {
    let nearest: CarPerformanceSample | undefined, difference = Infinity;
    for (const sample of samples) {
        if (Math.abs(sample.speed - state.speed) > Math.max(5, state.speed * 0.2)
            || Math.abs(lateralAcceleration(sample) - lateralAcceleration(state)) > Math.max(1, lateralAcceleration(state) * 0.25)) continue;
        const delta = stateDifference(sample, state);
        if (delta < difference) { nearest = sample; difference = delta; }
    }
    return nearest ? Math.max(0, nearest.deceleration - GRAVITY * nearest.grade) : 0;
}

/** Capture car behaviour once; Instructor aggression changes targets, not these measurements. */
export function buildReferenceCarPerformance(observations: readonly CarPerformanceSample[]): ReferenceCarPerformance {
    const samples = Object.freeze(observations.map((sample) => Object.freeze({ ...sample })));
    const coasting = Object.freeze(samples.filter((sample) => sample.coasting));
    const braking = Object.freeze(samples.flatMap((sample) => {
        if (sample.brake < 0.1 || sample.throttle > 0.01) return [];
        const brakeDeceleration = sample.deceleration - GRAVITY * sample.grade - coastResistance(coasting, sample);
        if (brakeDeceleration < 0.1) return [];
        return [Object.freeze({ ...sample, brakeDeceleration,
            combinedAcceleration: Math.hypot(brakeDeceleration, lateralAcceleration(sample)) })];
    }));
    return Object.freeze({ samples, coasting, braking });
}

/** Empirical combined-grip envelope, not a measured tyre limit. Only use observations
 * at comparable speeds, reserving lateral acceleration v²/r before estimating braking.
 * Gravity is added separately so downhill acceleration is never mistaken for free slowing.
 */
export function carDecelerationAt(performance: ReferenceCarPerformance, state: CarMotionState) {
    const coastingDeceleration = coastResistance(performance.coasting, state) + GRAVITY * state.grade;
    let grip: number | undefined, response: BrakeResponse | undefined, difference = Infinity;
    for (const sample of performance.braking) {
        if (Math.abs(sample.speed - state.speed) > Math.max(5, state.speed * 0.2)) continue;
        grip = Math.max(grip ?? 0, sample.combinedAcceleration);
        const delta = stateDifference(sample, state);
        if (delta < difference) { response = sample; difference = delta; }
    }
    const availableBrakeDeceleration = grip === undefined ? undefined
        : Math.sqrt(Math.max(0, grip ** 2 - lateralAcceleration(state) ** 2));
    return { coastingDeceleration, availableBrakeDeceleration,
        possibleDeceleration: availableBrakeDeceleration === undefined ? undefined : availableBrakeDeceleration + coastingDeceleration,
        decelerationPerBrake: response ? response.brakeDeceleration / response.brake : undefined };
}
