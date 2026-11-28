import { DRIVING_MIN_GEAR } from './driving-model-config';
import type { DrivingModelConfig } from './driving-model-config';
import type { DrivingAction } from './driving-observation';

export const DRIVING_DIRECTED_TRIAL_SECONDS = 0.5;
export const DRIVING_DIRECTED_INTERVAL_MS = 2000;
const MIN_SCORE_GAIN = 0.0001;

export interface DrivingDirectedOptions {
    throttleLimited: boolean;
}

function sampleRange(min: number, max: number, samples: number) {
    if (min === max) return [min];
    return Array.from({ length: samples }, (_, index) => index === samples - 1
        ? max : min + (max - min) * index / (samples - 1));
}

export function directedActionCandidates(baseline: DrivingAction, maxGear: number, throttleLimit: number,
    settings: DrivingModelConfig['learning']): DrivingAction[] {
    const candidates = [baseline];
    const maxPedal = Math.min(settings.directedPedalMax, throttleLimit);
    if (maxPedal < settings.directedPedalMin) return candidates;
    const pedals = sampleRange(settings.directedPedalMin, maxPedal, settings.directedPedalSamples);
    const steeringValues = sampleRange(settings.directedSteeringMin, settings.directedSteeringMax, settings.directedSteeringSamples);
    for (let gear = DRIVING_MIN_GEAR; gear <= maxGear; gear++) {
        for (const pedal of pedals) {
            for (const steering of steeringValues) {
                if (gear !== baseline.gear || pedal !== baseline.pedal || steering !== baseline.steering) {
                    candidates.push({ pedal, steering, gear });
                }
            }
        }
    }
    return candidates;
}

function explorationScore([first, second]: readonly [number, number]) {
    // Critic disagreement is a small exploration bonus, measured in reward units.
    return first / 2 + second / 2 + 0.25 * Math.abs(first - second);
}

export function selectDirectedAction(candidates: readonly DrivingAction[], scores: readonly (readonly [number, number])[]): DrivingAction | undefined {
    if (!candidates.length || scores.length !== candidates.length) return;
    const baselineScore = explorationScore(scores[0]);
    if (!Number.isFinite(baselineScore)) return;
    let best = 0;
    let bestScore = baselineScore + MIN_SCORE_GAIN;
    for (let index = 1; index < candidates.length; index++) {
        const score = explorationScore(scores[index]);
        if (Number.isFinite(score) && score > bestScore) {
            best = index;
            bestScore = score;
        }
    }
    if (best) return candidates[best];
}
