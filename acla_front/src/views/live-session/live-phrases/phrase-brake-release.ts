import type { PhraseSettings } from './phrase';

export const SLOWLY_RELEASE_BRAKE = 'Slowly release brake';
// A finite set lets every possible delivery be synthesized before live playback.
// Slower easing of deceleration stretches the release instruction more.
export const BRAKE_RELEASE_SETTINGS: readonly PhraseSettings[] = [1, 2 / 3, 1 / 3, 0]
    .map((timing) => ({ inputs: { timing } }));

/** Positive reduction in reference deceleration per second, in m/s³. */
export function brakeReleaseSettings(rate: number): PhraseSettings {
    return BRAKE_RELEASE_SETTINGS[rate <= 0.5 ? 0 : rate <= 1 ? 1 : rate <= 2 ? 2 : 3];
}
