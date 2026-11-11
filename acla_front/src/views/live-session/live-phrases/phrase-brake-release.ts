import type { PhraseSettings } from './phrase';

export const SLOWLY_RELEASE_BRAKE = 'Ease brake';
// A finite set lets every possible delivery be synthesized before live playback.
// Select the delivery closest to the remaining reference easing time.
export const BRAKE_RELEASE_SETTINGS: readonly PhraseSettings[] = [1, 2 / 3, 1 / 3, 0]
    .map((timing) => ({ inputs: { timing } }));

/** Match the remaining easing window to measured, prepared audio durations. */
export function brakeReleaseSettings(durationMs: number | undefined,
    preparedDuration: (settings: PhraseSettings) => number | undefined): PhraseSettings {
    const neutral = BRAKE_RELEASE_SETTINGS[BRAKE_RELEASE_SETTINGS.length - 1];
    if (durationMs === undefined || !Number.isFinite(durationMs) || durationMs <= 0) return neutral;
    return BRAKE_RELEASE_SETTINGS.map((settings) => ({ settings, duration: preparedDuration(settings) }))
        .filter((variant): variant is { settings: PhraseSettings; duration: number } =>
            variant.duration !== undefined && Number.isFinite(variant.duration) && variant.duration > 0)
        .sort((a, b) => Math.abs(a.duration - durationMs) - Math.abs(b.duration - durationMs)
            || a.duration - b.duration)[0]?.settings ?? neutral;
}
