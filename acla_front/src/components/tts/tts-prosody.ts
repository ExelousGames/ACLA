import apiService from 'services/api.service';
import { createTtsPack, type TtsPack, type TtsRequest } from './tts-service';

export type ProsodySetting = 'timing' | 'energy' | 'pitch';
export interface CurvePoint { readonly position: number; readonly value: number }
export interface AmplifyCurve {
    readonly setting: ProsodySetting;
    readonly points: readonly CurvePoint[];
}
export interface PhonemePreview {
    phonemes: string;
    tokens: { index: number; symbol: string; token_id: number }[];
    sample_rate: number;
    max_phonemes: number;
}

const controls = {
    timing: { neutral: 1, min: 0.25, max: 4 },
    energy: { neutral: 0, min: -12, max: 12 },
    pitch: { neutral: 0, min: -12, max: 12 },
};
const checkCancelled = (signal?: AbortSignal) => {
    if (signal?.aborted) throw new DOMException('Speech generation was cancelled.', 'AbortError');
};
const speechText = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || trimmed.length > 4000) throw new Error('Speech text must contain 1 to 4000 characters.');
    return trimmed;
};

/** Add each scaled amplitude to the backend's neutral value, without mutating the source curves. */
export function applyAmplifyCurves(curves: readonly AmplifyCurve[], inputs: readonly number[]) {
    if (curves.length !== inputs.length) throw new Error('Each amplify curve needs one input value.');
    const result: Partial<Record<ProsodySetting, CurvePoint[]>> = {};
    curves.forEach(({ setting, points }, index) => {
        if (!Object.prototype.hasOwnProperty.call(controls, setting) || result[setting]) {
            throw new Error('Use at most one amplify curve for each of timing, energy and pitch.');
        }
        if (!Number.isFinite(inputs[index])) throw new Error('Prosody inputs must be finite numbers.');
        if (points.length < 2 || points.length > 64 || points[0].position !== 0 || points[points.length - 1].position !== 1) {
            throw new Error('Amplify curves need 2 to 64 points with endpoints at 0 and 1.');
        }
        const { neutral, min, max } = controls[setting];
        result[setting] = points.map((point, pointIndex) => {
            if (!Number.isFinite(point.position) || point.position < 0 || point.position > 1
                || (pointIndex > 0 && point.position <= points[pointIndex - 1].position)
                || !Number.isFinite(point.value)) {
                throw new Error('Curve points must be finite with strictly increasing positions between 0 and 1.');
            }
            const value = neutral + point.value * inputs[index];
            if (!Number.isFinite(value) || value < min || value > max) {
                throw new Error(`${setting} curve values must be between ${min} and ${max} after amplification.`);
            }
            return { position: point.position, value };
        });
    });
    return result;
}

export async function previewTtsPhonemes(speech: Pick<TtsRequest, 'text' | 'language'>, signal?: AbortSignal): Promise<PhonemePreview> {
    const text = speechText(speech.text);
    checkCancelled(signal);
    const response = await apiService.post<PhonemePreview>('/voice/phonemes', {
        text, language: speech.language ?? 'en-us',
    }, { timeout: 120_000, signal });
    checkCancelled(signal);
    return response.data;
}

/** Curves and floating-point inputs correspond by array index; positions span the previewed speech. */
export async function synthesizeProsodyTts(
    curves: readonly AmplifyCurve[], inputs: readonly number[], speech: TtsRequest, signal?: AbortSignal,
): Promise<TtsPack> {
    checkCancelled(signal);
    const text = speechText(speech.text);
    const adjustments = applyAmplifyCurves(curves, inputs);
    const speed = speech.speed ?? 1;
    if (!Number.isFinite(speed) || speed < 0.5 || speed > 2) throw new Error('Speech speed must be between 0.5 and 2.');
    const request = { text, language: speech.language ?? 'en-us', speed, ...(speech.voice !== undefined ? { voice: speech.voice } : {}) };
    const preview = await previewTtsPhonemes(request, signal);
    const symbols = Array.from(preview.phonemes);
    if (!symbols.length || symbols.length > preview.max_phonemes || preview.tokens.length !== symbols.length
        || preview.tokens.some((token, index) => token.index !== index || token.symbol !== symbols[index])) {
        throw new Error('The speech service returned an invalid indexed phoneme preview.');
    }
    const response = await apiService.post<ArrayBuffer>('/voice/synthesize/prosody', {
        ...request,
        prosody: {
            expected_phonemes: preview.phonemes,
            curves: curves.length ? [{ start: 0, end: preview.tokens.length, ...adjustments }] : [],
        },
    }, { responseType: 'arraybuffer', timeout: 120_000, signal });
    checkCancelled(signal);
    return createTtsPack(text, response.data);
}
