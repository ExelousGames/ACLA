import type { AmplifyCurve, CurvePoint, ProsodySetting, TtsRequest } from 'components/tts';
import defaultCurves from './curves/default.json';

export type PhraseCurves = Readonly<Record<ProsodySetting, readonly CurvePoint[]>>;
export type PhraseInputs = Readonly<Record<ProsodySetting, number>>;
export const PHRASE_REACTION_ALLOWANCE_MS = 0;
export const PHRASE_SILENCE_LEVEL = 0.01;
export interface PhraseSpeechTiming {
    durationMs?: number;
    preparationTimeMs?: number;
    reactionAllowanceMs: number;
    /** The prepared clip was submitted to the audio player. */
    requestedAt?: number;
    /** First playback start notification; buffering resumes do not reset it. */
    playbackStartedAt?: number;
    /** Set when playback reaches the first audible frame, never by the trigger timestamp. */
    startedAt?: number;
    /** Terminal audio callback, including playback that was rejected or interrupted. */
    finishedAt?: number;
    outcome?: 'completed' | 'replaced' | 'rejected' | 'cancelled' | 'failed';
    releaseAt?: number;
}
export interface PhraseSettings extends Omit<TtsRequest, 'text'> {
    inputs?: Partial<PhraseInputs>;
    /** Higher values are scheduled first when several phrases are eligible. */
    priority?: number;
}
/** A prepared choice for the scheduler; audio variants have no entry or exit conditions. */
export interface PhraseAudioVariant {
    instance: PhraseInstance;
    durationMs: number;
}
const settings: readonly ProsodySetting[] = ['timing', 'energy', 'pitch'];

/** Speech prototype. Curves are repository assets; instances are independent, in-memory copies. */
export class Phrase {
    readonly speech: Readonly<TtsRequest>;
    readonly amplifyCurves: PhraseCurves;
    readonly inputs: PhraseInputs;
    readonly priority: number;

    constructor(readonly id: string, sentence: string, curves: PhraseCurves = defaultCurves, options: PhraseSettings = {}) {
        this.priority = options.priority ?? 0;
        this.speech = Object.freeze({ text: sentence.trim(), speed: options.speed ?? 1.2,
            language: options.language ?? 'en-us', ...(options.voice !== undefined ? { voice: options.voice } : {}) });
        this.amplifyCurves = Object.freeze({
            timing: Object.freeze(curves.timing.map((point) => Object.freeze({ ...point }))),
            energy: Object.freeze(curves.energy.map((point) => Object.freeze({ ...point }))),
            pitch: Object.freeze(curves.pitch.map((point) => Object.freeze({ ...point }))),
        });
        this.inputs = Object.freeze({ timing: 0, energy: 0, pitch: 0, ...options.inputs });
    }

    createInstance(overrides: PhraseSettings = {}): PhraseInstance {
        return new PhraseInstance(this, overrides);
    }

    createAudioInstances(overrides: readonly PhraseSettings[] = [{}]): PhraseInstance[] {
        const instances = overrides.map((options) => this.createInstance(options));
        return Array.from(new Map(instances.map((instance) => [instance.cacheKey, instance])).values());
    }

    get curves(): readonly AmplifyCurve[] {
        return settings.map((setting) => ({ setting, points: this.amplifyCurves[setting] }));
    }

    get inputValues(): readonly number[] { return settings.map((setting) => this.inputs[setting]); }
}

/** A runtime copy whose audio identity includes every speech setting and curve. Never persisted. */
export class PhraseInstance extends Phrase {
    readonly cacheKey: string;

    constructor(source: Phrase, overrides: PhraseSettings = {}) {
        super(source.id, source.speech.text, source.amplifyCurves, {
            ...source.speech, ...overrides, priority: overrides.priority ?? source.priority,
            inputs: { ...source.inputs, ...overrides.inputs },
        });
        this.cacheKey = JSON.stringify([this.speech, this.curves, this.inputValues]);
        Object.freeze(this);
    }
}
