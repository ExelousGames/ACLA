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
export interface PhraseOptions extends PhraseSettings {
    /** Interchangeable wordings in the random pool; entry/exit conditions belong to the phrase. */
    randomAlternatives?: readonly string[];
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
    /** Complete random wording pool, including the original sentence. */
    readonly randomAlternatives: readonly string[];
    readonly amplifyCurves: PhraseCurves;
    readonly inputs: PhraseInputs;
    readonly priority: number;

    constructor(readonly id: string, sentence: string, curves: PhraseCurves = defaultCurves, options: PhraseOptions = {}) {
        this.priority = options.priority ?? 0;
        this.randomAlternatives = Object.freeze(Array.from(new Set([sentence, ...(options.randomAlternatives ?? [])]
            .map((wording) => wording.trim()))));
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
        const instances = this.randomAlternatives.flatMap((sentence) => overrides.map((options) => new PhraseInstance(this, options, sentence)));
        return Array.from(new Map(instances.map((instance) => [instance.cacheKey, instance])).values());
    }

    /** Choose only at execution, after the scheduler has narrowed the prepared pool. */
    selectAudioInstance(allowed: readonly PhraseInstance[]): PhraseInstance | undefined {
        return allowed[allowed.length > 1 ? Math.floor(Math.random() * allowed.length) : 0];
    }

    get curves(): readonly AmplifyCurve[] {
        return settings.map((setting) => ({ setting, points: this.amplifyCurves[setting] }));
    }

    get inputValues(): readonly number[] { return settings.map((setting) => this.inputs[setting]); }
}

/** A runtime copy whose audio identity includes every speech setting and curve. Never persisted. */
export class PhraseInstance extends Phrase {
    readonly cacheKey: string;

    constructor(source: Phrase, overrides: PhraseSettings = {}, sentence = source.speech.text) {
        super(source.id, sentence, source.amplifyCurves, {
            ...source.speech, ...overrides, priority: overrides.priority ?? source.priority,
            randomAlternatives: undefined,
            inputs: { ...source.inputs, ...overrides.inputs },
        });
        this.cacheKey = JSON.stringify([this.speech, this.curves, this.inputValues]);
        Object.freeze(this);
    }
}
