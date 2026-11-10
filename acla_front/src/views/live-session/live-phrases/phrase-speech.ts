import { synthesizeProsodyTts, type TtsPack } from 'components/tts';
import { getPhraseActions, PHRASE_DEFINITIONS, type PhraseDefinition } from './phrase-engine';
import type { PhraseInstance } from './phrase';

export function getPhraseSpeechInstances(phrases: readonly PhraseDefinition[] = PHRASE_DEFINITIONS): PhraseInstance[] {
    const instances = [...phrases.filter((phrase) => phrase.referenceCue), ...phrases.filter((phrase) => !phrase.referenceCue)]
        .flatMap(getPhraseActions).flatMap((action) => action.phrases.flatMap((phrase) =>
            [{}, ...(action.preloadSettings ?? [])].map((settings) => phrase.createInstance(settings))));
    return Array.from(new Map(instances.map((instance) => [instance.cacheKey, instance])).values());
}

/** Request the complete speech batch at startup. Playback must only read the resulting cache. */
export async function loadAllPhraseSpeech(
    instances: readonly PhraseInstance[], packs: Map<string, TtsPack>, signal: AbortSignal,
    onPrepared?: () => void,
): Promise<{ complete: boolean; errors: unknown[] }> {
    const errors: unknown[] = [];
    // Serialize model work, but submit the full catalog through this single entry point.
    for (const instance of Array.from(new Map(instances.map((item) => [item.cacheKey, item])).values())) {
        if (signal.aborted) throw new DOMException('Speech generation was cancelled.', 'AbortError');
        if (packs.has(instance.cacheKey)) continue;
        try {
            const pack = await synthesizeProsodyTts(instance.curves, instance.inputValues, instance.speech, signal, true);
            if (signal.aborted) throw new DOMException('Speech generation was cancelled.', 'AbortError');
            packs.set(instance.cacheKey, pack);
            onPrepared?.();
        } catch (error) {
            if (signal.aborted) throw error;
            errors.push(error);
        }
    }
    return { complete: instances.every((instance) => packs.has(instance.cacheKey)), errors };
}
