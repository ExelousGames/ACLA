import type { OperationComponentRefDirectory } from 'contexts/OperationComponentRefContext';
import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import { overlaySessionClient } from 'views/floating-chat/overlay-display-client';
import type { AiOverlayPresentationSession } from 'views/floating-chat/ai-overlay-types';
import { PHRASE_DEFINITIONS, type PhraseEvent, type PhraseSnapshot } from './phrase-engine';

export const LIVE_PHRASE_DISPLAY_MS = 8_000;

export interface LivePhraseOverlaySnapshot {
    eventId: number;
    ruleId: string;
    name: string;
    sentence: string;
    timestamp: number;
}

// Each phrase owns a normal addon handle. Only plain display data crosses IPC.
export const createLivePhraseOverlayComponent = (componentName: string) => (
    new MutableAiOverlayComponent<LivePhraseOverlaySnapshot>(
        componentName,
        'live_phrase',
        (snapshot, publication) => {
            const remaining = snapshot ? snapshot.timestamp + LIVE_PHRASE_DISPLAY_MS - Date.now() : 0;
            return {
                placement: 'flow',
                requestedStatus: 'focus',
                transientDurationMs: Math.max(0, remaining),
                remove: remaining <= 0,
                presentationId: publication.presentationId,
            };
        },
    )
);

/** Main-window owner of phrase publication, session lifetime and addon handles. */
export class LivePhraseOverlay {
    private readonly sources;
    private lastEventId = 0;
    private generation = 0;
    private disposed = false;
    private ownedPresentationId: string | null = null;
    private pendingPresentation: Promise<AiOverlayPresentationSession> | null = null;

    constructor(
        private readonly directory: OperationComponentRefDirectory,
        private readonly name: string,
        private readonly onError: (error: unknown) => void,
    ) {
        this.sources = PHRASE_DEFINITIONS.map((rule) => ({
            rule,
            ref: { current: createLivePhraseOverlayComponent(`${name}:${rule.id}`) },
        }));
        this.sources.forEach(({ ref }) => directory.registerComponentRef(ref));
    }

    update(snapshot: PhraseSnapshot): void {
        if (this.disposed) return;
        if (snapshot.events.length === 0) {
            this.generation += 1;
            this.lastEventId = 0;
            this.sources.forEach(({ ref }) => ref.current.clear());
            return;
        }
        snapshot.events.filter((event) => event.id > this.lastEventId).forEach((event) => {
            this.lastEventId = event.id;
            void this.show(event, this.generation).catch((error) => {
                if (!this.disposed) this.onError(error);
            });
        });
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.generation += 1;
        this.sources.forEach(({ ref }) => {
            ref.current.clear();
            this.directory.unregisterComponentRef(ref);
        });
        if (this.ownedPresentationId) {
            void overlaySessionClient.destroy(this.ownedPresentationId).catch(this.onError);
        }
    }

    private async show(event: PhraseEvent, generation: number): Promise<void> {
        const source = this.sources.find(({ rule }) => rule.id === event.ruleId);
        if (!source || !overlaySessionClient.available()
            || Date.now() >= event.timestamp + LIVE_PHRASE_DISPLAY_MS) return;
        let presentation = overlaySessionClient.current();
        if (!presentation) {
            if (!this.pendingPresentation) {
                this.pendingPresentation = overlaySessionClient.create({
                    aiSessionId: this.name,
                    mode: 'live',
                    displayIdentity: { name: 'Kestrel', agentTags: ['Live phrases'] },
                }).then(async (created) => {
                    if (this.disposed) await overlaySessionClient.destroy(created.presentationId);
                    else this.ownedPresentationId = created.presentationId;
                    return created;
                }).finally(() => { this.pendingPresentation = null; });
            }
            presentation = await this.pendingPresentation;
        }
        if (this.disposed || generation !== this.generation || presentation.mode !== 'live'
            || overlaySessionClient.current()?.presentationId !== presentation.presentationId
            || Date.now() >= event.timestamp + LIVE_PHRASE_DISPLAY_MS) return;
        source.ref.current.publish({
            eventId: event.id,
            ruleId: event.ruleId,
            name: source.rule.name,
            sentence: event.sentence,
            timestamp: event.timestamp,
        }, { presentationId: presentation.presentationId });
    }
}
