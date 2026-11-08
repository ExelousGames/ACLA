import type { OperationComponentRefDirectory } from 'contexts/OperationComponentRefContext';
import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import { overlaySessionClient } from 'views/floating-chat/overlay-display-client';
import type { AiOverlayPresentationSession } from 'views/floating-chat/ai-overlay-types';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { PHRASE_DEFINITIONS, type PhraseEvent, type PhraseSnapshot } from './phrase-engine';
import { buildSteeringCornerOverlay, type SteeringCornerOverlaySnapshot } from './steering-corner-overlay';

export const LIVE_PHRASE_DISPLAY_MS = 8_000;

export interface LivePhraseOverlaySnapshot {
    eventId: number;
    ruleId: string;
    name: string;
    sentence: string;
    timestamp: number;
    expiresAt?: number;
}

// Each phrase owns a normal addon handle. Only plain display data crosses IPC.
export const createLivePhraseOverlayComponent = (componentName: string) => (
    new MutableAiOverlayComponent<LivePhraseOverlaySnapshot>(
        componentName,
        'live_phrase',
        (snapshot, publication) => {
            const remaining = snapshot ? (snapshot.expiresAt ?? snapshot.timestamp + LIVE_PHRASE_DISPLAY_MS) - Date.now() : 0;
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
    private readonly steeringSource;
    private steeringGeneration = 0;
    private steeringInput: { map: CircuitMapDto; reference: TopLapReferenceEntry; cornerKey: string } | null = null;
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
        this.steeringSource = { current: new MutableAiOverlayComponent<SteeringCornerOverlaySnapshot>(
            `${name}:steering-corner`, 'live_steering_corner', (_snapshot, publication) => ({
                placement: 'flow', requestedStatus: 'focus', transientDurationMs: null,
                presentationId: publication.presentationId,
            }),
        ) };
        directory.registerComponentRef(this.steeringSource);
    }

    update(snapshot: PhraseSnapshot, map: CircuitMapDto | null = null, reference: TopLapReferenceEntry | null = null): void {
        if (this.disposed) return;
        if (snapshot.events.length === 0) {
            this.generation += 1;
            this.lastEventId = 0;
            this.sources.forEach(({ ref }) => ref.current.clear());
        }
        snapshot.events.filter((event) => event.id > this.lastEventId).forEach((event) => {
            this.lastEventId = event.id;
            void this.show(event, this.generation).catch((error) => {
                if (!this.disposed) this.onError(error);
            });
        });
        this.updateSteering(snapshot, map, reference);
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.generation += 1;
        this.sources.forEach(({ ref }) => {
            ref.current.clear();
            this.directory.unregisterComponentRef(ref);
        });
        this.clearSteering();
        this.directory.unregisterComponentRef(this.steeringSource);
        if (this.ownedPresentationId) {
            void overlaySessionClient.destroy(this.ownedPresentationId).catch(this.onError);
        }
    }

    private clearSteering(): void {
        this.steeringGeneration += 1;
        this.steeringInput = null;
        this.steeringSource.current.clear();
    }

    private updateSteering(snapshot: PhraseSnapshot, map: CircuitMapDto | null, reference: TopLapReferenceEntry | null): void {
        const corner = snapshot.steeringCorner;
        if (!snapshot.telemetryReady || !snapshot.referenceLoaded || !snapshot.mapReady
            || !snapshot.state.path.includes('Steering') || !corner || !map || !reference) {
            this.clearSteering();
            return;
        }
        const cornerKey = JSON.stringify([corner.sectionId, corner.cornerStartPosition, corner.cornerEndPosition]);
        if (this.steeringInput?.map === map && this.steeringInput.reference === reference
            && this.steeringInput.cornerKey === cornerKey) return;
        this.clearSteering();
        this.steeringInput = { map, reference, cornerKey };
        const display = buildSteeringCornerOverlay(map, reference, corner);
        if (display) void this.showSteering(display, this.steeringGeneration).catch((error) => {
            if (!this.disposed) this.onError(error);
        });
    }

    private createPresentation(): Promise<AiOverlayPresentationSession> {
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
        return this.pendingPresentation;
    }

    private async showSteering(snapshot: SteeringCornerOverlaySnapshot, generation: number): Promise<void> {
        if (!overlaySessionClient.available()) return;
        const presentation = overlaySessionClient.current() ?? await this.createPresentation();
        if (this.disposed || generation !== this.steeringGeneration || presentation.mode !== 'live'
            || overlaySessionClient.current()?.presentationId !== presentation.presentationId) return;
        this.steeringSource.current.publish(snapshot, { presentationId: presentation.presentationId });
    }

    private async show(event: PhraseEvent, generation: number): Promise<void> {
        const source = this.sources.find(({ rule }) => rule.id === event.ruleId);
        if (!source || !overlaySessionClient.available()
            || Date.now() >= (event.expiresAt ?? event.timestamp + LIVE_PHRASE_DISPLAY_MS)) return;
        const presentation = overlaySessionClient.current() ?? await this.createPresentation();
        if (this.disposed || generation !== this.generation || presentation.mode !== 'live'
            || overlaySessionClient.current()?.presentationId !== presentation.presentationId
            || Date.now() >= (event.expiresAt ?? event.timestamp + LIVE_PHRASE_DISPLAY_MS)) return;
        source.ref.current.publish({
            eventId: event.id,
            ruleId: event.ruleId,
            name: source.rule.name,
            sentence: event.sentence,
            timestamp: event.timestamp,
            ...(event.expiresAt !== undefined ? { expiresAt: event.expiresAt } : {}),
        }, { presentationId: presentation.presentationId });
    }
}
