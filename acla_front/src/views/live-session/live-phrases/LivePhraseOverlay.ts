import type { OperationComponentRefDirectory } from 'contexts/OperationComponentRefContext';
import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import { overlaySessionClient } from 'views/floating-chat/overlay-display-client';
import type { AiOverlayPresentationSession } from 'views/floating-chat/ai-overlay-types';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { PhraseSnapshot } from './phrase-engine';
import { buildSteeringCornerOverlay, type SteeringCornerOverlaySnapshot } from './steering-corner-overlay';

export const LIVE_PHRASE_DISPLAY_MS = 2_000;

export interface LivePhraseOverlaySnapshot {
    eventId: number;
    ruleId: string;
    name: string;
    sentence: string;
    timestamp: number;
    expiresAt?: number;
}

/** Main-window owner of steering-corner publication, session lifetime and addon handles. */
export class LivePhraseOverlay {
    private readonly steeringSource;
    private steeringGeneration = 0;
    private lastEventId = 0;
    private steeringTimer: ReturnType<typeof setTimeout> | null = null;
    private disposed = false;
    private ownedPresentationId: string | null = null;
    private pendingPresentation: Promise<AiOverlayPresentationSession> | null = null;

    constructor(
        private readonly directory: OperationComponentRefDirectory,
        private readonly name: string,
        private readonly onError: (error: unknown) => void,
    ) {
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
        this.updateSteering(snapshot, map, reference);
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.clearSteering();
        this.directory.unregisterComponentRef(this.steeringSource);
        if (this.ownedPresentationId) {
            void overlaySessionClient.destroy(this.ownedPresentationId).catch(this.onError);
        }
    }

    private clearSteering(): void {
        this.steeringGeneration += 1;
        if (this.steeringTimer !== null) clearTimeout(this.steeringTimer);
        this.steeringTimer = null;
        this.steeringSource.current.clear();
    }

    private updateSteering(snapshot: PhraseSnapshot, map: CircuitMapDto | null, reference: TopLapReferenceEntry | null): void {
        if (snapshot.events.length === 0) {
            this.lastEventId = 0;
            this.clearSteering();
            return;
        }
        const event = snapshot.events.filter(({ id, ruleId }) => id > this.lastEventId && ruleId === 'reference-steering').pop();
        this.lastEventId = Math.max(this.lastEventId, ...snapshot.events.map(({ id }) => id));
        if (!snapshot.trackGuideEnabled) {
            this.clearSteering();
            return;
        }
        if (!event || event.timestamp + LIVE_PHRASE_DISPLAY_MS <= Date.now()) return;
        const corner = snapshot.steeringCorner;
        if (!snapshot.telemetryReady || !snapshot.referenceLoaded || !snapshot.mapReady
            || !corner || !map || !reference) return;
        // Capture geometry once alongside the audio trigger. Later telemetry and map
        // updates must neither change this card nor extend its lifetime.
        const display = buildSteeringCornerOverlay(map, reference, corner);
        if (!display) return;
        const generation = ++this.steeringGeneration;
        if (this.steeringTimer !== null) clearTimeout(this.steeringTimer);
        this.steeringTimer = setTimeout(() => this.clearSteering(), event.timestamp + LIVE_PHRASE_DISPLAY_MS - Date.now());
        void this.showSteering(display, generation).catch((error) => {
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

}
