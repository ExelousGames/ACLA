import type { OperationComponentRefDirectory } from 'contexts/OperationComponentRefContext';
import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import { overlaySessionClient } from 'views/floating-chat/overlay-display-client';
import type { AiOverlayPresentationSession } from 'views/floating-chat/ai-overlay-types';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { PhraseSnapshot } from './phrase-engine';
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

/** Main-window owner of steering-corner publication, session lifetime and addon handles. */
export class LivePhraseOverlay {
    private readonly steeringSource;
    private steeringGeneration = 0;
    private steeringInput: { map: CircuitMapDto; reference: TopLapReferenceEntry; cornerKey: string } | null = null;
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

}
