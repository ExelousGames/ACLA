import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import { OVERLAY_HOLD_MS } from 'views/floating-chat/ai-overlay-types';
import type { AiMapDisplayPayload } from './AiMapToolDisplay';

export const createAiMapOverlayComponent = (componentName: string) => (
    new MutableAiOverlayComponent<AiMapDisplayPayload>(
        componentName,
        'map',
        (_snapshot, publication) => ({
            placement: 'flow',
            requestedStatus: publication.requestedStatus ?? 'expanded',
            transientDurationMs: OVERLAY_HOLD_MS,
            presentationId: publication.presentationId,
        }),
    )
);
