import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import { OVERLAY_HOLD_MS } from 'views/floating-chat/ai-overlay-types';
import type { AiMessageOverlaySnapshot } from './AiMessageDisplay';

export const createAiMessageOverlayComponent = (componentName: string) => (
    new MutableAiOverlayComponent<AiMessageOverlaySnapshot>(
        componentName,
        'ai_message',
        (_snapshot, publication) => ({
            placement: 'flow',
            requestedStatus: publication.requestedStatus ?? 'expanded',
            shellSlot: 'speech',
            transientDurationMs: null,
            presentationId: publication.presentationId,
        }),
        (event) => event.event === 'visual_complete'
            ? { removeAfterMs: OVERLAY_HOLD_MS }
            : undefined,
    )
);
