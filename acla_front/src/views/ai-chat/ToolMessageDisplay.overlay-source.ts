import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import { OVERLAY_HOLD_MS } from 'views/floating-chat/ai-overlay-types';
import type { ToolStatusOverlaySnapshot } from './ToolMessageDisplay';

export const createToolStatusOverlayComponent = (componentName: string) => (
    new MutableAiOverlayComponent<ToolStatusOverlaySnapshot>(
        componentName,
        'tool_status',
        (_snapshot, publication) => ({
            placement: 'flow',
            requestedStatus: publication.requestedStatus ?? 'expanded',
            transientDurationMs: OVERLAY_HOLD_MS,
            presentationId: publication.presentationId,
        }),
    )
);
