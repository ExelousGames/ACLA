import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import {
    OVERLAY_COMPARISON_COMPLETION_PAUSE_MS,
    OVERLAY_HOLD_MS,
} from 'views/floating-chat/ai-overlay-types';
import { getDriverExpertReplayDurationMs } from './DriverExpertComparisonGraph';
import type { DriverExpertComparisonSnapshot } from './DriverExpertComparisonOverlay';

export const createDriverExpertComparisonOverlayComponent = (componentName: string) => (
    new MutableAiOverlayComponent<DriverExpertComparisonSnapshot>(
        componentName,
        'driver_expert_comparison',
        (snapshot, publication) => ({
            placement: 'flow',
            requestedStatus: publication.requestedStatus ?? 'expanded',
            transientDurationMs: snapshot
                ? Math.max(
                    OVERLAY_HOLD_MS,
                    getDriverExpertReplayDurationMs(snapshot.comparison)
                        + OVERLAY_COMPARISON_COMPLETION_PAUSE_MS,
                )
                : null,
            presentationId: publication.presentationId,
        }),
    )
);
