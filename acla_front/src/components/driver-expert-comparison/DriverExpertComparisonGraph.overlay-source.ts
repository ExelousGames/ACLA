import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import type { DriverExpertComparisonSnapshot } from './DriverExpertComparisonOverlay';

export const createDriverExpertComparisonOverlayComponent = (
    componentName: string,
    onRendererEvent: (event: string) => void = () => undefined,
) => (
    new MutableAiOverlayComponent<DriverExpertComparisonSnapshot>(
        componentName,
        'driver_expert_comparison',
        (_snapshot, publication) => ({
            placement: 'flow',
            requestedStatus: publication.requestedStatus ?? 'expanded',
            transientDurationMs: null,
            presentationId: publication.presentationId,
        }),
        (event) => onRendererEvent(event.event),
    )
);
