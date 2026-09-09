import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import type { DriverExpertComparisonSnapshot } from './DriverExpertComparisonOverlay';
import { synthesizeTtsPack } from 'components/tts';

export const getDriverExpertComparisonNarration = (snapshot: DriverExpertComparisonSnapshot): string => (
    [snapshot.title, ...(snapshot.labelGroups ?? []).filter((group) => group.subLabels.length > 0)
        .map((group) => `${group.category === 'mistakes' ? 'Mistakes' : group.category === 'expert' ? 'Expert' : 'Recovery'}: ${group.subLabels.join(', ')}`)]
        .join('. ')
);

export const prepareDriverExpertComparisonVoices = async (
    snapshots: readonly DriverExpertComparisonSnapshot[],
    signal?: AbortSignal,
): Promise<DriverExpertComparisonSnapshot[]> => {
    const voices = await synthesizeTtsPack(snapshots.map((snapshot) => ({
        text: getDriverExpertComparisonNarration(snapshot),
    })), signal);
    return snapshots.map((snapshot, index) => ({ ...snapshot, voice: voices[index] }));
};

export const createDriverExpertComparisonOverlayComponent = (
    componentName: string,
    onRendererEvent: (event: string) => void = () => undefined,
) => (
    new MutableAiOverlayComponent<DriverExpertComparisonSnapshot>(
        componentName,
        'driver_expert_comparison',
        (_snapshot, publication) => ({
            placement: 'flow',
            requestedStatus: publication.requestedStatus ?? 'focus',
            transientDurationMs: null,
            presentationId: publication.presentationId,
        }),
        (event) => onRendererEvent(event.event),
    )
);
