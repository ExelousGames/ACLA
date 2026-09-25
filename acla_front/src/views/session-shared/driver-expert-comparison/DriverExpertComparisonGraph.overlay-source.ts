import { MutableAiOverlayComponent } from 'views/floating-chat/MutableAiOverlayComponent';
import type { DriverExpertComparisonSnapshot } from './DriverExpertComparisonOverlay';
import { synthesizeTtsPack, type TtsPack } from 'components/tts';
import type { AiOverlayRendererEvent } from 'views/floating-chat/ai-overlay-types';

export const getDriverExpertComparisonNarration = (snapshot: DriverExpertComparisonSnapshot): string => (
    [snapshot.title.replace(/(?:^|:\s*)Driver vs Expert$/i, '').trim(), ...(snapshot.labelGroups ?? []).filter((group) => group.subLabels.length > 0)
        .map((group) => `${group.category === 'mistakes' ? 'Mistakes' : group.category === 'expert' ? 'Expert' : 'Recovery'}: ${group.subLabels.join(', ')}`)]
        .filter(Boolean)
        .join('. ')
);

export const prepareDriverExpertComparisonVoices = async (
    snapshots: readonly DriverExpertComparisonSnapshot[],
    signal?: AbortSignal,
): Promise<Array<TtsPack | undefined>> => {
    const narrations = snapshots.map(getDriverExpertComparisonNarration);
    const voices = await synthesizeTtsPack(narrations.filter(Boolean).map((text) => ({ text })), signal);
    let voiceIndex = 0;
    return narrations.map((narration) => narration ? voices[voiceIndex++] : undefined);
};

export const createDriverExpertComparisonOverlayComponent = (
    componentName: string,
    onRendererEvent: (event: AiOverlayRendererEvent) => void = () => undefined,
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
        onRendererEvent,
    )
);
