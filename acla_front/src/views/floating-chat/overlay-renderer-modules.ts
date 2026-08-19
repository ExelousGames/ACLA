import type { AiOverlayRenderer } from './ai-overlay-types';
import { goalOverlayRenderer } from 'components/ai-engineering-tools/Goal';
import { liveRangeTodoListOverlayRenderer } from 'components/ai-engineering-tools/LiveRangeTodoList';
import { procedurePlanOverlayRenderer } from 'components/ai-engineering-tools/ProcedurePlan';
import { driverExpertComparisonOverlayRenderer } from 'components/driver-expert-comparison/DriverExpertComparisonGraph';
import { aiMapToolDisplayOverlayRenderer } from 'views/lap-analysis/ai-chat/AiMapToolDisplay';
import { aiMessageDisplayOverlayRenderer } from 'views/lap-analysis/ai-chat/AiMessageDisplay';
import { toolMessageDisplayOverlayRenderer } from 'views/lap-analysis/ai-chat/ToolMessageDisplay';
import { baselineProgressDisplayOverlayRenderer } from 'views/live-session/BaselineProgressDisplay';

const builtInRenderers: readonly AiOverlayRenderer[] = [
    goalOverlayRenderer,
    liveRangeTodoListOverlayRenderer,
    procedurePlanOverlayRenderer,
    driverExpertComparisonOverlayRenderer,
    aiMapToolDisplayOverlayRenderer,
    aiMessageDisplayOverlayRenderer,
    toolMessageDisplayOverlayRenderer,
    baselineProgressDisplayOverlayRenderer,
];

const renderers = new Map<string, AiOverlayRenderer>();
let builtInsRegistered = false;

export const registerAiOverlayRenderer = (renderer: AiOverlayRenderer): void => {
    if (!renderer.componentType.trim()) throw new Error('Overlay renderer componentType is required.');
    const existing = renderers.get(renderer.componentType);
    if (existing === renderer) return;
    if (existing) {
        throw new Error(`Duplicate overlay renderer componentType '${renderer.componentType}'.`);
    }
    renderers.set(renderer.componentType, renderer);
};

export const registerBuiltInAiOverlayRenderers = (): void => {
    if (builtInsRegistered) return;
    builtInRenderers.forEach(registerAiOverlayRenderer);
    builtInsRegistered = true;
};

export const getAiOverlayRenderer = (componentType: string): AiOverlayRenderer => {
    const renderer = renderers.get(componentType);
    if (!renderer) throw new Error(`Unknown overlay componentType '${componentType}'.`);
    return renderer;
};

export const clearAiOverlayRenderersForTests = (): void => {
    renderers.clear();
    builtInsRegistered = false;
};
