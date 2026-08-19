import React from 'react';
import { act, render, screen } from '@testing-library/react';
import FloatingChat from './FloatingChat';
import { registerAiOverlayRenderer } from './overlay-renderer-modules';
import { aiMessageDisplayOverlayRenderer } from 'views/lap-analysis/ai-chat/AiMessageDisplay';
import { toolMessageDisplayOverlayRenderer } from 'views/lap-analysis/ai-chat/ToolMessageDisplay';
import type {
    AiOverlayPresentationAcknowledgement,
    AiOverlayPresentationSnapshot,
    AiOverlayRendererEvent,
} from './ai-overlay-types';

let presentationListener: ((presentation: AiOverlayPresentationSnapshot) => void) | null = null;
const acknowledgements: AiOverlayPresentationAcknowledgement[] = [];
const rendererEvents: AiOverlayRendererEvent[] = [];
const resizeFloatingChat = jest.fn();

const session = {
    presentationId: 'presentation-1',
    aiSessionId: 'ai-1',
    mode: 'live' as const,
    displayIdentity: { name: 'Kestrel', emotion: 'idle', agentTags: ['Live'] },
};

const presentation = (
    cards: AiOverlayPresentationSnapshot['cards'],
    presentationRevision = 1,
): AiOverlayPresentationSnapshot => ({
    presentationId: session.presentationId,
    presentationRevision,
    session,
    cards,
});

const card = (
    componentName: string,
    componentType: string,
    snapshot: unknown,
    options: Partial<AiOverlayPresentationSnapshot['cards'][number]> = {},
): AiOverlayPresentationSnapshot['cards'][number] => ({
    componentName,
    componentType,
    snapshot,
    revision: 1,
    metadata: {},
    status: 'expanded',
    placement: 'flow',
    shellSlot: 'card',
    ...options,
});

beforeAll(() => {
    registerAiOverlayRenderer(aiMessageDisplayOverlayRenderer);
    registerAiOverlayRenderer(toolMessageDisplayOverlayRenderer);
});

beforeEach(() => {
    jest.useFakeTimers();
    presentationListener = null;
    acknowledgements.length = 0;
    rendererEvents.length = 0;
    resizeFloatingChat.mockReset();
    (window as any).ResizeObserver = class {
        observe() {}
        disconnect() {}
    };
    (window as any).electronAPI = {
        onOverlayPresentation: (listener: typeof presentationListener) => {
            presentationListener = listener;
            return () => { presentationListener = null; };
        },
        acknowledgeOverlayPresentation: (ack: AiOverlayPresentationAcknowledgement) => {
            acknowledgements.push(ack);
        },
        emitOverlayRendererEvent: (event: AiOverlayRendererEvent) => rendererEvents.push(event),
        reportOverlayReady: jest.fn(),
        resizeFloatingChat,
    };
});

afterEach(() => {
    jest.useRealTimers();
});

describe('FloatingChat presentation renderer', () => {
    it('does not show a readiness label while idle', () => {
        render(<FloatingChat />);

        expect(screen.queryByText('Overlay ready')).not.toBeInTheDocument();
    });

    it('renders the manager-supplied order and preserves card DOM classes and text', () => {
        const { container } = render(<FloatingChat />);
        act(() => presentationListener?.(presentation([
            card('tool:second', 'tool_status', {
                runId: 'second', name: 'query', title: 'Second tool', status: 'completed', ok: true,
            }),
            card('tool:first', 'tool_status', {
                runId: 'first', name: 'query', title: 'First tool', status: 'started',
            }),
        ])));

        const items = Array.from(container.querySelectorAll('.overlay-list-item'));
        expect(items).toHaveLength(2);
        expect(items[0]).toHaveAttribute('data-component-name', 'tool:second');
        expect(items[1]).toHaveAttribute('data-component-name', 'tool:first');
        expect(screen.getByText('Second tool')).toBeInTheDocument();
        expect(screen.getByText('First tool')).toBeInTheDocument();
        expect(acknowledgements.at(-1)).toMatchObject({ accepted: true });
    });

    it('uses manager-selected folded and full-size statuses without local arbitration', () => {
        registerAiOverlayRenderer({
            componentType: 'comparison-test',
            validateSnapshot: (value): value is { title: string } => Boolean((value as any)?.title),
            renderOverlay: (value: any, status) => <div>{`${status}:${value.title}`}</div>,
            dimensions: {
                expanded: { width: 500, height: 300 },
                folded: { width: 320, height: 58 },
                full_size: { width: 760, height: 500 },
            },
        });
        const { container } = render(<FloatingChat />);
        act(() => presentationListener?.(presentation([
            card('folded-card', 'comparison-test', { title: 'Summary' }, { status: 'folded' }),
            card('full-card', 'comparison-test', { title: 'Graph' }, { status: 'full_size' }),
        ])));

        expect(container.querySelector('[data-component-name="folded-card"]'))
            .toHaveClass('overlay-list-item--folded');
        expect(container.querySelector('[data-component-name="full-card"]'))
            .toHaveClass('overlay-list-item--full-size-active');
        expect(container.querySelector('.overlay-shell'))
            .toHaveAttribute('data-full-size-component-name', 'full-card');
        expect(screen.getByText('folded:Summary')).toBeInTheDocument();
        expect(screen.getByText('full_size:Graph')).toBeInTheDocument();
        expect(resizeFloatingChat).toHaveBeenCalledWith(760, 500);
    });

    it('finishes a speaking animation once across manager presentation updates', () => {
        render(<FloatingChat />);
        const messageCard = card('message:session', 'ai_message', { text: 'Hello driver' }, {
                revision: 4,
                shellSlot: 'speech',
                metadata: { name: 'Engineer', emotion: 'vibing', agentTags: ['Race'] },
            });
        act(() => presentationListener?.(presentation([messageCard])));
        expect(screen.getByText('Engineer')).toBeInTheDocument();
        expect(screen.getByText('Race')).toBeInTheDocument();

        act(() => jest.advanceTimersByTime(28 * 'Hello driver'.length));
        expect(screen.getByText('Hello driver')).toBeInTheDocument();
        expect(rendererEvents).toEqual([{
            presentationId: session.presentationId,
            componentName: 'message:session',
            revision: 4,
            event: 'visual_complete',
        }]);

        act(() => presentationListener?.(presentation([messageCard], 2)));
        act(() => jest.advanceTimersByTime(28 * 'Hello driver'.length));

        expect(screen.getByText('Hello driver')).toBeInTheDocument();
        expect(rendererEvents).toHaveLength(1);
    });

    it('rejects unknown component types, invalid snapshots, and stale presentations', () => {
        render(<FloatingChat />);
        act(() => presentationListener?.(presentation([])));
        act(() => presentationListener?.(presentation([], 1)));
        expect(acknowledgements.at(-1)).toMatchObject({
            accepted: false,
            error: 'Stale overlay presentation revision.',
        });

        act(() => presentationListener?.(presentation([
            card('unknown', 'not-registered', { value: true }),
        ], 2)));
        expect(acknowledgements.at(-1)?.error).toContain('Unknown overlay componentType');

        act(() => presentationListener?.(presentation([
            card('bad-tool', 'tool_status', { title: 'missing required data' }),
        ], 3)));
        expect(acknowledgements.at(-1)?.error).toContain('Invalid snapshot');
    });
});
