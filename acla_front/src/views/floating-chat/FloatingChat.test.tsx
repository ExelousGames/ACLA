import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
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

const bounds = (width: number, height: number): DOMRect => ({
    width,
    height,
    top: 0,
    right: width,
    bottom: height,
    left: 0,
    x: 0,
    y: 0,
    toJSON: () => ({}),
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
        expect(items[0]).toHaveClass('overlay-list-item--deck');
        expect(items[1]).toHaveClass('overlay-list-item--deck');
        expect(items[0]).toHaveStyle({ zIndex: 1 });
        expect(items[1]).toHaveStyle({ zIndex: 2 });
        expect(items[0]).toHaveAttribute('data-component-name', 'tool:second');
        expect(items[1]).toHaveAttribute('data-component-name', 'tool:first');
        expect(screen.getByText('Second tool')).toBeInTheDocument();
        expect(screen.getByText('First tool')).toBeInTheDocument();
        expect(acknowledgements.at(-1)).toMatchObject({ accepted: true });
    });

    it('keeps the message, focused card, and expanded siblings inside a card stack', () => {
        registerAiOverlayRenderer({
            componentType: 'comparison-test',
            validateSnapshot: (value): value is { title: string } => Boolean((value as any)?.title),
            renderOverlay: (value: any, status) => (
                <button type="button">{`${status}:${value.title}`}</button>
            ),
            dimensions: {
                expanded: { width: 500, height: 300 },
                folded: { width: 320, height: 58 },
                focus: { width: 760, height: 500 },
            },
        });
        const { container } = render(<FloatingChat />);
        const shell = container.querySelector('.overlay-shell') as HTMLElement;
        jest.spyOn(shell, 'getBoundingClientRect').mockReturnValue(bounds(760, 704));
        act(() => presentationListener?.(presentation([
            card('message:session', 'ai_message', { text: 'Deck introduction' }, {
                shellSlot: 'speech',
            }),
            card('summary-card', 'comparison-test', { title: 'Summary' }),
            card('focus-card', 'comparison-test', { title: 'Graph' }, { status: 'focus' }),
            card('details-card', 'comparison-test', { title: 'Details' }),
        ])));

        const speaking = container.querySelector('.overlay-shell__speaking') as HTMLElement;
        const deck = container.querySelector('.overlay-display-list') as HTMLElement;
        const items = Array.from(deck.querySelectorAll<HTMLElement>('.overlay-list-item'));
        const summary = items[0];
        const focused = items[2];

        expect(speaking.compareDocumentPosition(deck) & Node.DOCUMENT_POSITION_FOLLOWING)
            .toBeTruthy();
        expect(items.map((item) => item.dataset.componentName)).toEqual([
            'summary-card',
            'details-card',
            'focus-card',
        ]);
        expect(deck).toHaveClass('overlay-card-stack');
        expect(deck).toHaveAttribute('aria-label', 'Overlay card stack');
        expect(summary).toHaveClass('overlay-list-item--deck', 'overlay-list-item--stacked');
        expect(summary).not.toHaveClass('overlay-list-item--folded');
        expect(items[1]).toHaveClass('overlay-list-item--deck', 'overlay-list-item--stacked');
        expect(items[1]).not.toHaveClass('overlay-list-item--folded');
        expect(focused).toHaveClass(
            'overlay-list-item--deck',
            'overlay-list-item--stacked',
            'overlay-list-item--focus-active',
        );
        items.forEach((item) => expect(item).not.toHaveAttribute('aria-hidden'));
        expect(summary.style.getPropertyValue('--overlay-stack-position')).toBe('0');
        expect(focused.style.getPropertyValue('--overlay-stack-position')).toBe('2');
        expect(shell).toHaveStyle({ width: '760px' });
        expect(shell.style.height).toBe('');
        expect(focused).not.toHaveStyle({ height: '500px' });
        expect(focused).not.toHaveStyle({ flexBasis: '500px' });
        expect(focused).toHaveAttribute('data-focus-active', 'true');
        expect(focused).toHaveAttribute('data-renderer-width', '760');
        expect(focused).toHaveAttribute('data-renderer-height', '500');
        expect(screen.getByTestId('overlay-ai-message')).toBeInTheDocument();
        expect(screen.getByText('expanded:Summary')).toBeInTheDocument();
        expect(screen.getByText('focus:Graph')).toBeInTheDocument();
        expect(screen.getByText('expanded:Details')).toBeInTheDocument();
        expect(resizeFloatingChat).toHaveBeenLastCalledWith(760, 704);

        fireEvent.mouseOver(summary);
        fireEvent.focus(screen.getByRole('button', { name: 'expanded:Summary' }));
        expect(Array.from(deck.querySelectorAll<HTMLElement>('.overlay-list-item')))
            .toEqual(items);
    });

    it('keeps overlay chrome and sibling cards stacked in focus mode without speech', () => {
        const { container } = render(<FloatingChat />);
        act(() => presentationListener?.(presentation([
            card('focus-card', 'comparison-test', { title: 'Graph' }, { status: 'focus' }),
            card('baseline-card', 'comparison-test', { title: 'Baseline' }),
        ])));

        const shell = container.querySelector('.overlay-shell') as HTMLElement;
        const stacked = container.querySelector(
            '[data-component-name="baseline-card"]',
        ) as HTMLElement;

        expect(shell.querySelector('.overlay-shell__header')).toBeVisible();
        expect(screen.getByText('Kestrel')).toBeVisible();
        expect(screen.getByText('focus:Graph')).toBeVisible();
        expect(screen.getByText('expanded:Baseline')).toBeVisible();
        expect(stacked.closest('.overlay-card-stack')).toBeInTheDocument();
        expect(stacked).toHaveClass('overlay-list-item--stacked');
        expect(stacked).not.toHaveClass('overlay-list-item--folded');
        expect(stacked).not.toHaveAttribute('aria-hidden');
    });

    it('keeps overlay chrome visible for standalone map visualizations', () => {
        const { container } = render(<FloatingChat />);
        act(() => presentationListener?.(presentation([
            card('map:track', 'map', {
                status: 'unavailable',
                title: 'Track map',
                reason: 'No map loaded',
            }),
        ])));

        const shell = container.querySelector('.overlay-shell') as HTMLElement;
        expect(shell.querySelector('.overlay-shell__header')).toBeVisible();
        expect(screen.getByText('Kestrel')).toBeVisible();
        expect(container.querySelector('[data-component-name="map:track"]'))
            .toBeInTheDocument();
    });

    it('keeps overlay chrome visible when the presentation contains only cards', () => {
        const { container } = render(<FloatingChat />);
        act(() => presentationListener?.(presentation([
            card('tool:surface', 'tool_status', {
                runId: 'surface', name: 'query', title: 'Card-owned surface', status: 'completed', ok: true,
            }),
        ])));

        const shell = container.querySelector('.overlay-shell') as HTMLElement;
        expect(shell.querySelector('.overlay-shell__header')).toBeVisible();
        expect(screen.getByText('Kestrel')).toBeVisible();
        expect(container.querySelector('.overlay-list-item'))
            .toContainElement(screen.getByText('Card-owned surface'));
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
        expect(screen.queryByText('Race')).not.toBeInTheDocument();
        expect(screen.queryByText('Live')).not.toBeInTheDocument();

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
