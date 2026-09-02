import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import AiChat from '../ai-chat';
import type { AssistantActiveScreen } from '../../assistant-session-mode';
import { createAiToolOperation } from 'components/ai-engineering-tools';

const mockVoiceCleanup = jest.fn();
const mockVoiceStop = jest.fn();
const mockVoiceStart = jest.fn(() => Promise.resolve());
const mockVoiceSendUserText = jest.fn(() => false);
const mockUseVoiceConversation = jest.fn();
const mockOverlayCreate = jest.fn();
const mockOverlayDestroy = jest.fn<Promise<void>, [string]>(() => Promise.resolve());
const mockOverlaySetEnabled = jest.fn<Promise<void>, [boolean]>(() => Promise.resolve());
const mockFindComponentRef = jest.fn(() => null);
const mockRegisterComponentRef = jest.fn();
const mockUnregisterComponentRef = jest.fn();
const mockGoalRender = jest.fn();
const mockProcedurePlanRender = jest.fn();
let mockRegisteredAiChatHandle: any;

jest.mock('../use-voice-conversation', () => ({
    useVoiceConversation: (options: Record<string, unknown>) => mockUseVoiceConversation(options),
}));

jest.mock('../ai-command-registry', () => ({
    createAiCommandRegistry: jest.fn(() => ({})),
    startAgentRuntime: jest.fn(() => Promise.resolve({ status: 'started' })),
}));

jest.mock('contexts/AiLabelsContext', () => ({
    useAiLabels: () => ({
        getLabelName: jest.fn(),
        getCategoryLabels: jest.fn(() => []),
        loading: false,
        error: null,
    }),
}));

jest.mock('contexts/UserSummaryContext', () => ({
    useUserSummary: () => ({
        userSummary: {},
        userSummaryLoading: false,
        userSummaryError: '',
    }),
}));

jest.mock('contexts/CircuitMapsContext', () => ({
    useCircuitMaps: () => ({
        getCircuitMapById: jest.fn(() => Promise.resolve(null)),
        getCircuitMapByTrack: jest.fn(() => Promise.resolve(null)),
    }),
}));

jest.mock('contexts/AiToolComponentRefContext', () => {
    const actual = jest.requireActual('contexts/AiToolComponentRefContext');
    return {
        ...actual,
        useAiToolComponentRefs: () => ({
            directory: {
                findComponentRef: mockFindComponentRef,
                registerComponentRef: mockRegisterComponentRef,
                unregisterComponentRef: mockUnregisterComponentRef,
            },
            revision: 0,
        }),
        useRegisterAiToolComponentRef: (ref: { current: unknown }) => {
            mockRegisteredAiChatHandle = ref.current;
        },
    };
});

jest.mock('views/lap-analysis/recording-state', () => {
    const actual = jest.requireActual('views/lap-analysis/recording-state');
    return {
        ...actual,
        isLiveSessionAiAvailable: () => true,
    };
});

jest.mock('components/ai-engineering-tools', () => {
    const actual = jest.requireActual('components/ai-engineering-tools');
    return {
        ...actual,
        Goal: (props: unknown) => {
            mockGoalRender(props);
            return null;
        },
        ProcedurePlan: (props: unknown) => {
            mockProcedurePlanRender(props);
            return null;
        },
        LiveRangeTodoList: () => null,
        LiveRangeTodoListRunner: actual.LiveRangeTodoListRunner,
        isProcedurePlanClearEvent: jest.fn(() => false),
        isProcedurePlanOptOutRequest: jest.fn(() => false),
        isProcedurePlanStartEvent: jest.fn(() => false),
    };
});

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: jest.fn(() => Promise.resolve({ data: {} })) },
}));

jest.mock('utils/environment', () => ({
    detectEnvironment: () => 'web',
}));

jest.mock('views/floating-chat/overlay-display-client', () => ({
    overlaySessionClient: {
        available: () => true,
        current: () => null,
        create: (descriptor: Record<string, unknown>) => mockOverlayCreate(descriptor),
        destroy: (presentationId: string) => mockOverlayDestroy(presentationId),
        setEnabled: (enabled: boolean) => mockOverlaySetEnabled(enabled),
    },
}));

const frontDeskScreen = (overrides: Partial<AssistantActiveScreen> = {}): AssistantActiveScreen => ({
    assistantMode: 'front_desk',
    label: 'Front Desk',
    ...overrides,
});

const getLatestMainVoiceOptions = () => {
    const call = [...mockUseVoiceConversation.mock.calls]
        .reverse()
        .find(([options]) => options.conversationRole === 'main');
    if (!call) throw new Error('Main voice conversation was not rendered.');
    return call[0] as Record<string, any>;
};

const getLatestAgentVoiceOptions = () => {
    const call = [...mockUseVoiceConversation.mock.calls]
        .reverse()
        .find(([options]) => options.conversationRole === 'agent');
    if (!call) throw new Error('Agent voice conversation was not rendered.');
    return call[0] as Record<string, any>;
};

const lifecycleGoalRequest = () => ({
    name: 'Lifecycle goal',
    steps: [{ id: 'collect', title: 'Collect data', name: 'collect' }],
    determination: {
        tool: { name: 'determine' },
        operator: 'eq',
        target: 0,
    },
});

const lifecycleProcedurePlan = () => ({
    goal: 'Lifecycle plan',
    requests: [{ type: 'tool_call', title: 'Read data', name: 'read' }],
});

const operationWithValue = (value: unknown) => createAiToolOperation(value, 'complete');

describe('AiChat conversation lifecycle', () => {
    beforeEach(() => {
        localStorage.clear();
        HTMLElement.prototype.scrollIntoView = jest.fn();
        mockVoiceCleanup.mockClear();
        mockVoiceStop.mockClear();
        mockVoiceStart.mockClear();
        mockVoiceSendUserText.mockReset();
        mockVoiceSendUserText.mockReturnValue(false);
        mockUseVoiceConversation.mockReset();
        mockUseVoiceConversation.mockImplementation(() => {
            React.useEffect(() => () => mockVoiceCleanup(), []);
            return {
                state: 'idle',
                micDisabled: false,
                micLevel: 0,
                error: null,
                start: mockVoiceStart,
                stop: mockVoiceStop,
                setMicDisabled: jest.fn(),
                sendUserText: mockVoiceSendUserText,
                sendToolStatus: jest.fn(() => true),
                sendToolResult: jest.fn(() => true),
            };
        });
        mockOverlayCreate.mockReset();
        mockOverlayCreate.mockResolvedValue({ presentationId: 'presentation-default' });
        mockOverlayDestroy.mockReset();
        mockOverlayDestroy.mockResolvedValue(undefined);
        mockOverlaySetEnabled.mockClear();
        mockRegisterComponentRef.mockClear();
        mockUnregisterComponentRef.mockClear();
        mockFindComponentRef.mockClear();
        mockGoalRender.mockClear();
        mockProcedurePlanRender.mockClear();
        mockRegisteredAiChatHandle = undefined;
        delete (window as any).electronAPI;
    });

    it('omits the transcript label and clock from the assistant panel', () => {
        const { container } = render(
            <AiChat
                name="dashboard-assistant"
                activeScreen={{ assistantMode: 'live', label: 'Live Session' }}
            />,
        );

        expect(screen.queryByText('LIVE TRANSCRIPT')).not.toBeInTheDocument();
        expect(container.querySelector('.ai-chat__transcript-head')).toBeNull();
        expect(container.querySelector('.ai-chat__transcript-time')).toBeNull();
    });

    it('serializes only the canonical mode fields for main and agent contexts', async () => {
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        const mainContext = getLatestMainVoiceOptions().sessionContext;

        expect(mainContext).toEqual({ session_mode: 'front_desk' });

        view.rerender(
            <AiChat
                name="dashboard-assistant"
                activeScreen={{ assistantMode: 'live', label: 'Live Session' }}
            />,
        );
        await act(async () => {
            await mockRegisteredAiChatHandle.startAgentSession('track_guide').result;
        });

        await waitFor(() => {
            expect(getLatestAgentVoiceOptions().sessionContext.agent_mode).toBe('track_guide');
        });
        const agentContext = getLatestAgentVoiceOptions().sessionContext;
        expect(agentContext).toEqual({
            session_mode: 'live',
            agent_mode: 'track_guide',
        });
        expect(getLatestAgentVoiceOptions()).not.toHaveProperty('agentMode');
    });

    it('uses assistant mode and recorded session identity as the remount boundary', () => {
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        const frontDeskClientSessionId = getLatestMainVoiceOptions().clientSessionId;

        view.rerender(
            <AiChat
                name="dashboard-assistant"
                activeScreen={frontDeskScreen({ label: 'Maps', componentName: 'session-analysis' })}
            />,
        );
        expect(getLatestMainVoiceOptions().clientSessionId).toBe(frontDeskClientSessionId);

        view.rerender(
            <AiChat
                name="dashboard-assistant"
                activeScreen={{ assistantMode: 'recorded', label: 'Lap A', recordedSessionId: 'session-a' }}
            />,
        );
        const firstRecordedClientSessionId = getLatestMainVoiceOptions().clientSessionId;
        expect(firstRecordedClientSessionId).not.toBe(frontDeskClientSessionId);

        view.rerender(
            <AiChat
                name="dashboard-assistant"
                activeScreen={{ assistantMode: 'recorded', label: 'Lap A renamed', recordedSessionId: 'session-a' }}
            />,
        );
        expect(getLatestMainVoiceOptions().clientSessionId).toBe(firstRecordedClientSessionId);

        view.rerender(
            <AiChat
                name="dashboard-assistant"
                activeScreen={{ assistantMode: 'recorded', label: 'Lap B', recordedSessionId: 'session-b' }}
            />,
        );
        expect(getLatestMainVoiceOptions().clientSessionId).not.toBe(firstRecordedClientSessionId);
    });

    it('initializes the live range runner on demand and removes it when empty', () => {
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);

        expect(mockRegisterComponentRef).not.toHaveBeenCalled();
        let runner: any;
        act(() => {
            runner = mockRegisteredAiChatHandle.initializeLiveRangeTodoList();
        });
        expect(mockRegisterComponentRef).toHaveBeenCalledTimes(1);
        const runnerRef = mockRegisterComponentRef.mock.calls[0][0];
        expect(runnerRef.current.getComponentName()).toBe('live-range-todo-list');
        act(() => {
            runner.addEvent({
                id: 'completed-event',
                normalized_position: 0.5,
                content: { title: 'Completed event' },
                taskStart: jest.fn(),
            });
            runner.clear();
        });
        expect(mockUnregisterComponentRef).toHaveBeenCalledWith(runnerRef);

        let replacementRunner: any;
        act(() => {
            replacementRunner = mockRegisteredAiChatHandle.initializeLiveRangeTodoList();
        });
        expect(replacementRunner).not.toBe(runner);
        expect(mockRegisterComponentRef).toHaveBeenCalledTimes(2);

        view.unmount();
    });

    it('registers a goal runner before dispatch and renders its snapshots', async () => {
        let resolveCollect!: (value: unknown) => void;
        const collect = new Promise((resolve) => {
            resolveCollect = resolve;
        });
        const dispatch = jest.fn((toolName: string) => {
            expect(mockRegisterComponentRef).toHaveBeenCalledTimes(1);
            expect(mockRegisterComponentRef.mock.calls[0][0].current.getComponentType()).toBe('goal');
            if (toolName === 'collect') return createAiToolOperation(collect, 'complete');
            return operationWithValue({ status: 'ready', data: 0 });
        });
        const { container } = render(
            <AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />,
        );

        let operation: any;
        act(() => {
            operation = mockRegisteredAiChatHandle.createGoal(lifecycleGoalRequest(), dispatch);
        });
        const toolList = container.querySelector('.ai-chat__tool-list');
        const messages = container.querySelector('.ai-chat__msgs');
        expect(toolList).toBeInTheDocument();
        expect(toolList!.compareDocumentPosition(messages as Node) & Node.DOCUMENT_POSITION_FOLLOWING)
            .toBeTruthy();
        expect(mockGoalRender).toHaveBeenLastCalledWith(expect.objectContaining({
            snapshot: expect.objectContaining({ name: 'Lifecycle goal', status: 'running' }),
        }));

        let result: any;
        await act(async () => {
            resolveCollect({ status: 'complete' });
            result = await operation.result;
        });

        expect(result).toMatchObject({ goal: 'Lifecycle goal', status: 'achieved' });
        expect(result).not.toHaveProperty('name');
        expect(mockGoalRender.mock.calls.map(([props]) => props.snapshot)).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ name: 'Lifecycle goal', status: 'running' }),
                expect.objectContaining({ name: 'Lifecycle goal', status: 'achieved' }),
            ]),
        );
    });

    it('registers a procedure plan runner before dispatch and renders its snapshots', async () => {
        let resolveRead!: (value: unknown) => void;
        const read = new Promise((resolve) => {
            resolveRead = resolve;
        });
        const dispatch = jest.fn(() => {
            expect(mockRegisterComponentRef).toHaveBeenCalledTimes(1);
            expect(mockRegisterComponentRef.mock.calls[0][0].current.getComponentType())
                .toBe('procedure_plan');
            return createAiToolOperation(read, 'complete');
        });
        render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);

        let operation: any;
        act(() => {
            operation = mockRegisteredAiChatHandle
                .createProcedurePlan(lifecycleProcedurePlan(), dispatch);
        });
        expect(mockProcedurePlanRender).toHaveBeenLastCalledWith(expect.objectContaining({
            plan: expect.objectContaining({
                goal: 'Lifecycle plan',
                requests: [expect.objectContaining({ status: 'running' })],
            }),
        }));

        let result: any;
        await act(async () => {
            resolveRead({ status: 'complete' });
            result = await operation.result;
        });

        expect(result).toMatchObject({ status: 'complete', goal: 'Lifecycle plan' });
        expect(mockProcedurePlanRender.mock.calls.map(([props]) => props.plan)).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    goal: 'Lifecycle plan',
                    requests: [expect.objectContaining({ status: 'running' })],
                }),
                expect.objectContaining({
                    goal: 'Lifecycle plan',
                    requests: [expect.objectContaining({ status: 'complete' })],
                }),
            ]),
        );
    });

    it('disposes owned workflow runners on replacement and conversation unmount', () => {
        const never = new Promise<any>(() => undefined);
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        let goalOperation: any;
        act(() => {
            goalOperation = mockRegisteredAiChatHandle.createGoal(
                lifecycleGoalRequest(),
                jest.fn(() => createAiToolOperation(never, 'complete')),
            );
        });
        void goalOperation.result.catch(() => undefined);
        const goalRef = mockRegisterComponentRef.mock.calls[0][0];
        const goalDispose = jest.spyOn(goalRef.current, 'dispose');

        let planOperation: any;
        act(() => {
            planOperation = mockRegisteredAiChatHandle.createProcedurePlan(
                lifecycleProcedurePlan(),
                jest.fn(() => createAiToolOperation(never, 'complete')),
            );
        });
        void planOperation.result.catch(() => undefined);
        const planRef = mockRegisterComponentRef.mock.calls[1][0];
        const planDispose = jest.spyOn(planRef.current, 'dispose');

        expect(goalDispose).toHaveBeenCalledTimes(1);
        expect(mockUnregisterComponentRef).toHaveBeenCalledWith(goalRef);
        expect(mockUnregisterComponentRef.mock.invocationCallOrder[0])
            .toBeLessThan(mockRegisterComponentRef.mock.invocationCallOrder[1]);

        view.unmount();

        expect(planDispose).toHaveBeenCalledTimes(1);
        expect(mockUnregisterComponentRef).toHaveBeenCalledWith(planRef);
    });

    it('disposes an owned workflow runner during an agent runtime reset', async () => {
        const never = new Promise<any>(() => undefined);
        render(
            <AiChat
                name="dashboard-assistant"
                activeScreen={{ assistantMode: 'live', label: 'Live Session' }}
            />,
        );
        let goalOperation: any;
        act(() => {
            goalOperation = mockRegisteredAiChatHandle.createGoal(
                lifecycleGoalRequest(),
                jest.fn(() => createAiToolOperation(never, 'complete')),
            );
        });
        void goalOperation.result.catch(() => undefined);
        const goalRef = mockRegisterComponentRef.mock.calls[0][0];
        const goalDispose = jest.spyOn(goalRef.current, 'dispose');

        await act(async () => {
            await mockRegisteredAiChatHandle.startAgentSession('track_guide').result;
        });

        expect(goalDispose).toHaveBeenCalledTimes(1);
        expect(mockUnregisterComponentRef).toHaveBeenCalledWith(goalRef);
    });

    it('starts a conversation when mounted under StrictMode', async () => {
        render(
            <React.StrictMode>
                <AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />
            </React.StrictMode>,
        );

        expect(mockRegisterComponentRef).not.toHaveBeenCalled();
        let activeRunner: any;
        act(() => {
            activeRunner = mockRegisteredAiChatHandle.initializeLiveRangeTodoList();
            activeRunner.addEvent({
                id: 'strict-mode-event',
                normalized_position: 0.5,
                content: { title: 'Strict mode event' },
                taskStart: jest.fn(),
            });
            activeRunner.updateEvents([{
                id: 'strict-mode-event',
                content: { description: 'Updated after StrictMode replay' },
            }]);
        });
        expect(mockRegisterComponentRef).toHaveBeenCalledTimes(1);
        expect(mockUnregisterComponentRef).not.toHaveBeenCalled();
        expect(activeRunner.getSnapshot()?.events).toEqual([
            expect.objectContaining({
                id: 'strict-mode-event',
                content: {
                    title: 'Strict mode event',
                    description: 'Updated after StrictMode replay',
                },
            }),
        ]);

        fireEvent.click(screen.getByRole('button', { name: 'Toggle voice session' }));

        await waitFor(() => expect(mockVoiceStart).toHaveBeenCalledTimes(1));
    });

    it('merges consecutive spoken transcript fragments into one driver bubble', () => {
        jest.useFakeTimers();
        jest.setSystemTime(new Date('2026-08-19T12:00:00'));
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        const onEvent = getLatestMainVoiceOptions().onEvent;

        try {
            act(() => {
                onEvent({ kind: 'user_transcript', text: '  Brake earlier ', source: 'voice' });
            });
            const originalBubble = view.container.querySelector('.ai-chat__msg--driver');
            const originalTimestamp = originalBubble
                ?.querySelector('.ai-chat__msg-stamp')
                ?.textContent;

            jest.setSystemTime(new Date('2026-08-19T13:00:00'));
            act(() => {
                onEvent({ kind: 'user_transcript', text: ' then ease off  ', source: 'voice' });
            });

            const mergedBubble = view.container.querySelector('.ai-chat__msg--driver');
            expect(view.container.querySelectorAll('.ai-chat__msg--driver')).toHaveLength(1);
            expect(screen.getByText('Brake earlier then ease off')).toBeInTheDocument();
            expect(mergedBubble).toBe(originalBubble);
            expect(mergedBubble?.querySelector('.ai-chat__msg-stamp')?.textContent)
                .toBe(originalTimestamp);
        } finally {
            view.unmount();
            jest.useRealTimers();
        }
    });

    it('starts a new driver bubble after an assistant bubble', () => {
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        const onEvent = getLatestMainVoiceOptions().onEvent;

        act(() => {
            onEvent({ kind: 'user_transcript', text: 'First driver turn', source: 'voice' });
            onEvent({ kind: 'assistant_transcript', text: 'Assistant reply' });
            onEvent({ kind: 'user_transcript', text: 'Second driver turn', source: 'voice' });
        });

        expect(view.container.querySelectorAll('.ai-chat__msg--driver')).toHaveLength(2);
        expect(screen.getByText('First driver turn')).toBeInTheDocument();
        expect(screen.getByText('Second driver turn')).toBeInTheDocument();
    });

    it('starts a new driver bubble after a tool bubble', () => {
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        const onEvent = getLatestMainVoiceOptions().onEvent;

        act(() => {
            onEvent({ kind: 'user_transcript', text: 'Check the map', source: 'voice' });
            onEvent({
                kind: 'tool_call',
                runId: 'tool-run-1',
                name: 'show_map',
                title: 'Showing map',
                status: 'completed',
                final: true,
            });
            onEvent({ kind: 'user_transcript', text: 'Now compare laps', source: 'voice' });
        });

        expect(view.container.querySelectorAll('.ai-chat__msg--driver')).toHaveLength(2);
        expect(screen.getByText('Showing map')).toBeInTheDocument();
        expect(screen.getByText('Now compare laps')).toBeInTheDocument();
    });

    it('keeps typed transcript echoes as separate driver bubbles', () => {
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        const onEvent = getLatestMainVoiceOptions().onEvent;

        act(() => {
            onEvent({ kind: 'user_transcript', text: 'Typed first', source: 'typed' });
            onEvent({ kind: 'user_transcript', text: 'Typed second', source: 'typed' });
        });

        expect(view.container.querySelectorAll('.ai-chat__msg--driver')).toHaveLength(2);
        expect(screen.getByText('Typed first')).toBeInTheDocument();
        expect(screen.getByText('Typed second')).toBeInTheDocument();
    });

    it('shows a typed message immediately and deduplicates its backend echo', () => {
        mockVoiceSendUserText.mockReturnValue(true);
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);

        fireEvent.change(screen.getByPlaceholderText('Ask the front desk.'), {
            target: { value: 'Show my latest lap' },
        });
        fireEvent.click(screen.getByTitle('Send'));

        expect(mockVoiceSendUserText).toHaveBeenCalledWith('Show my latest lap');
        expect(screen.getByText('Show my latest lap')).toBeInTheDocument();
        expect(view.container.querySelectorAll('.ai-chat__msg--driver')).toHaveLength(1);
        expect(screen.getByPlaceholderText('Ask the front desk.')).toHaveValue('');

        act(() => {
            getLatestMainVoiceOptions().onEvent({
                kind: 'user_transcript',
                text: 'Show my latest lap',
                source: 'typed',
            });
        });

        expect(view.container.querySelectorAll('.ai-chat__msg--driver')).toHaveLength(1);
    });

    it('keeps main and agent transcript merging isolated', async () => {
        render(
            <AiChat
                name="dashboard-assistant"
                activeScreen={{ assistantMode: 'live', label: 'Live Session' }}
            />,
        );

        act(() => {
            getLatestMainVoiceOptions().onEvent({
                kind: 'user_transcript',
                text: 'Main fragment one',
                source: 'voice',
            });
        });
        await act(async () => {
            await mockRegisteredAiChatHandle.startAgentSession('track_guide').result;
        });
        act(() => {
            getLatestMainVoiceOptions().onEvent({
                kind: 'user_transcript',
                text: 'Main fragment two',
                source: 'voice',
            });
            getLatestAgentVoiceOptions().onEvent({
                kind: 'user_transcript',
                text: 'Agent fragment one',
                source: 'voice',
            });
            getLatestAgentVoiceOptions().onEvent({
                kind: 'user_transcript',
                text: 'Agent fragment two',
                source: 'voice',
            });
        });

        expect(screen.getByText('Agent fragment one Agent fragment two')).toBeInTheDocument();
        expect(screen.queryByText('Main fragment one Main fragment two')).not.toBeInTheDocument();

        await act(async () => {
            await mockRegisteredAiChatHandle.stopAgentSession().result;
        });

        expect(screen.getByText('Main fragment one Main fragment two')).toBeInTheDocument();
        expect(screen.queryByText('Agent fragment one Agent fragment two')).not.toBeInTheDocument();
    });

    it('preserves shell preferences across automatic resets', async () => {
        const idleGif = 'data:image/gif;base64,R0lGODlhAQABAAAAACw=';
        localStorage.setItem('acla-emotion-gifs', JSON.stringify({ idle: idleGif }));
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);

        fireEvent.change(screen.getByRole('combobox', { name: 'Chat LLM model' }), {
            target: { value: 'hosted:qwen/qwen3-32b' },
        });
        fireEvent.click(screen.getByRole('button', { name: 'Debug' }));
        fireEvent.click(screen.getByRole('button', { name: 'Overlay Off' }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Overlay On' })).toBeInTheDocument());

        view.rerender(
            <AiChat
                name="dashboard-assistant"
                activeScreen={{ assistantMode: 'user_summary', label: 'User Summary' }}
            />,
        );
        expect(screen.getByRole('combobox', { name: 'Chat LLM model' })).toHaveValue('hosted:qwen/qwen3-32b');
        expect(screen.getByRole('button', { name: 'Debug' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByRole('button', { name: 'Overlay On' })).toHaveAttribute('aria-pressed', 'true');
        fireEvent.click(screen.getByRole('button', { name: 'Emotes' }));
        expect(screen.getByRole('img', { name: 'idle' })).toHaveAttribute('src', idleGif);
    });

    it('destroys an overlay whose asynchronous creation finishes after an identity reset', async () => {
        let resolveOverlay: (presentation: { presentationId: string }) => void = () => undefined;
        mockOverlayCreate.mockReturnValueOnce(new Promise((resolve) => {
            resolveOverlay = resolve;
        }));
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);

        fireEvent.click(screen.getByRole('button', { name: 'Toggle voice session' }));
        expect(mockOverlayCreate).toHaveBeenCalledTimes(1);
        view.rerender(
            <AiChat
                name="dashboard-assistant"
                activeScreen={{ assistantMode: 'user_summary', label: 'User Summary' }}
            />,
        );

        await act(async () => {
            resolveOverlay({ presentationId: 'late-presentation' });
            await Promise.resolve();
        });

        expect(mockOverlayDestroy).toHaveBeenCalledWith('late-presentation');
        expect(mockVoiceStart).not.toHaveBeenCalled();
    });
});
