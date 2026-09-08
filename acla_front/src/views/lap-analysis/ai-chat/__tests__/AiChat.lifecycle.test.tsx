import { asTool, type ToolDispatcher } from 'components/ai-operations/tool';
import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import AiChat from '../ai-chat';
import type { AssistantActiveScreen } from '../../assistant-session-mode';
import {
    createOperation, asWorkflow, isProcedurePlanOptOutRequest, isProcedurePlanStartEvent,
    type OperationExecutionOutput,
} from 'components/ai-operations';
import { createAiCommandRegistry, createWorkflowToolDispatcher } from '../ai-command-registry';

const mockVoiceCleanup = jest.fn();
const mockVoiceStop = jest.fn();
const mockVoiceReset = jest.fn();
const mockVoiceStart = jest.fn(() => Promise.resolve());
const mockVoiceSendUserText = jest.fn(() => false);
const mockSetMicDisabled = jest.fn();
let mockVoiceState = 'idle';
let mockMicDisabled = false;
const mockUseVoiceConversation = jest.fn();
const mockOverlayCreate = jest.fn();
const mockOverlayDestroy = jest.fn<Promise<void>, [string]>(() => Promise.resolve());
const mockOverlaySetEnabled = jest.fn<Promise<void>, [boolean]>(() => Promise.resolve());
const mockFindComponentRef = jest.fn(() => null);
const mockRegisterComponentRef = jest.fn();
const mockUnregisterComponentRef = jest.fn();
const mockComponentDirectory = {
    findComponentRef: mockFindComponentRef,
    registerComponentRef: mockRegisterComponentRef,
    unregisterComponentRef: mockUnregisterComponentRef,
};
const mockGetCircuitMapById = jest.fn(() => Promise.resolve(null));
const mockRepeatablePlanRender = jest.fn();
const mockProcedurePlanRender = jest.fn();
let mockRegisteredAiChatHandle: any;

jest.mock('../use-voice-conversation', () => ({
    useVoiceConversation: (options: Record<string, unknown>) => mockUseVoiceConversation(options),
}));

jest.mock('../ai-command-registry', () => ({
    createAiCommandRegistry: jest.fn(() => ({})),
    createWorkflowToolDispatcher: jest.fn(() => Object.assign(jest.fn(), { validate: jest.fn() })),
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
        getCircuitMapById: mockGetCircuitMapById,
        getCircuitMapByTrack: jest.fn(() => Promise.resolve(null)),
    }),
}));

jest.mock('contexts/OperationComponentRefContext', () => {
    const actual = jest.requireActual('contexts/OperationComponentRefContext');
    return {
        ...actual,
        useOperationComponentRefs: () => ({
            directory: mockComponentDirectory,
            revision: 0,
        }),
        useRegisterOperationComponentRef: (ref: { current: unknown }) => {
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

jest.mock('components/ai-operations', () => {
    const actual = jest.requireActual('components/ai-operations');
    return {
        ...actual,
        RepeatablePlan: (props: unknown) => {
            mockRepeatablePlanRender(props);
            return null;
        },
        ProcedurePlan: (props: unknown) => {
            mockProcedurePlanRender(props);
            return <div data-testid="procedure-plan" />;
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
    create_repeatable_plan: {
        name: 'Lifecycle goal',
        tools: [{ collect: { id: 'collect', title: 'Collect data' } }],
        stop_when: {
            tool: { determine: {} },
            operator: 'eq',
            target: 0,
        },
    },
});

const lifecycleProcedurePlan = () => ({
    set_procedure_plan: {
        goal: 'Lifecycle plan',
        tools: [{ read: { title: 'Read data', arguments: {} } }],
    },
});

const operationWithValue = (value: OperationExecutionOutput) => asTool(createOperation(value, 'complete'));

describe('AiChat conversation lifecycle', () => {
    beforeEach(() => {
        localStorage.clear();
        HTMLElement.prototype.scrollIntoView = jest.fn();
        mockVoiceCleanup.mockClear();
        mockVoiceStop.mockClear();
        mockVoiceReset.mockClear();
        mockGetCircuitMapById.mockReset();
        mockGetCircuitMapById.mockResolvedValue(null);
        mockVoiceStart.mockClear();
        mockVoiceSendUserText.mockReset();
        mockVoiceSendUserText.mockReturnValue(false);
        mockVoiceState = 'idle';
        mockMicDisabled = false;
        mockSetMicDisabled.mockClear();
        mockUseVoiceConversation.mockReset();
        mockUseVoiceConversation.mockImplementation(() => {
            React.useEffect(() => () => mockVoiceCleanup(), []);
            return {
                state: mockVoiceState,
                micDisabled: mockMicDisabled,
                micLevel: 0,
                error: null,
                start: mockVoiceStart,
                stop: mockVoiceStop,
                reset: mockVoiceReset,
                setMicDisabled: mockSetMicDisabled,
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
        mockRepeatablePlanRender.mockClear();
        mockProcedurePlanRender.mockClear();
        mockRegisteredAiChatHandle = undefined;
        (createAiCommandRegistry as jest.Mock).mockReturnValue({});
        (createWorkflowToolDispatcher as jest.Mock).mockImplementation(
            jest.requireActual('../ai-command-registry').createWorkflowToolDispatcher,
        );
        const operations = jest.requireActual('components/ai-operations');
        (isProcedurePlanOptOutRequest as jest.Mock).mockImplementation(operations.isProcedurePlanOptOutRequest);
        (isProcedurePlanStartEvent as jest.Mock).mockImplementation(operations.isProcedurePlanStartEvent);
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

    it('resets on assistant mode and recorded session changes without remounting', () => {
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        const frontDeskClientSessionId = getLatestMainVoiceOptions().clientSessionId;
        const input = screen.getByRole('textbox');

        view.rerender(
            <AiChat
                name="dashboard-assistant"
                activeScreen={frontDeskScreen({ label: 'Maps', componentName: 'session-analysis' })}
            />,
        );
        expect(getLatestMainVoiceOptions().clientSessionId).toBe(frontDeskClientSessionId);
        expect(mockVoiceReset).not.toHaveBeenCalled();

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
        expect(mockVoiceReset).toHaveBeenCalledTimes(4);
        expect(mockVoiceCleanup).not.toHaveBeenCalled();
        expect(screen.getByRole('textbox')).toBe(input);
    });

    it.each(['live', 'front_desk', 'user_summary', 'recorded'] as const)(
        'preserves messages and drafts when only screen metadata changes in %s mode',
        (assistantMode) => {
            const activeScreen = frontDeskScreen({ assistantMode, recordedSessionId: 'session-1' });
            const view = render(<AiChat name="dashboard-assistant" activeScreen={activeScreen} />);
            const options = getLatestMainVoiceOptions();
            const input = screen.getByRole('textbox');
            act(() => options.onEvent({ kind: 'assistant_transcript', text: 'Current response' }));
            fireEvent.change(input, { target: { value: 'Current draft' } });

            view.rerender(<AiChat name="dashboard-assistant" activeScreen={{
                ...activeScreen,
                label: 'Updated label',
                recordedSessionId: assistantMode === 'recorded' ? 'session-1' : 'session-2',
            }} />);

            expect(mockVoiceReset).not.toHaveBeenCalled();
            expect(mockVoiceCleanup).not.toHaveBeenCalled();
            expect(getLatestMainVoiceOptions().clientSessionId).toBe(options.clientSessionId);
            expect(screen.getByText('Current response')).toBeInTheDocument();
            expect(screen.getByRole('textbox')).toBe(input);
            expect(input).toHaveValue('Current draft');
        },
    );

    it.each(['explicit reset', 'mode change', 'recorded session change'])(
        'clears conversation resources after %s and can start again in the same component', async (action) => {
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        if (action === 'recorded session change') {
            view.rerender(<AiChat name="dashboard-assistant" activeScreen={{
                assistantMode: 'recorded', label: 'Race 1', recordedSessionId: 'session-1',
            }} />);
            mockVoiceReset.mockClear();
        }
        const oldOptions = getLatestMainVoiceOptions();
        const input = screen.getByRole('textbox');
        fireEvent.click(screen.getByRole('button', { name: 'Start assistant' }));
        await waitFor(() => expect(mockVoiceStart).toHaveBeenCalledTimes(1));
        act(() => oldOptions.onEvent({ kind: 'assistant_transcript', text: 'Old response' }));
        fireEvent.change(input, { target: { value: 'Unsent draft' } });
        const child = asTool(createOperation(new Promise<OperationExecutionOutput>(() => undefined), 'complete'));
        const abort = jest.spyOn(child, 'abort');
        act(() => {
            const operation = mockRegisteredAiChatHandle.createProcedurePlan(
                lifecycleProcedurePlan(), toolDispatcher(jest.fn(() => child)),
            );
            void operation.result.catch(() => undefined);
        });
        expect(screen.getByTestId('procedure-plan')).toBeInTheDocument();

        if (action === 'explicit reset') {
            act(() => mockRegisteredAiChatHandle.resetSession());
        } else {
            view.rerender(<AiChat name="dashboard-assistant" activeScreen={{
                assistantMode: 'recorded', label: 'Race 2', recordedSessionId: 'session-2',
            }} />);
        }

        expect(mockVoiceReset).toHaveBeenCalledTimes(2);
        expect(mockVoiceCleanup).not.toHaveBeenCalled();
        expect(mockOverlayDestroy).toHaveBeenCalledWith('presentation-default');
        expect(abort).toHaveBeenCalledTimes(1);
        expect(screen.queryByTestId('procedure-plan')).not.toBeInTheDocument();
        expect(screen.queryByText('Old response')).not.toBeInTheDocument();
        expect(screen.getByRole('textbox')).toBe(input);
        expect(input).toHaveValue('');
        const newOptions = getLatestMainVoiceOptions();
        expect(newOptions.clientSessionId).not.toBe(oldOptions.clientSessionId);
        act(() => {
            oldOptions.onEvent({ kind: 'assistant_transcript', text: 'Late response' });
            newOptions.onEvent({ kind: 'assistant_transcript', text: 'Late identified response', clientSessionId: oldOptions.clientSessionId });
            newOptions.onEvent({ kind: 'assistant_transcript', text: 'New response' });
        });
        expect(screen.queryByText('Late response')).not.toBeInTheDocument();
        expect(screen.queryByText('Late identified response')).not.toBeInTheDocument();
        expect(screen.getByText('New response')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Start assistant' }));
        await waitFor(() => expect(mockVoiceStart).toHaveBeenCalledTimes(2));
        act(() => mockRegisteredAiChatHandle.resetSession());
        expect(getLatestMainVoiceOptions().clientSessionId).not.toBe(newOptions.clientSessionId);
        expect(screen.queryByText('New response')).not.toBeInTheDocument();
        view.unmount();
        expect(mockVoiceCleanup).toHaveBeenCalledTimes(2);
        },
    );

    it('clears an active agent and its live runner when switching assistant modes', async () => {
        const view = render(<AiChat name="dashboard-assistant" activeScreen={{ assistantMode: 'live', label: 'Live Session' }} />);
        await act(async () => {
            await mockRegisteredAiChatHandle.startAgentSession('track_guide').result;
        });
        const agentOptions = getLatestAgentVoiceOptions();
        let dispose: jest.SpyInstance;
        act(() => {
            agentOptions.onEvent({ kind: 'assistant_transcript', text: 'Old agent response' });
            dispose = jest.spyOn(mockRegisteredAiChatHandle.initializeLiveRangeTodoList(), 'dispose');
            mockRegisteredAiChatHandle.setLivePerformanceAnalystEnabled(true);
        });
        view.rerender(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        expect(dispose!).toHaveBeenCalledTimes(1);
        expect(getLatestAgentVoiceOptions().clientSessionId).toBeUndefined();
        expect(mockRegisteredAiChatHandle.getLivePerformanceAnalystState().enabled).toBe(false);
        expect(mockRegisteredAiChatHandle.getOpportunityTelemetryRows()).toEqual([]);
        act(() => agentOptions.onEvent({ kind: 'assistant_transcript', text: 'Late agent response' }));
        await act(async () => { await mockRegisteredAiChatHandle.stopAgentSession().result; });
        expect(screen.queryByText('Old agent response')).not.toBeInTheDocument();
        expect(screen.queryByText('Late agent response')).not.toBeInTheDocument();
        expect(mockVoiceCleanup).not.toHaveBeenCalled();
    });

    it.each(['pending', 'in flight'])('ignores a %s map lookup that finishes after a session reset', async (stage) => {
        let resolveMap!: (value: null) => void;
        mockGetCircuitMapById.mockReturnValueOnce(new Promise((resolve) => { resolveMap = resolve; }));
        render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        let operation: any;
        act(() => { operation = mockRegisteredAiChatHandle.showMap({ map_id: 'old-map' }); });
        if (stage === 'in flight') {
            await act(async () => { await Promise.resolve(); });
        }
        expect(mockGetCircuitMapById).toHaveBeenCalledTimes(stage === 'in flight' ? 1 : 0);
        act(() => mockRegisteredAiChatHandle.resetSession());
        await act(async () => {
            resolveMap(null);
            await operation.result;
        });
        expect(screen.queryAllByText('Map is not available')).toHaveLength(0);
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

    it('registers a repeatable plan before dispatch and renders its snapshots', async () => {
        let resolveCollect!: (value: OperationExecutionOutput) => void;
        const collect = new Promise<OperationExecutionOutput>((resolve) => {
            resolveCollect = resolve;
        });
        const dispatch = jest.fn((toolName: string) => {
            expect(mockRegisterComponentRef).toHaveBeenCalledTimes(1);
            expect(mockRegisterComponentRef.mock.calls[0][0].current.getComponentType()).toBe('repeatable-plan');
            if (toolName === 'collect') return asTool(createOperation(collect, 'complete'));
            return operationWithValue({ status: 'ready', data: 0 });
        });
        const { container } = render(
            <AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />,
        );

        let operation: any;
        act(() => {
            operation = mockRegisteredAiChatHandle.createRepeatablePlan(lifecycleGoalRequest(), toolDispatcher(dispatch));
        });
        const toolList = container.querySelector('.ai-chat__tool-list');
        const messages = container.querySelector('.ai-chat__msgs');
        expect(toolList).toBeInTheDocument();
        expect(toolList!.compareDocumentPosition(messages as Node) & Node.DOCUMENT_POSITION_FOLLOWING)
            .toBeTruthy();
        expect(mockRepeatablePlanRender).toHaveBeenLastCalledWith(expect.objectContaining({
            snapshot: expect.objectContaining({ name: 'Lifecycle goal', status: 'running' }),
        }));

        let result: any;
        await act(async () => {
            resolveCollect({ status: 'complete' });
            result = await operation.result;
        });

        expect(result).toMatchObject({ goal: 'Lifecycle goal', status: 'achieved' });
        expect(result).not.toHaveProperty('name');
        expect(mockRepeatablePlanRender.mock.calls.map(([props]) => props.snapshot)).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ name: 'Lifecycle goal', status: 'running' }),
                expect.objectContaining({ name: 'Lifecycle goal', status: 'achieved' }),
            ]),
        );
    });

    it('registers a procedure plan runner before dispatch and renders its snapshots', async () => {
        let resolveRead!: (value: OperationExecutionOutput) => void;
        const read = new Promise<OperationExecutionOutput>((resolve) => {
            resolveRead = resolve;
        });
        const dispatch = jest.fn(() => {
            expect(mockRegisterComponentRef).toHaveBeenCalledTimes(1);
            expect(mockRegisterComponentRef.mock.calls[0][0].current.getComponentType())
                .toBe('procedure_plan');
            return asTool(createOperation(read, 'complete'));
        });
        render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);

        let operation: any;
        act(() => {
            operation = mockRegisteredAiChatHandle
                .createProcedurePlan(lifecycleProcedurePlan(), toolDispatcher(dispatch));
        });
        expect(mockProcedurePlanRender).toHaveBeenLastCalledWith(expect.objectContaining({
            plan: expect.objectContaining({
                goal: 'Lifecycle plan',
                requests: [expect.objectContaining({ status: 'running' })],
            }),
        }));
        expect(screen.getByTestId('procedure-plan')).toBeInTheDocument();

        let result: any;
        await act(async () => {
            resolveRead({ status: 'complete' });
            result = await operation.result;
        });

        expect(result).toMatchObject({ status: 'complete', goal: 'Lifecycle plan' });
        expect(screen.queryByTestId('procedure-plan')).not.toBeInTheDocument();
        expect(mockUnregisterComponentRef).toHaveBeenCalledWith(
            mockRegisterComponentRef.mock.calls[0][0],
        );
        expect(mockProcedurePlanRender.mock.calls.map(([props]) => props.plan)).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    goal: 'Lifecycle plan',
                    requests: [expect.objectContaining({ status: 'running' })],
                }),
            ]),
        );
    });

    it('disposes owned workflow runners on replacement and conversation unmount', () => {
        const never = new Promise<any>(() => undefined);
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        let goalOperation: any;
        act(() => {
            goalOperation = mockRegisteredAiChatHandle.createRepeatablePlan(
                lifecycleGoalRequest(),
                toolDispatcher(jest.fn(() => asTool(createOperation(never, 'complete')))),
            );
        });
        void goalOperation.result.catch(() => undefined);
        const goalRef = mockRegisterComponentRef.mock.calls[0][0];
        const goalDispose = jest.spyOn(goalRef.current, 'dispose');

        let planOperation: any;
        act(() => {
            planOperation = mockRegisteredAiChatHandle.createProcedurePlan(
                lifecycleProcedurePlan(),
                toolDispatcher(jest.fn(() => asTool(createOperation(never, 'complete')))),
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
            goalOperation = mockRegisteredAiChatHandle.createRepeatablePlan(
                lifecycleGoalRequest(),
                toolDispatcher(jest.fn(() => asTool(createOperation(never, 'complete')))),
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

    it.each(['procedure legacy', 'procedure later tool', 'repeatable legacy', 'repeatable stop tool'])(
        'preserves the mounted workflow for rejected %s creation', async (scenario) => {
            const child = asTool(createOperation(new Promise<Record<string, unknown>>(() => undefined), 'complete'));
            const abort = jest.spyOn(child, 'abort');
            const dispatch = toolDispatcher(jest.fn(() => child));
            render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
            let active: any;
            act(() => { active = mockRegisteredAiChatHandle.createProcedurePlan(lifecycleProcedurePlan(), dispatch); });
            void active.result.catch(() => undefined);
            const mounted = mockRegisterComponentRef.mock.calls[0][0].current;
            const dispose = jest.spyOn(mounted, 'dispose');
            const invalidDispatch = Object.assign(jest.fn(() => child), { validate: jest.fn((name: string) => {
                if (name === 'forbidden') throw new Error('Forbidden tool');
            }) });
            const procedureInput = lifecycleProcedurePlan();
            const repeatableInput = lifecycleGoalRequest();
            let rejected: any;
            act(() => {
                if (scenario.startsWith('procedure')) {
                    const input = scenario.endsWith('legacy')
                        ? { goal: 'Legacy', requests: [{ name: 'read', title: 'Read', payload: {} }] }
                        : { set_procedure_plan: { ...procedureInput.set_procedure_plan,
                            tools: [...procedureInput.set_procedure_plan.tools, { forbidden: { title: 'Forbidden', arguments: {} } }],
                        } };
                    rejected = mockRegisteredAiChatHandle.createProcedurePlan(input, invalidDispatch);
                } else {
                    const input = scenario.endsWith('legacy')
                        ? { name: 'Legacy', steps: [], stop_when: {} }
                        : { create_repeatable_plan: { ...repeatableInput.create_repeatable_plan,
                            stop_when: { ...repeatableInput.create_repeatable_plan.stop_when, tool: { forbidden: {} } },
                        } };
                    rejected = mockRegisteredAiChatHandle.createRepeatablePlan(input, invalidDispatch);
                }
            });
            await expect(rejected.result).rejects.toThrow();
            expect(mockRegisterComponentRef).toHaveBeenCalledTimes(1);
            expect(mockUnregisterComponentRef).not.toHaveBeenCalled();
            expect(dispose).not.toHaveBeenCalled();
            expect(abort).not.toHaveBeenCalled();
            expect(invalidDispatch).not.toHaveBeenCalled();
            expect(screen.getByTestId('procedure-plan')).toBeInTheDocument();
        },
    );

    it.each([
        'legacy', 'mixed', 'extra transport field', 'missing active handler',
        'unregistered_tool', 'set_procedure_plan',
        'create_repeatable_plan', 'retry_repeatable_plan_task', 'advance_plan_step',
        'clear_procedure_plan', 'get_live_range_todo_list',
        'add_event_to_live_range_todo_list', 'add_filtered_driver_expert_comparisons_to_live_range_todo_list',
    ])('rejects %s status input before handlers or workflow replacement', (scenario) => {
        const statusHandler = jest.fn(() => operationWithValue({ status: 'complete' }));
        (createAiCommandRegistry as jest.Mock).mockReturnValue({ show_map: statusHandler, [scenario]: statusHandler });
        const child = asTool(createOperation(new Promise<Record<string, unknown>>(() => undefined), 'complete'));
        const abort = jest.spyOn(child, 'abort');
        const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        let active: any;
        act(() => { active = mockRegisteredAiChatHandle.createProcedurePlan(lifecycleProcedurePlan(), toolDispatcher(jest.fn(() => child))); });
        void active.result.catch(() => undefined);
        const mounted = mockRegisterComponentRef.mock.calls[0][0].current;
        const dispose = jest.spyOn(mounted, 'dispose');
        const legacy = { event: 'procedure_plan_started', goal: 'Legacy', requests: [{ name: 'show_map', title: 'Map', payload: {} }] };
        const input = { event: 'procedure_plan_started', set_procedure_plan: {
            goal: 'Status plan', tools: [{ show_map: { title: 'Map', arguments: {} } }],
        } };
        const data = scenario === 'legacy' ? legacy : scenario === 'mixed' ? { ...input, requests: legacy.requests }
            : scenario === 'extra transport field' ? { ...input, current_request: 0 }
                : { ...input, set_procedure_plan: { ...input.set_procedure_plan,
                    tools: [...input.set_procedure_plan.tools, {
                        [scenario === 'missing active handler' ? 'stop_agent_session' : scenario]: { title: 'Later tool', arguments: {} },
                    }],
                } };
        act(() => getLatestMainVoiceOptions().onEvent({ kind: 'tool_status', data }));
        expect(statusHandler).not.toHaveBeenCalled();
        expect(dispose).not.toHaveBeenCalled();
        expect(abort).not.toHaveBeenCalled();
        expect(mockRegisterComponentRef).toHaveBeenCalledTimes(1);
        expect(mockUnregisterComponentRef).not.toHaveBeenCalled();
        expect(screen.getByTestId('procedure-plan')).toBeInTheDocument();
        error.mockRestore();
    });

    it.each(['show_map', 'collect_live_baseline'])(
        'executes %s from a status envelope in a front desk session with literal arguments', async (toolName) => {
            const handler = jest.fn(() => operationWithValue({ status: 'complete' }));
            (createAiCommandRegistry as jest.Mock).mockReturnValue({ [toolName]: handler });
            render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
            const args = { args: { track: 'test' }, parameters: { view: 1 } };
            await act(async () => {
                getLatestMainVoiceOptions().onEvent({ kind: 'tool_status', data: {
                    event: 'procedure_plan_started', set_procedure_plan: {
                        goal: 'Run tool', tools: [{ [toolName]: { title: 'Tool', arguments: args } }],
                    },
                } });
                for (let index = 0; index < 8; index += 1) await Promise.resolve();
            });
            expect(handler).toHaveBeenCalledWith(args, undefined);
            expect(createWorkflowToolDispatcher).toHaveBeenCalledWith(expect.objectContaining({ sessionMode: 'front_desk', conversationRole: 'main' }));
            expect(screen.queryByTestId('procedure-plan')).not.toBeInTheDocument();
        },
    );

    it.each(['generic operation', 'workflow'])('rejects a status handler returning a %s without conversion', async (kind) => {
        const raw = createOperation({}, 'complete');
        const child = kind === 'workflow' ? asWorkflow(raw) : raw;
        const notify = jest.spyOn(child, 'notifyTerminated');
        const handler = jest.fn(() => child);
        const later = jest.fn(() => operationWithValue({}));
        const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        (createAiCommandRegistry as jest.Mock).mockReturnValue({ show_map: handler, stop_agent_session: later });
        render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        await act(async () => {
            getLatestMainVoiceOptions().onEvent({ kind: 'tool_status', data: { set_procedure_plan: {
                goal: 'Show map', tools: [{ show_map: { title: 'Map', arguments: {} } }, { stop_agent_session: { title: 'Stop', arguments: {} } }],
            } } });
            for (let index = 0; index < 8; index += 1) await Promise.resolve();
        });
        expect(handler).toHaveBeenCalledTimes(1);
        expect(later).not.toHaveBeenCalled();
        expect(notify).not.toHaveBeenCalled();
        expect(child).not.toHaveProperty('kind', 'tool');
        expect(error).toHaveBeenCalled();
        error.mockRestore();
    });

    it('keeps procedure opt-out after a malformed start status', async () => {
        const handler = jest.fn(() => operationWithValue({}));
        const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        (createAiCommandRegistry as jest.Mock).mockReturnValue({ show_map: handler });
        render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        const input = { set_procedure_plan: { goal: 'Map', tools: [{ show_map: { title: 'Map', arguments: {} } }] } };
        act(() => {
            const { onEvent } = getLatestMainVoiceOptions();
            onEvent({ kind: 'user_transcript', text: 'Stop the plan.' });
            onEvent({ kind: 'tool_status', data: { event: 'procedure_plan_started', set_procedure_plan: { goal: 'Broken' } } });
            onEvent({ kind: 'tool_status', data: input });
        });
        expect(handler).not.toHaveBeenCalled();
        expect(mockRegisterComponentRef).not.toHaveBeenCalled();
        await act(async () => {
            getLatestMainVoiceOptions().onEvent({ kind: 'tool_status', data: { ...input, event: 'procedure_plan_started' } });
            for (let index = 0; index < 8; index += 1) await Promise.resolve();
        });
        expect(handler).toHaveBeenCalledTimes(1);
        error.mockRestore();
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

        fireEvent.click(screen.getByRole('button', { name: 'Start assistant' }));

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

    it.each(['listening', 'speaking'])('only toggles microphone input while %s', (voiceState) => {
        mockVoiceState = voiceState;
        const view = render(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        fireEvent.click(screen.getByRole('button', { name: 'Disable microphone' }));
        expect(mockSetMicDisabled).toHaveBeenLastCalledWith(true);
        mockMicDisabled = true;
        view.rerender(<AiChat name="dashboard-assistant" activeScreen={frontDeskScreen()} />);
        fireEvent.click(screen.getByRole('button', { name: 'Enable microphone' }));
        expect(mockSetMicDisabled).toHaveBeenLastCalledWith(false);
        expect(mockVoiceStop).not.toHaveBeenCalled();
        expect(mockOverlayDestroy).not.toHaveBeenCalled();
        expect(mockVoiceStart).not.toHaveBeenCalled();
    });

    it.each(['main', 'agent'] as const)(
        'destroys a pending %s overlay after reset without restarting voice', async (target) => {
        let resolveOverlay: (presentation: { presentationId: string }) => void = () => undefined;
        mockOverlayCreate.mockReturnValueOnce(new Promise((resolve) => {
            resolveOverlay = resolve;
        }));
        const view = render(<AiChat name="dashboard-assistant" activeScreen={{
            assistantMode: 'live', label: 'Live Session',
        }} />);

        if (target === 'main') {
            fireEvent.click(screen.getByRole('button', { name: 'Start assistant' }));
        } else {
            await act(async () => {
                await mockRegisteredAiChatHandle.startAgentSession('track_guide').result;
            });
        }
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
        },
    );
});

const toolDispatcher = (dispatch: (...args: any[]) => ReturnType<ToolDispatcher>): ToolDispatcher => Object.assign(dispatch, { validate: jest.fn() }) as ToolDispatcher;
