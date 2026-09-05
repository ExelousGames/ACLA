import { act, renderHook } from '@testing-library/react';
import apiService from 'services/api.service';
import { useVoiceConversation } from '../use-voice-conversation';
import { createAiToolDeferred, createControlledAiToolOperation } from '../ai-tool-base';

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: {
        openWebSocket: jest.fn(),
    },
}));

class MockVoiceWebSocket {
    readyState = 0;
    binaryType: BinaryType = 'blob';
    readonly url = 'ws://voice.test/voice/stream';
    onopen: ((event: Event) => void) | null = null;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;
    send = jest.fn();
    close = jest.fn((code?: number, reason?: string) => {
        this.readyState = 3;
        void code;
        void reason;
    });

    open() {
        this.readyState = 1;
        this.onopen?.(new Event('open'));
    }

    message(payload: object) {
        this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
    }

    error() {
        this.onerror?.(new Event('error'));
    }

    serverClose(code: number, reason: string) {
        this.readyState = 3;
        this.onclose?.({ code, reason } as CloseEvent);
    }
}

interface MockTrack extends MediaStreamTrack {
    stop: jest.Mock;
}

interface MockStream extends MediaStream {
    tracks: MockTrack[];
}

class MockAudioContext {
    readonly sampleRate: number;
    currentTime = 0;
    destination = {} as AudioDestinationNode;
    audioWorklet = {
        addModule: jest.fn().mockResolvedValue(undefined),
    } as unknown as AudioWorklet;
    close = jest.fn().mockResolvedValue(undefined);
    createMediaStreamSource = jest.fn(() => ({ connect: jest.fn() }));
    createBuffer = jest.fn(() => ({
        copyToChannel: jest.fn(),
        duration: 0.01,
    }));
    createBufferSource = jest.fn(() => ({
        buffer: null,
        connect: jest.fn(),
        start: jest.fn(),
        onended: null,
    }));

    constructor(options?: AudioContextOptions) {
        this.sampleRate = options?.sampleRate || 48000;
        mockAudioContexts.push(this);
    }
}

class MockAudioWorkletNode {
    port = { onmessage: null as ((event: MessageEvent) => void) | null };
    disconnect = jest.fn();

    constructor() {
        mockWorkletNodes.push(this);
    }
}

const mockOpenWebSocket = apiService.openWebSocket as jest.MockedFunction<
    typeof apiService.openWebSocket
>;
const mockGetUserMedia = jest.fn();
let mockSockets: MockVoiceWebSocket[] = [];
let mockStreams: MockStream[] = [];
let mockAudioContexts: MockAudioContext[] = [];
let mockWorkletNodes: MockAudioWorkletNode[] = [];

const startAndOpen = async (result: { current: ReturnType<typeof useVoiceConversation> }) => {
    await act(async () => {
        await result.current.start();
    });
    const socket = mockSockets[mockSockets.length - 1];
    act(() => socket.open());
    return socket;
};

const markReady = (
    socket: MockVoiceWebSocket,
    chatSessionId: string,
    resumed: boolean,
) => {
    act(() => socket.message({
        type: 'chat_session_ready',
        chat_session_id: chatSessionId,
        resumed,
    }));
};

describe('useVoiceConversation chat session lifecycle', () => {
    beforeEach(() => {
        mockSockets = [];
        mockStreams = [];
        mockAudioContexts = [];
        mockWorkletNodes = [];
        mockGetUserMedia.mockReset();
        mockGetUserMedia.mockImplementation(async () => {
            const track = {
                enabled: true,
                stop: jest.fn(),
            } as unknown as MockTrack;
            const stream = {
                tracks: [track],
                getAudioTracks: () => [track],
                getTracks: () => [track],
            } as unknown as MockStream;
            mockStreams.push(stream);
            return stream;
        });
        mockOpenWebSocket.mockReset();
        mockOpenWebSocket.mockImplementation(() => {
            const socket = new MockVoiceWebSocket();
            mockSockets.push(socket);
            return socket as unknown as WebSocket;
        });

        Object.defineProperty(globalThis, 'WebSocket', {
            configurable: true,
            value: { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 },
        });
        Object.defineProperty(globalThis, 'AudioContext', {
            configurable: true,
            value: MockAudioContext,
        });
        Object.defineProperty(globalThis, 'AudioWorkletNode', {
            configurable: true,
            value: MockAudioWorkletNode,
        });
        Object.defineProperty(navigator, 'mediaDevices', {
            configurable: true,
            value: { getUserMedia: mockGetUserMedia },
        });
        jest.spyOn(console, 'error').mockImplementation(() => undefined);
        jest.spyOn(console, 'warn').mockImplementation(() => undefined);
        jest.spyOn(console, 'groupCollapsed').mockImplementation(() => undefined);
        jest.spyOn(console, 'groupEnd').mockImplementation(() => undefined);
        jest.spyOn(console, 'log').mockImplementation(() => undefined);
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    it('creates a session and gates traffic until chat_session_ready', async () => {
        const { result, rerender } = renderHook(
            ({ sessionContext }) => useVoiceConversation({
                sessionId: 'telemetry-1',
                chatLlmModel: 'openai:gpt-4.1',
                clientSessionId: 'client-1',
                sessionContext,
            }),
            { initialProps: { sessionContext: { session_mode: 'recorded' as const, version: 1 } } },
        );

        const socket = await startAndOpen(result);

        expect(mockOpenWebSocket).toHaveBeenCalledWith('/voice/stream', expect.objectContaining({
            session_id: 'telemetry-1',
            client_session_id: 'client-1',
            chat_llm_model: 'openai:gpt-4.1',
            chat_session_action: 'create',
            chat_session_id: undefined,
        }));
        const query = mockOpenWebSocket.mock.calls[0][1] as Record<string, unknown>;
        expect(query).not.toHaveProperty('session_mode');
        expect(query).not.toHaveProperty('agent_mode');
        expect(result.current.state).toBe('connecting');
        const sessionInfo = JSON.parse(socket.send.mock.calls[0][0]);
        expect(sessionInfo).toMatchObject({
            type: 'session_info',
            client_session_id: 'client-1',
            session_context: { session_mode: 'recorded' },
        });
        expect(sessionInfo).not.toHaveProperty('session_mode');
        expect(sessionInfo).not.toHaveProperty('agent_mode');
        expect(sessionInfo).not.toHaveProperty('tools');
        expect(sessionInfo).not.toHaveProperty('tool_metadata');
        expect(sessionInfo).not.toHaveProperty('query_scope_schema');
        expect(sessionInfo).not.toHaveProperty('tool_result_handling');
        expect(Object.keys(sessionInfo.session_context)).toEqual(['session_mode']);

        expect(result.current.sendUserText('hello')).toBe(false);
        expect(result.current.sendToolStatus({ status: 'working' })).toBe(false);
        expect(result.current.sendToolResult({ id: 'tool-1', name: 'test', final: true, result: {} })).toBe(false);
        await expect(result.current.executeToolCall({ name: 'test' })).resolves.toBeNull();
        act(() => mockWorkletNodes[0].port.onmessage?.({
            data: { type: 'pcm', buffer: new ArrayBuffer(4) },
        } as MessageEvent));
        rerender({ sessionContext: { session_mode: 'recorded', version: 2 } });
        expect(socket.send).toHaveBeenCalledTimes(1);

        markReady(socket, 'server-chat-1', false);
        expect(result.current.state).toBe('listening');
        expect(result.current.sendUserText('hello')).toBe(true);
        expect(JSON.parse(socket.send.mock.calls[1][0])).toMatchObject({
            type: 'user_text',
            text: 'hello',
            session_context: { session_mode: 'recorded' },
        });

        act(() => mockWorkletNodes[0].port.onmessage?.({
            data: { type: 'pcm', buffer: new ArrayBuffer(4) },
        } as MessageEvent));
        expect(socket.send.mock.calls[2][0]).toBeInstanceOf(ArrayBuffer);

        rerender({ sessionContext: { session_mode: 'recorded', version: 3 } });
        expect(JSON.parse(socket.send.mock.calls[3][0])).toEqual({
            type: 'session_context',
            session_context: { session_mode: 'recorded' },
        });
    });

    it('retains the server ID and immediately disposes resources after a transport error', async () => {
        const { result } = renderHook(() => useVoiceConversation({ clientSessionId: 'client-1' }));
        const firstSocket = await startAndOpen(result);
        expect(JSON.parse(firstSocket.send.mock.calls[0][0])).toMatchObject({
            type: 'session_info',
            session_context: { session_mode: 'live' },
        });
        markReady(firstSocket, 'server-chat-1', false);

        act(() => firstSocket.error());
        expect(result.current.state).toBe('error');
        expect(firstSocket.close).toHaveBeenCalledWith(4001, 'connection failed');
        expect(mockStreams[0].tracks[0].stop).toHaveBeenCalled();
        expect(mockAudioContexts[0].close).toHaveBeenCalled();

        await act(async () => {
            await result.current.start();
        });
        const secondSocket = mockSockets[1];

        expect(firstSocket.close).toHaveBeenCalledTimes(1);
        expect(mockOpenWebSocket).toHaveBeenNthCalledWith(2, '/voice/stream', expect.objectContaining({
            chat_session_action: 'resume',
            chat_session_id: 'server-chat-1',
            client_session_id: 'client-1',
        }));

        act(() => secondSocket.open());
        act(() => firstSocket.message({
            type: 'chat_session_ready',
            chat_session_id: 'late-chat',
            resumed: false,
        }));
        act(() => firstSocket.serverClose(1000, 'late close'));
        expect(result.current.state).toBe('connecting');

        markReady(secondSocket, 'server-chat-1', true);
        expect(result.current.state).toBe('listening');
        act(() => firstSocket.error());
        expect(result.current.state).toBe('listening');
    });

    it('treats every non-client close as retryable and resumes the retained session', async () => {
        const { result } = renderHook(() => useVoiceConversation());
        const firstSocket = await startAndOpen(result);
        markReady(firstSocket, 'server-chat-1', false);

        act(() => firstSocket.serverClose(1000, 'server shutdown'));
        expect(result.current.state).toBe('error');
        expect(result.current.error).toContain('Voice connection closed (1000)');

        await act(async () => {
            await result.current.start();
        });
        expect(mockOpenWebSocket).toHaveBeenNthCalledWith(2, '/voice/stream', expect.objectContaining({
            chat_session_action: 'resume',
            chat_session_id: 'server-chat-1',
        }));
    });

    it('does not let pending microphone setup resurrect an explicitly stopped hook', async () => {
        let resolvePermission: ((stream: MediaStream) => void) | undefined;
        const track = {
            enabled: true,
            stop: jest.fn(),
        } as unknown as MockTrack;
        const stream = {
            tracks: [track],
            getAudioTracks: () => [track],
            getTracks: () => [track],
        } as unknown as MockStream;
        mockGetUserMedia.mockImplementationOnce(() => new Promise<MediaStream>((resolve) => {
            resolvePermission = resolve;
        }));
        const { result } = renderHook(() => useVoiceConversation());
        let startPromise: Promise<void> | undefined;

        act(() => {
            startPromise = result.current.start();
        });
        act(() => result.current.stop());
        await act(async () => {
            resolvePermission?.(stream);
            await startPromise;
        });

        expect(track.stop).toHaveBeenCalled();
        expect(mockOpenWebSocket).not.toHaveBeenCalled();
        expect(result.current.state).toBe('idle');
    });

    it('resumes after system teardown and creates a new identity on a fresh mount', async () => {
        const first = renderHook(() => useVoiceConversation());
        const firstSocket = await startAndOpen(first.result);
        markReady(firstSocket, 'server-chat-1', false);

        act(() => first.result.current.stop());
        expect(first.result.current.state).toBe('idle');
        await act(async () => {
            await first.result.current.start();
        });
        expect(mockOpenWebSocket).toHaveBeenNthCalledWith(2, '/voice/stream', expect.objectContaining({
            chat_session_action: 'resume',
            chat_session_id: 'server-chat-1',
        }));

        first.unmount();
        const remounted = renderHook(() => useVoiceConversation());
        await act(async () => {
            await remounted.result.current.start();
        });
        expect(mockOpenWebSocket).toHaveBeenNthCalledWith(3, '/voice/stream', expect.objectContaining({
            chat_session_action: 'create',
            chat_session_id: undefined,
        }));
    });

    it.each(['close', 'error', 'service error', 'stop', 'unmount'])(
        'immediately aborts all tool entry points on %s and ignores late results',
        async (disconnect) => {
            const onEvent = jest.fn();
            const progress = createAiToolDeferred<{ status: string }>();
            const cleanups = [jest.fn(), jest.fn(), jest.fn()];
            const controls = cleanups.map((cleanup) => createControlledAiToolOperation<
                Record<string, unknown>, { status: string }
            >([progress.promise], cleanup));
            let nextControl = 0;
            const handler = jest.fn(() => controls[nextControl++].operation);
            const { result, unmount } = renderHook(() => useVoiceConversation({
                clientSessionId: 'client-1', toolHandlers: { test_tool: handler }, onEvent,
            }));
            const socket = await startAndOpen(result);
            markReady(socket, 'server-chat-1', false);
            act(() => {
                socket.message({ type: 'tool_call', id: 'relay-1', name: 'test_tool' });
                socket.message({
                    type: 'assistant_transcript', text: '<function=test_tool>{}</function>',
                });
            });
            let directResult: ReturnType<typeof result.current.executeToolCall>;
            act(() => {
                directResult = result.current.executeToolCall({ id: 'direct-1', name: 'test_tool' });
            });
            expect(handler).toHaveBeenCalledTimes(3);
            const sentBeforeDisconnect = socket.send.mock.calls.length;
            act(() => {
                if (disconnect === 'close') socket.serverClose(1000, 'session replaced');
                else if (disconnect === 'error') socket.error();
                else if (disconnect === 'service error') socket.message({
                    type: 'error', error_type: 'ServiceUnavailable', message: 'AI service disconnected',
                });
                else if (disconnect === 'stop') result.current.stop();
                else unmount();
                // Must happen synchronously, before React effects or promises run.
                cleanups.forEach((cleanup) => expect(cleanup).toHaveBeenCalledTimes(1));
                controls.forEach((control) => expect(control.signal.aborted).toBe(true));
            });
            expect(mockStreams[0].tracks[0].stop).toHaveBeenCalled();
            expect(mockWorkletNodes[0].disconnect).toHaveBeenCalled();
            mockAudioContexts.forEach((context) => expect(context.close).toHaveBeenCalled());
            await act(async () => {
                await expect(directResult!).resolves.toMatchObject({ ok: false, message: 'AI tool operation was aborted.' });
                progress.resolve({ status: 'too late' });
                controls.forEach((control) => control.resolve('complete', { value: 'too late' }));
                await Promise.resolve();
            });
            expect(socket.send).toHaveBeenCalledTimes(sentBeforeDisconnect);
            const completed = onEvent.mock.calls.map(([event]) => event)
                .filter((event) => event.kind === 'tool_call' && event.final);
            expect(completed).toHaveLength(3);
            completed.forEach((event) => expect(event).toMatchObject({
                ok: false, clientSessionId: 'client-1', message: 'AI tool operation was aborted.',
            }));
            act(() => socket.message({ type: 'tool_call', id: 'late-call', name: 'test_tool' }));
            expect(handler).toHaveBeenCalledTimes(3);
        },
    );

    it('keeps tools and text chat active while the microphone is disabled', async () => {
        const cleanup = jest.fn();
        const control = createControlledAiToolOperation<Record<string, unknown>>([], cleanup);
        const { result } = renderHook(() => useVoiceConversation({
            toolHandlers: { test_tool: () => control.operation },
        }));
        const socket = await startAndOpen(result);
        markReady(socket, 'server-chat-1', false);
        act(() => socket.message({ type: 'tool_call', id: 'call-1', name: 'test_tool' }));
        act(() => result.current.setMicDisabled(true));
        expect(result.current.state).toBe('listening');
        expect(mockStreams[0].tracks[0].enabled).toBe(false);
        expect(cleanup).not.toHaveBeenCalled();
        expect(socket.close).not.toHaveBeenCalled();
        expect(result.current.sendUserText('Still connected')).toBe(true);
        const sentBeforeAudio = socket.send.mock.calls.length;
        act(() => mockWorkletNodes[0].port.onmessage?.({
            data: { type: 'pcm', buffer: new ArrayBuffer(4) },
        } as MessageEvent));
        expect(socket.send).toHaveBeenCalledTimes(sentBeforeAudio);
        await act(async () => control.resolve('complete', { value: 1 }));
        expect(JSON.parse(socket.send.mock.calls.at(-1)![0])).toMatchObject({
            id: 'call-1', final: true, result: { status: 'complete', value: 1 },
        });
        act(() => result.current.setMicDisabled(false));
        expect(mockStreams[0].tracks[0].enabled).toBe(true);
    });

    it('aborts the old connection on identity changes and isolates a newly created session', async () => {
        const cleanup = jest.fn();
        const control = createControlledAiToolOperation<Record<string, unknown>>([], cleanup);
        const handler = jest.fn(() => control.operation);
        const { result, rerender } = renderHook(({ clientSessionId }) => useVoiceConversation({
            clientSessionId, toolHandlers: { test_tool: handler },
        }), { initialProps: { clientSessionId: 'client-1' } });
        const oldSocket = await startAndOpen(result);
        markReady(oldSocket, 'server-chat-1', false);
        act(() => oldSocket.message({ type: 'tool_call', id: 'call-1', name: 'test_tool' }));
        rerender({ clientSessionId: 'client-2' });
        expect(cleanup).toHaveBeenCalledTimes(1);
        expect(result.current.state).toBe('idle');
        const newSocket = await startAndOpen(result);
        expect(mockOpenWebSocket).toHaveBeenLastCalledWith('/voice/stream', expect.objectContaining({
            client_session_id: 'client-2', chat_session_action: 'create', chat_session_id: undefined,
        }));
        markReady(newSocket, 'server-chat-2', false);
        await act(async () => {
            control.resolve('complete', { value: 'late' });
            oldSocket.error();
        });
        expect(result.current.state).toBe('listening');
        expect(newSocket.send).toHaveBeenCalledTimes(1);
    });

    it('executes fresh tools after resuming without reviving the aborted operation', async () => {
        const controls = [
            createControlledAiToolOperation<Record<string, unknown>>(),
            createControlledAiToolOperation<Record<string, unknown>>(),
        ];
        let index = 0;
        const { result } = renderHook(() => useVoiceConversation({
            toolHandlers: { test_tool: () => controls[index++].operation },
        }));
        const oldSocket = await startAndOpen(result);
        markReady(oldSocket, 'current-session', false);
        act(() => oldSocket.message({ type: 'tool_call', id: 'old', name: 'test_tool' }));
        act(() => oldSocket.serverClose(1006, 'network lost'));
        const resumed = await startAndOpen(result);
        markReady(resumed, 'current-session', true);
        act(() => resumed.message({ type: 'tool_call', id: 'new', name: 'test_tool' }));
        await act(async () => {
            controls[0].resolve('complete', { value: 'old' });
            controls[1].resolve('complete', { value: 'new' });
        });
        expect(controls[0].signal.aborted).toBe(true);
        expect(controls[1].signal.aborted).toBe(false);
        const frames = resumed.send.mock.calls.map(([frame]) => JSON.parse(frame));
        expect(frames.filter((frame) => frame.final)).toEqual([
            expect.objectContaining({ id: 'new', result: { status: 'complete', value: 'new' } }),
        ]);
    });

    it('clears a replaced session identity before the user starts a new main session', async () => {
        const { result } = renderHook(() => useVoiceConversation());
        const socket = await startAndOpen(result);
        markReady(socket, 'replaced-chat', false);
        act(() => socket.message({
            type: 'error', error_type: 'ChatSessionReplaced', message: 'A new main session replaced this session.',
        }));
        expect(result.current.error).toBe('A new main session replaced this session.');
        await startAndOpen(result);
        expect(mockOpenWebSocket).toHaveBeenLastCalledWith('/voice/stream', expect.objectContaining({
            chat_session_action: 'create', chat_session_id: undefined,
        }));
    });

    it('shows ChatSessionNotFound and creates a new session on the next click', async () => {
        const { result } = renderHook(() => useVoiceConversation());
        const firstSocket = await startAndOpen(result);
        markReady(firstSocket, 'stale-chat', false);
        act(() => firstSocket.error());

        const resumeSocket = await startAndOpen(result);
        act(() => resumeSocket.message({
            type: 'error',
            error_type: 'ChatSessionNotFound',
            message: 'The requested chat session no longer exists.',
        }));
        act(() => resumeSocket.serverClose(1008, 'chat session policy violation'));

        expect(result.current.state).toBe('error');
        expect(result.current.error).toBe('The requested chat session no longer exists.');

        await act(async () => {
            await result.current.start();
        });
        expect(mockOpenWebSocket).toHaveBeenNthCalledWith(3, '/voice/stream', expect.objectContaining({
            chat_session_action: 'create',
            chat_session_id: undefined,
        }));
    });

    it('rejects a resume ready frame for a different server session', async () => {
        const { result } = renderHook(() => useVoiceConversation());
        const firstSocket = await startAndOpen(result);
        markReady(firstSocket, 'server-chat-1', false);
        act(() => firstSocket.error());

        const resumeSocket = await startAndOpen(result);
        markReady(resumeSocket, 'different-chat', true);

        expect(result.current.state).toBe('error');
        expect(result.current.error).toContain('unexpected chat session');
        expect(resumeSocket.close).toHaveBeenCalledWith(4002, 'invalid chat session handshake');

        await act(async () => {
            await result.current.start();
        });
        expect(mockOpenWebSocket).toHaveBeenNthCalledWith(3, '/voice/stream', expect.objectContaining({
            chat_session_action: 'resume',
            chat_session_id: 'server-chat-1',
        }));
    });
});
