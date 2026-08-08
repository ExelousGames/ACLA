import { act, renderHook } from '@testing-library/react';
import apiService from 'services/api.service';
import { useVoiceConversation } from '../use-voice-conversation';

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
            { initialProps: { sessionContext: { session_mode: 'recorded', version: 1 } } },
        );

        const socket = await startAndOpen(result);

        expect(mockOpenWebSocket).toHaveBeenCalledWith('/voice/stream', expect.objectContaining({
            session_id: 'telemetry-1',
            client_session_id: 'client-1',
            chat_llm_model: 'openai:gpt-4.1',
            chat_session_action: 'create',
            chat_session_id: undefined,
        }));
        expect(result.current.state).toBe('connecting');
        expect(JSON.parse(socket.send.mock.calls[0][0])).toMatchObject({
            type: 'frontend_info',
            client_session_id: 'client-1',
            session_context: { session_mode: 'recorded', version: 1 },
        });

        expect(result.current.sendUserText('hello')).toBe(false);
        expect(result.current.sendToolStatus({ status: 'working' })).toBe(false);
        expect(result.current.sendToolResult({ id: 'tool-1', name: 'test', result: {} })).toBe(false);
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
            session_context: { session_mode: 'recorded', version: 2 },
        });

        act(() => mockWorkletNodes[0].port.onmessage?.({
            data: { type: 'pcm', buffer: new ArrayBuffer(4) },
        } as MessageEvent));
        expect(socket.send.mock.calls[2][0]).toBeInstanceOf(ArrayBuffer);

        rerender({ sessionContext: { session_mode: 'recorded', version: 3 } });
        expect(JSON.parse(socket.send.mock.calls[3][0])).toEqual({
            type: 'session_context',
            session_context: { session_mode: 'recorded', version: 3 },
        });
    });

    it('retains the server ID after a transport error and disposes old resources before resume', async () => {
        const { result } = renderHook(() => useVoiceConversation({ clientSessionId: 'client-1' }));
        const firstSocket = await startAndOpen(result);
        markReady(firstSocket, 'server-chat-1', false);

        act(() => firstSocket.error());
        expect(result.current.state).toBe('error');

        await act(async () => {
            await result.current.start();
        });
        const secondSocket = mockSockets[1];

        expect(firstSocket.close).toHaveBeenCalledWith(1000, 'connection retry');
        expect(mockStreams[0].tracks[0].stop).toHaveBeenCalled();
        expect(mockAudioContexts[0].close).toHaveBeenCalled();
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

    it('clears conversation identity on explicit stop and on a fresh hook mount', async () => {
        const first = renderHook(() => useVoiceConversation());
        const firstSocket = await startAndOpen(first.result);
        markReady(firstSocket, 'server-chat-1', false);

        act(() => first.result.current.stop());
        expect(first.result.current.state).toBe('idle');
        await act(async () => {
            await first.result.current.start();
        });
        expect(mockOpenWebSocket).toHaveBeenNthCalledWith(2, '/voice/stream', expect.objectContaining({
            chat_session_action: 'create',
            chat_session_id: undefined,
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
