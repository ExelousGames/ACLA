import { act, renderHook, waitFor } from '@testing-library/react';
import { recordSessionVideo } from './session-video';
import { useSessionVideo } from './use-session-video';

jest.mock('./session-video', () => ({ recordSessionVideo: jest.fn() }));

const stream = {} as MediaStream;
const otherStream = {} as MediaStream;
const initialProps = { stream: null as MediaStream | null, filePath: null as string | null, recording: false };
const mockRecord = recordSessionVideo as jest.Mock;
let stop: jest.Mock;

beforeEach(() => {
    mockRecord.mockReset();
    stop = jest.fn().mockResolvedValue(undefined);
    mockRecord.mockReturnValue({ stop });
});

it('starts video with telemetry only when a screen was shared before recording', async () => {
    const { result, rerender } = renderHook(({ stream, filePath, recording }) => useSessionVideo(stream, filePath, recording), {
        initialProps: { ...initialProps, stream },
    });
    await act(async () => undefined);
    expect(mockRecord).not.toHaveBeenCalled();

    rerender({ stream, filePath: 'session.jsonl', recording: true });
    await act(async () => undefined);
    await waitFor(() => expect(result.current.saving).toBe(true));
    expect(mockRecord).toHaveBeenCalledTimes(1);
    expect(mockRecord).toHaveBeenCalledWith(stream, 'session.jsonl', expect.any(Function));

    rerender({ stream, filePath: 'session.jsonl', recording: false });
    await waitFor(() => expect(result.current.saving).toBe(false));
    expect(stop).toHaveBeenCalledTimes(1);
});

it('keeps a telemetry-only session without video when screen sharing starts later', async () => {
    const { rerender } = renderHook(({ stream, filePath, recording }) => useSessionVideo(stream, filePath, recording), { initialProps });
    rerender({ stream: null, filePath: 'session.jsonl', recording: true });
    await act(async () => undefined);
    rerender({ stream, filePath: 'session.jsonl', recording: true });
    await act(async () => undefined);
    expect(mockRecord).not.toHaveBeenCalled();
});

it('does not restart video after sharing stops, the source changes, or telemetry resumes', async () => {
    const { rerender } = renderHook(({ stream, filePath, recording }) => useSessionVideo(stream, filePath, recording), {
        initialProps,
    });
    rerender({ stream, filePath: 'session.jsonl', recording: true });
    await waitFor(() => expect(mockRecord).toHaveBeenCalledTimes(1));
    rerender({ stream: null, filePath: 'session.jsonl', recording: true });
    await waitFor(() => expect(stop).toHaveBeenCalledTimes(1));
    rerender({ stream: otherStream, filePath: 'session.jsonl', recording: true });
    await act(async () => undefined);
    rerender({ stream: otherStream, filePath: 'session.jsonl', recording: false });
    await act(async () => undefined);
    rerender({ stream: otherStream, filePath: 'resumed.jsonl', recording: true });
    await act(async () => undefined);
    expect(mockRecord).toHaveBeenCalledTimes(1);
});

it.each([false, true])('does not start video for an existing session (recording: %s)', async (recording) => {
    const { rerender } = renderHook(({ stream, filePath, recording }) => useSessionVideo(stream, filePath, recording), {
        initialProps: { stream, filePath: 'restored.jsonl', recording },
    });
    await act(async () => undefined);
    rerender({ stream, filePath: 'restored.jsonl', recording: true });
    await act(async () => undefined);
    expect(mockRecord).not.toHaveBeenCalled();
});

it('does not retry failed video recording during the same telemetry session', async () => {
    mockRecord.mockImplementationOnce(() => { throw new Error('Video unavailable'); });
    const { result, rerender } = renderHook(({ stream, filePath, recording }) => useSessionVideo(stream, filePath, recording), {
        initialProps: { ...initialProps, stream },
    });
    rerender({ stream, filePath: 'session.jsonl', recording: true });
    await waitFor(() => expect(result.current.error).toBe('Video unavailable'));
    expect(result.current.saving).toBe(false);
    rerender({ stream: otherStream, filePath: 'session.jsonl', recording: true });
    await act(async () => undefined);
    expect(mockRecord).toHaveBeenCalledTimes(1);
});
