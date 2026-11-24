import { ReadableStream, WritableStream } from 'stream/web';
import { recordSessionVideo } from './session-video';

function deferred<T = void>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>(done => { resolve = done; });
    return { promise, resolve };
}

const mockStart = jest.fn();
const mockFinalize = jest.fn();
const mockCancel = jest.fn();
const mockAdd = jest.fn();
const mockAddVideoTrack = jest.fn();
const mockMp4Format = jest.fn();
let mockTarget: WritableStream<Uint8Array>;

jest.mock('mediabunny', () => ({
    AppendOnlyStreamTarget: class {
        constructor(target: typeof mockTarget) { mockTarget = target; }
    },
    Mp4OutputFormat: class {
        constructor(options: unknown) { mockMp4Format(options); }
    },
    Output: class {
        start = mockStart;
        finalize = mockFinalize;
        cancel = mockCancel;
        addVideoTrack = mockAddVideoTrack;
    },
    VideoSampleSource: class { add = mockAdd; },
    VideoSample: class {
        constructor(frame: VideoFrame, timing: { timestamp: number; duration: number }) {
            Object.assign(this, { close: () => frame.close(), ...timing });
        }
    },
}));

let sourceReady: ReturnType<typeof deferred<void>>;
let sourceController: ReadableStreamDefaultController<VideoFrame>;
const mockTrack = { getSettings: () => ({ frameRate: 20 }) };
const stream = { getVideoTracks: () => [mockTrack] } as unknown as MediaStream;

function frame(timestamp: number, duration?: number) {
    return { timestamp, duration, close: jest.fn() } as unknown as VideoFrame;
}

function capture(onError = jest.fn()) {
    return recordSessionVideo(stream, 'telemetry.jsonl', performance.timeOrigin + 1000, onError);
}

beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(performance, 'now').mockReturnValue(1250);
    mockStart.mockResolvedValue(undefined);
    mockFinalize.mockResolvedValue(undefined);
    mockCancel.mockResolvedValue(undefined);
    mockAdd.mockResolvedValue(undefined);
    window.sessionVideo = {
        create: jest.fn().mockResolvedValue({ filePath: 'video.mp4', startedAt: 1 }),
        append: jest.fn().mockResolvedValue(undefined), finish: jest.fn().mockResolvedValue(undefined),
        list: jest.fn().mockResolvedValue([]), read: jest.fn(), delete: jest.fn(), reveal: jest.fn(),
    };

    sourceReady = deferred();
    Object.defineProperty(window, 'WritableStream', { configurable: true, value: WritableStream });
    Object.defineProperty(window, 'VideoEncoder', { configurable: true, value: jest.fn() });
    Object.defineProperty(window, 'MediaStreamTrackProcessor', { configurable: true, value: jest.fn(() => ({
        readable: new ReadableStream<VideoFrame>({ start(controller) {
            sourceController = controller as unknown as ReadableStreamDefaultController<VideoFrame>;
            sourceReady.resolve();
        } }),
    })) });
});

afterEach(() => {
    delete window.sessionVideo;
    for (const name of ['MediaStreamTrackProcessor', 'VideoEncoder', 'WritableStream']) Reflect.deleteProperty(window, name);
    jest.restoreAllMocks();
});

it('encodes the actual capture frame times relative to the first frame and saves its telemetry time', async () => {
    const first = frame(2200000);
    const second = frame(2265000, 42000);
    const pending = capture();
    await sourceReady.promise;
    sourceController.enqueue(first);
    sourceController.enqueue(second);
    sourceController.close();
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    await pending.stop();

    expect(MediaStreamTrackProcessor).toHaveBeenCalledWith({ track: mockTrack, maxBufferSize: 8 });
    expect(mockMp4Format).toHaveBeenCalledWith({ fastStart: 'fragmented', minimumFragmentDuration: 1 });
    expect(mockAdd).toHaveBeenNthCalledWith(1, expect.objectContaining({ timestamp: 0, duration: 0.05 }));
    expect(mockAdd).toHaveBeenNthCalledWith(2, expect.objectContaining({ timestamp: 0.065, duration: 0.042 }));
    expect(first.close).toHaveBeenCalledTimes(1);
    expect(second.close).toHaveBeenCalledTimes(1);
    expect(window.sessionVideo!.finish).toHaveBeenCalledWith('video.mp4', 0.25);
});

it('reports unavailable recording without creating a file', () => {
    Reflect.deleteProperty(window, 'VideoEncoder');
    expect(() => capture()).toThrow('Local video recording is unavailable');
    expect(window.sessionVideo!.create).not.toHaveBeenCalled();
});

it('waits for pending MP4 bytes before saving synchronization and splits writes at the bridge limit', async () => {
    const adding = deferred();
    const writing = deferred();
    const append = window.sessionVideo!.append as jest.Mock;
    append.mockImplementationOnce(() => { writing.resolve(); return appending.promise; });
    const appending = deferred();
    mockAdd.mockImplementationOnce(() => { adding.resolve(); return Promise.resolve(); });
    mockFinalize.mockImplementationOnce(async () => {
        const writer = mockTarget.getWriter();
        await writer.write(new Uint8Array(512 * 1024 + 5));
        await writer.close();
    });
    const pending = capture();
    await sourceReady.promise;
    sourceController.enqueue(frame(10000));
    await adding.promise;
    const stopping = pending.stop();
    await writing.promise;
    expect(window.sessionVideo!.finish).not.toHaveBeenCalled();
    appending.resolve();
    await stopping;

    expect(append.mock.calls.map(([, bytes]) => bytes.byteLength)).toEqual([512 * 1024, 5]);
    expect(window.sessionVideo!.finish).toHaveBeenCalledWith('video.mp4', 0.25);
});

it('cancels a pending start without opening the encoder or writing sync metadata', async () => {
    const creating = deferred<{ filePath: string }>();
    const requested = deferred();
    (window.sessionVideo!.create as jest.Mock).mockImplementationOnce(() => {
        requested.resolve();
        return creating.promise;
    });
    const pending = capture();
    await requested.promise;
    const stopping = pending.stop();
    creating.resolve({ filePath: 'video.mp4' });
    await stopping;

    expect(MediaStreamTrackProcessor).not.toHaveBeenCalled();
    expect(mockStart).not.toHaveBeenCalled();
    expect(window.sessionVideo!.finish).toHaveBeenCalledWith('video.mp4');
});

it('finishes an empty recording without writing sync metadata', async () => {
    const pending = capture();
    await sourceReady.promise;
    await pending.stop();

    expect(mockStart).not.toHaveBeenCalled();
    expect(mockFinalize).not.toHaveBeenCalled();
    expect(mockCancel).toHaveBeenCalledTimes(1);
    expect(window.sessionVideo!.finish).toHaveBeenCalledWith('video.mp4');
});

it('surfaces errors and releases the file while preserving saved bytes', async () => {
    const error = jest.fn();
    const adding = deferred();
    mockAdd.mockImplementationOnce(() => { adding.resolve(); return Promise.resolve(); });
    mockFinalize.mockRejectedValueOnce(new Error('Disk full'));
    const pending = capture(error);
    await sourceReady.promise;
    sourceController.enqueue(frame(10000));
    await adding.promise;
    await expect(pending.stop()).rejects.toThrow('Disk full');

    expect(error).toHaveBeenCalledWith(expect.objectContaining({ message: 'Disk full' }));
    expect(window.sessionVideo!.finish).toHaveBeenCalledWith('video.mp4');
    expect(window.sessionVideo!.delete).not.toHaveBeenCalled();
});
