import { recordSessionVideo } from './session-video';

let recorder: any;
const deferred = () => {
    let resolve!: (value?: any) => void;
    const promise = new Promise<any>(done => { resolve = done; });
    return { promise, resolve };
};
const emit = (values: number[]) => recorder.ondataavailable({ data: {
    size: values.length,
    slice: () => ({ arrayBuffer: async () => Uint8Array.from(values).buffer }),
} });

beforeEach(() => {
    window.sessionVideo = {
        create: jest.fn().mockResolvedValue({ filePath: 'video.mp4', startedAt: 1 }),
        append: jest.fn().mockResolvedValue(undefined), finish: jest.fn().mockResolvedValue(undefined),
        list: jest.fn().mockResolvedValue([]), read: jest.fn(), delete: jest.fn(), reveal: jest.fn(),
    };
    const MockRecorder = jest.fn().mockImplementation(() => {
        recorder = { state: 'inactive', start: jest.fn(() => { recorder.state = 'recording'; }),
            stop: jest.fn(() => { recorder.state = 'inactive'; }) };
        return recorder;
    });
    Object.assign(MockRecorder, { isTypeSupported: jest.fn(() => true) });
    Object.defineProperty(window, 'MediaRecorder', { configurable: true, value: MockRecorder });
});
afterEach(() => { delete window.sessionVideo; Reflect.deleteProperty(window, 'MediaRecorder'); });

it.each(['video/mp4;codecs=avc1', 'video/mp4'])('records using supported MP4 type %s', async (mimeType) => {
    (MediaRecorder.isTypeSupported as jest.Mock).mockImplementation(type => type === mimeType || type === 'video/webm');
    const stream = {} as MediaStream;
    const capture = recordSessionVideo(stream, 'telemetry.jsonl', jest.fn());
    expect(MediaRecorder).toHaveBeenCalledWith(stream, { mimeType, videoBitsPerSecond: 2500000 });
    await Promise.resolve();
    const stopped = capture.stop();
    await Promise.resolve();
    recorder.onstop();
    await stopped;
});

it('reports unsupported MP4 without creating a file or falling back to WebM', () => {
    (MediaRecorder.isTypeSupported as jest.Mock).mockImplementation(type => type.startsWith('video/webm'));
    expect(() => recordSessionVideo({} as MediaStream, 'telemetry.jsonl', jest.fn())).toThrow('MP4 recording is unavailable');
    expect(MediaRecorder).not.toHaveBeenCalled();
    expect(window.sessionVideo!.create).not.toHaveBeenCalled();
});

it('waits for the final media event and all disk writes before completing', async () => {
    const write = deferred();
    (window.sessionVideo!.append as jest.Mock).mockReturnValueOnce(write.promise);
    const capture = recordSessionVideo({} as MediaStream, 'telemetry.jsonl', jest.fn());
    await Promise.resolve();
    emit([1, 2]);
    const stopped = capture.stop();
    let complete = false;
    void stopped.then(() => { complete = true; });
    await Promise.resolve();
    emit([3, 4]);
    recorder.onstop();
    await Promise.resolve();
    expect(complete).toBe(false);
    expect(window.sessionVideo!.finish).not.toHaveBeenCalled();
    write.resolve();
    await stopped;
    expect((window.sessionVideo!.append as jest.Mock).mock.calls.map(([, bytes]) => Array.from(bytes))).toEqual([[1, 2], [3, 4]]);
    expect(window.sessionVideo!.finish).toHaveBeenCalledWith('video.mp4');
});

it('cancels a pending start without starting MediaRecorder', async () => {
    const creating = deferred();
    (window.sessionVideo!.create as jest.Mock).mockReturnValue(creating.promise);
    const capture = recordSessionVideo({} as MediaStream, 'telemetry.jsonl', jest.fn());
    const stopped = capture.stop();
    creating.resolve({ filePath: 'video.mp4' });
    await stopped;
    expect(recorder.start).not.toHaveBeenCalled();
    expect(window.sessionVideo!.finish).toHaveBeenCalledWith('video.mp4');
});

it('surfaces disk errors and releases the file while preserving saved bytes', async () => {
    (window.sessionVideo!.append as jest.Mock).mockRejectedValue(new Error('Disk full'));
    const error = jest.fn();
    const capture = recordSessionVideo({} as MediaStream, 'telemetry.jsonl', error);
    await Promise.resolve();
    emit([1]);
    const stopped = capture.stop();
    recorder.onstop();
    await expect(stopped).rejects.toThrow('Disk full');
    expect(error).toHaveBeenCalledWith(expect.objectContaining({ message: 'Disk full' }));
    expect(window.sessionVideo!.finish).toHaveBeenCalled();
    expect(window.sessionVideo!.delete).not.toHaveBeenCalled();
});
