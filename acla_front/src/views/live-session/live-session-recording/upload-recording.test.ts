import apiService from 'services/api.service';
import { uploadRecording } from './upload-recording';

jest.mock('services/api.service', () => ({ __esModule: true, default: { post: jest.fn(), delete: jest.fn() } }));
const post = apiService.post as jest.Mock;
const video = { filePath: 'local.mp4', name: 'local.mp4', mimeType: 'video/mp4', size: 4, startedAt: 1 };
const options = () => ({
    metadata: { sessionName: 'Race', mapName: 'Track', carName: 'Car', userId: 'user', game_recorded_from: 'acc' as const },
    filePath: 'telemetry.jsonl', videos: [video], onProgress: jest.fn(),
    stream: jest.fn(async (onChunk) => { await onChunk([{ speed: 100 }]); return { rowCount: 1, totalBytes: 10 }; }),
});
beforeEach(() => {
    jest.clearAllMocks();
    (apiService.delete as jest.Mock).mockResolvedValue({});
    post.mockImplementation(async (url) => ({ data: url.endsWith('/init') ? { uploadId: 'upload-1' } : {} }));
    window.sessionVideo = { create: jest.fn(), append: jest.fn(), finish: jest.fn(), list: jest.fn(),
        read: jest.fn().mockResolvedValue({ data: 'AQIDBA==', bytesRead: 4 }), delete: jest.fn(), reveal: jest.fn() };
});
afterEach(() => { delete window.sessionVideo; jest.useRealTimers(); });

it('uploads every video clip before completing the same telemetry session', async () => {
    const upload = options();
    upload.videos.push({ ...video, filePath: 'second.mp4', name: 'second.mp4' });
    await uploadRecording(upload);
    expect(post.mock.calls.map(([url]) => url)).toEqual([
        '/racing-session/upload/init', '/racing-session/upload/chunk?uploadId=upload-1',
        '/racing-session/upload/video-chunk?uploadId=upload-1', '/racing-session/upload/video-chunk?uploadId=upload-1',
        '/racing-session/upload/complete?uploadId=upload-1',
    ]);
    expect(post.mock.calls[0][1].videos).toEqual(upload.videos.map(({ filePath, ...metadata }) => metadata));
    expect(post.mock.calls[3][1]).toEqual({ videoIndex: 1, chunkIndex: 0, data: 'AQIDBA==' });
    expect(window.sessionVideo!.delete).not.toHaveBeenCalled();
});

it('retains local files and never completes a session when the video cannot be read', async () => {
    (window.sessionVideo!.read as jest.Mock).mockResolvedValue({ data: '', bytesRead: 0 });
    await expect(uploadRecording(options())).rejects.toThrow('local video changed');
    expect(post.mock.calls.some(([url]) => url.includes('/complete'))).toBe(false);
    expect(window.sessionVideo!.delete).not.toHaveBeenCalled();
});

it('retries an interrupted video chunk with the same index', async () => {
    jest.useFakeTimers();
    let attempts = 0;
    post.mockImplementation(async (url) => {
        if (url.includes('video-chunk') && attempts++ === 0) throw new Error('Connection lost');
        return { data: { uploadId: 'upload-1' } };
    });
    const uploading = uploadRecording(options());
    for (let i = 0; i < 20; i++) await Promise.resolve();
    jest.runOnlyPendingTimers();
    await uploading;
    const calls = post.mock.calls.filter(([url]) => url.includes('video-chunk'));
    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual(calls[1]);
});
