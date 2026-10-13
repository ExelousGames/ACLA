import { RacingSessionController } from './racing-session.controller';
import { RacingSessionService } from './racing-session.service';
import { BadRequestException, ForbiddenException } from '@nestjs/common';
import { ObjectId } from 'mongodb';
import { Readable } from 'stream';
import { GRIDFS_BUCKETS } from '../gridfs/gridfs.service';

const req = { user: { userId: 'user-1' } };
const metadata = { sessionName: 'Race', mapName: 'Track', carName: 'Car', userId: 'user-1', game_recorded_from: 'acc',
    videos: [{ name: 'video.mp4', mimeType: 'video/mp4', size: 4, startedAt: 1 }] };
let controller: RacingSessionController;
let service: any;

beforeEach(() => {
    jest.useFakeTimers();
    service = {
        listSessionsWithVideos: jest.fn(),
        uploadSessionVideoChunk: jest.fn().mockResolvedValue(new ObjectId()),
        uploadSessionChunk: jest.fn().mockResolvedValue(new ObjectId()),
        createRacingSessionFromChunks: jest.fn().mockResolvedValue({ _id: 'session-1' }),
        deleteUploadFiles: jest.fn().mockResolvedValue(undefined),
    };
    controller = new RacingSessionController(service, {} as any, {} as any, {} as any);
});
afterEach(() => { jest.clearAllTimers(); jest.useRealTimers(); });

it('lists sessions with videos for the authenticated user, ignoring supplied user IDs', async () => {
    const result = { list: [{ sessionId: 'session-1', created_date: new Date('2026-10-13T12:00:00.000Z') }] };
    service.listSessionsWithVideos.mockResolvedValue(result);
    await expect(controller.listSessionsWithVideos({ ...req, query: { userId: 'user-2' } })).resolves.toEqual(result);
    expect(service.listSessionsWithVideos).toHaveBeenCalledWith('user-1');
});

it.each([{}, { user: {} }])('rejects listing sessions with videos without an authenticated user ID: %p', async (request) => {
    await expect(controller.listSessionsWithVideos(request)).rejects.toBeInstanceOf(ForbiddenException);
    expect(service.listSessionsWithVideos).not.toHaveBeenCalled();
});

it('rejects multiple videos before creating an upload', async () => {
    await expect(controller.initUpload({ ...metadata, videos: [metadata.videos[0], {
        ...metadata.videos[0], name: 'second.mp4',
    }] }, req)).rejects.toThrow('Only one video is allowed per session');
    expect((controller as any).uploadStates.size).toBe(0);
    expect(service.uploadSessionVideoChunk).not.toHaveBeenCalled();
});

it('rejects chunks for a second video in a session', async () => {
    const { uploadId } = await controller.initUpload(metadata, req);
    await expect(controller.receiveVideoChunk(req, { videoIndex: 1, chunkIndex: 0, data: 'AQIDBA==' }, uploadId))
        .rejects.toBeInstanceOf(BadRequestException);
    expect(service.uploadSessionVideoChunk).not.toHaveBeenCalled();
});

it('persists the video on the same session, ignores duplicate chunks, and refuses incomplete videos', async () => {
    const { uploadId } = await controller.initUpload(metadata, req);
    await controller.receiveChunk({ chunk: [{ speed: 100 }], chunkIndex: 0 }, uploadId);
    await expect(controller.completeUpload({}, uploadId, req)).rejects.toBeInstanceOf(BadRequestException);
    const chunk = { videoIndex: 0, chunkIndex: 0, data: Buffer.from([1, 2]).toString('base64') };
    await controller.receiveVideoChunk(req, chunk, uploadId);
    await controller.receiveVideoChunk(req, chunk, uploadId);
    expect(service.uploadSessionVideoChunk).toHaveBeenCalledTimes(1);
    expect(service.uploadSessionVideoChunk).toHaveBeenCalledWith(Buffer.from([1, 2]),
        { uploadId, userId: 'user-1', videoIndex: 0, chunkIndex: 0 });
    await controller.receiveVideoChunk(req, { ...chunk, chunkIndex: 1, data: Buffer.from([3, 4]).toString('base64') }, uploadId);
    await expect(controller.completeUpload({}, uploadId, req)).resolves.toMatchObject({ sessionId: 'session-1' });
    expect(service.createRacingSessionFromChunks).toHaveBeenCalledWith('Race', 'Track', 'Car', 'user-1', 'acc',
        [expect.any(ObjectId)], 1, 1000, [{ ...metadata.videos[0], chunkFileIds: [expect.any(ObjectId), expect.any(ObjectId)] }]);
});

it('enforces ownership and bounded video chunks', async () => {
    const other = { user: { userId: 'user-2' } };
    await expect(controller.initUpload(metadata, other)).rejects.toBeInstanceOf(ForbiddenException);
    const { uploadId } = await controller.initUpload(metadata, req);
    const chunk = { videoIndex: 0, chunkIndex: 0, data: 'AQIDBA==' };
    await expect(controller.receiveVideoChunk(other, chunk, uploadId)).rejects.toBeInstanceOf(ForbiddenException);
    await expect(controller.completeUpload({}, uploadId, other)).rejects.toBeInstanceOf(ForbiddenException);
    await expect(controller.receiveVideoChunk(req, { ...chunk, data: 'not base64!' }, uploadId)).rejects.toBeInstanceOf(BadRequestException);
    await expect(controller.receiveVideoChunk(req, { ...chunk, data: Buffer.alloc(5).toString('base64') }, uploadId)).rejects.toBeInstanceOf(BadRequestException);
    await expect(controller.receiveVideoChunk(req, { ...chunk, chunkIndex: 1 }, uploadId)).rejects.toBeInstanceOf(BadRequestException);
    expect(service.uploadSessionVideoChunk).not.toHaveBeenCalled();
});

it('cleans up incomplete uploads without removing the local recording', async () => {
    const { uploadId } = await controller.initUpload(metadata, req);
    await controller.receiveVideoChunk(req, { videoIndex: 0, chunkIndex: 0, data: 'AQIDBA==' }, uploadId);
    await controller.cancelUpload(req, uploadId);
    expect(service.deleteUploadFiles).toHaveBeenCalledWith([expect.any(ObjectId)]);
    await expect(controller.completeUpload({}, uploadId, req)).rejects.toThrow('expired');
});

it('streams video chunks in order and filters access by the session owner', async () => {
    const ids = [new ObjectId(), new ObjectId()];
    const model = { findOne: jest.fn().mockReturnValue({ select: () => ({ exec: async () => ({
        videos: [{ ...metadata.videos[0], chunkFileIds: ids }],
    }) }) }) };
    const gridfs = { downloadStream: jest.fn().mockResolvedValueOnce(Readable.from([Buffer.from([1, 2])]))
        .mockResolvedValueOnce(Readable.from([Buffer.from([3, 4])])) };
    const sessions = new RacingSessionService(model as any, gridfs as any);
    const sessionId = new ObjectId().toString();
    const video = await sessions.downloadSessionVideo('user-1', sessionId, 0);
    const chunks: Buffer[] = [];
    for await (const chunk of video.stream) chunks.push(chunk);
    expect(Buffer.concat(chunks)).toEqual(Buffer.from([1, 2, 3, 4]));
    expect(model.findOne).toHaveBeenCalledWith({ _id: sessionId, user_id: 'user-1' });
    await expect(sessions.downloadSessionVideo('user-1', sessionId, 5)).rejects.toThrow('Video not found');
});

it.each(['video/webm', 'video/avi'])('rejects %s uploads', async (mimeType) => {
    await expect(controller.initUpload({ ...metadata, videos: [{ ...metadata.videos[0],
        mimeType }] }, req)).rejects.toBeInstanceOf(BadRequestException);
});

it('stores MP4 video chunks with the matching MIME type and extension', async () => {
    const gridfs = { uploadFile: jest.fn().mockResolvedValue(new ObjectId()) };
    const sessions = new RacingSessionService({} as any, gridfs as any);
    const chunk = Buffer.from([1, 2]);
    const info = { userId: 'user-1', uploadId: 'upload-1', videoIndex: 0, chunkIndex: 1 };
    await sessions.uploadSessionVideoChunk(chunk, info);
    expect(gridfs.uploadFile).toHaveBeenCalledWith(chunk, 'upload-1-video-0-1.mp4',
        { ...info, contentType: 'video/mp4' }, GRIDFS_BUCKETS.RACING_SESSIONS);
});

it('downloads MP4 recordings with matching response headers', async () => {
    const stream = { on: jest.fn(), pipe: jest.fn(), destroy: jest.fn() };
    service.downloadSessionVideo = jest.fn().mockResolvedValue({ mimeType: 'video/mp4', size: 4, stream });
    const response = { setHeader: jest.fn(), on: jest.fn(), destroy: jest.fn() };
    await controller.downloadVideo(req, 'session-1', '0', response as any);
    expect(response.setHeader).toHaveBeenCalledWith('Content-Type', 'video/mp4');
    expect(response.setHeader).toHaveBeenCalledWith('Content-Disposition', 'attachment; filename="session-video-1.mp4"');
    expect(stream.pipe).toHaveBeenCalledWith(response);
});

it('only lists MP4 recordings while preserving their download indexes', async () => {
    service.listSessionVideos = jest.fn().mockResolvedValue([
        { ...metadata.videos[0], name: 'legacy.webm', mimeType: 'video/webm' },
        metadata.videos[0],
    ]);
    await expect(controller.listVideos(req, 'session-1')).resolves.toEqual([{ index: 1, ...metadata.videos[0] }]);
});

it('rejects WebM downloads while keeping the original MP4 download index', async () => {
    const id = new ObjectId();
    const model = { findOne: jest.fn().mockReturnValue({ select: () => ({ exec: async () => ({
        videos: [
            { ...metadata.videos[0], name: 'legacy.webm', mimeType: 'video/webm', chunkFileIds: [new ObjectId()] },
            { ...metadata.videos[0], chunkFileIds: [id] },
        ],
    }) }) }) };
    const gridfs = { downloadStream: jest.fn().mockResolvedValue(Readable.from([Buffer.from([1, 2, 3, 4])])) };
    const sessions = new RacingSessionService(model as any, gridfs as any);
    const sessionId = new ObjectId().toString();
    await expect(sessions.downloadSessionVideo('user-1', sessionId, 0)).rejects.toThrow('Video not found');
    expect(gridfs.downloadStream).not.toHaveBeenCalled();
    const video = await sessions.downloadSessionVideo('user-1', sessionId, 1);
    const chunks: Buffer[] = [];
    for await (const chunk of video.stream) chunks.push(chunk);
    expect(Buffer.concat(chunks)).toEqual(Buffer.from([1, 2, 3, 4]));
    expect(gridfs.downloadStream).toHaveBeenCalledWith(id, GRIDFS_BUCKETS.RACING_SESSIONS);
});
