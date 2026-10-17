import fixtures from './fixtures/top-lap-reference.json';
import type { loadTopLapReferenceModel as LoadModel } from '../topLapReferenceService';

jest.mock('../api.service', () => ({ __esModule: true, default: { get: jest.fn() } }));

let loadModel: typeof LoadModel;
let get: jest.Mock;
let chunks: Buffer[];
let totalSize: number;
const prepareUrl = '/ai-model/active/top_lap_reference/prepare-chunked';

const serveModel = (payload: unknown) => {
    const bytes = Buffer.from(JSON.stringify(payload), 'utf8');
    totalSize = bytes.length;
    // Split inside a UTF-8 character as well as across JSON and base64 values.
    const split = Math.max(1, bytes.indexOf(Buffer.from('é')) + 1);
    chunks = [bytes.subarray(0, split)];
    for (let index = split; index < bytes.length; index += 256) chunks.push(bytes.subarray(index, index + 256));
    get.mockImplementation(async (url: string) => {
        if (url === prepareUrl) return { data: {
            success: true, chunking: { sessionId: 'model-session', totalChunks: chunks.length, totalSize },
        } };
        const index = Number(url.split('/').pop());
        return { data: { success: true, sessionId: 'model-session', chunkIndex: index,
            totalChunks: chunks.length, data: chunks[index].toString('base64') } };
    });
};

beforeEach(() => {
    jest.resetModules();
    loadModel = require('../topLapReferenceService').loadTopLapReferenceModel;
    get = require('../api.service').default.get;
});

it('downloads every chunk of the entire model and shares it between callers', async () => {
    serveModel({ top_lap_store: { 'référence': fixtures.protocol5F, other: fixtures.iracing } });
    const first = loadModel();
    expect(loadModel()).toBe(first);
    const entries = await first;
    expect(entries).toEqual([fixtures.expected, { ...fixtures.expected, game: 'iracing', track: 'spa - layout', car: 'GT3', avg_grip_int: 4 }]);
    expect(get).toHaveBeenCalledTimes(chunks.length + 1);
    chunks.forEach((_, index) => expect(get).toHaveBeenCalledWith(`/ai-model/active/chunked-data/model-session/${index}`));
    expect(await loadModel()).toBe(entries);
    expect(get).toHaveBeenCalledTimes(chunks.length + 1);
});

it('discards incomplete downloads and starts a new session on retry', async () => {
    serveModel({ top_lap_store: { lap: fixtures.protocol4F } });
    totalSize += 1;
    await expect(loadModel()).rejects.toThrow('Incomplete');
    totalSize -= 1;
    await expect(loadModel()).resolves.toEqual([fixtures.expected]);
    expect(get.mock.calls.filter(([url]) => url === prepareUrl)).toHaveLength(2);
});

it('rejects failed or mismatched chunk responses even when HTTP succeeds', async () => {
    serveModel({ top_lap_store: { lap: fixtures.protocol5C } });
    const serve = get.getMockImplementation()!;
    get.mockImplementation(async (url: string) => {
        const result = await serve(url);
        return url === prepareUrl ? result : { data: { ...result.data, chunkIndex: 999 } };
    });
    await expect(loadModel()).rejects.toThrow('Incomplete');
});

it('reports an unavailable active model', async () => {
    get.mockResolvedValue({ data: { success: false, message: 'No active model found' } });
    await expect(loadModel()).rejects.toThrow('No active model found');
    expect(get).toHaveBeenCalledTimes(1);
});

it.each(['invalidPositions', 'nonFinite', 'forbiddenGlobal'] as const)('rejects %s model entries', async (key) => {
    serveModel({ top_lap_store: { invalid: fixtures[key] } });
    await expect(loadModel()).rejects.toThrow();
});

it('defaults legacy entries to ACC and retains all feature data', async () => {
    serveModel({ top_lap_store: { old: fixtures.legacy } });
    await expect(loadModel()).resolves.toEqual([fixtures.expected]);
});
