import apiService from 'services/api.service';
import {
    analyzeRecordedTelemetry,
    RECORDED_ANALYSIS_MAX_BYTES,
    RECORDED_ANALYSIS_MAX_ROWS,
} from './analyze-recorded-telemetry';

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: jest.fn() },
}));
const mockPost = apiService.post as jest.Mock;
const rows = (count: number) => Array.from({ length: count }, (_, index) => ({ index }));
const responseFor = (records: Record<string, any>[]) => ({ data: {
    status: 'success', session_id: 'live-baseline', samples_analyzed: records.length,
    expert_time_available: true,
    segments: [{
        id: 'same-local-id', track_section: 'spa1', start_index: 0, end_index: records.length,
        labels: [{ label_name: 'EA', start_index: 0, end_index: records.length }],
        expert_reference_data: records.map((row, raw_index) => ({
            raw_index, expert_time_difference: row.index * 10,
        })),
        time_gap: {
            start_ms: records[0].index * 10,
            end_ms: records[records.length - 1].index * 10,
            delta_ms: (records[records.length - 1].index - records[0].index) * 10,
        },
    }],
} });
const analyze = (table: Record<string, any>[], overrides = {}) => analyzeRecordedTelemetry({
    sessionId: 'saved-session', track: 'Spa', car: 'GT3', table,
    signal: new AbortController().signal, onProgress: jest.fn(), ...overrides,
});

beforeEach(() => {
    mockPost.mockReset();
    mockPost.mockImplementation(async (_url, body) => responseFor(body.records));
});

it('analyzes the entire recording sequentially with bounded requests and global indices', async () => {
    const table = rows(RECORDED_ANALYSIS_MAX_ROWS * 2 + 37);
    const onProgress = jest.fn();
    let inFlight = 0;
    const responses: ReturnType<typeof responseFor>[] = [];
    mockPost.mockImplementation(async (_url, body) => {
        inFlight += 1;
        expect(inFlight).toBe(1);
        await Promise.resolve();
        inFlight -= 1;
        const response = responseFor(body.records);
        responses.push(response);
        return response;
    });
    const result = await analyze(table, { onProgress });
    expect(mockPost.mock.calls.length).toBeGreaterThan(2);
    for (const [url, body, config] of mockPost.mock.calls) {
        expect(url).toBe('/racing-session/analyze-live-recorded-analysis');
        expect(body.records.length).toBeLessThanOrEqual(RECORDED_ANALYSIS_MAX_ROWS);
        expect(body.records[0]).toBe(table[body.records[0].index]);
        expect(config).toEqual({ timeout: 120000, signal: expect.any(AbortSignal) });
    }
    expect(result.session_id).toBe('saved-session');
    expect(result.samples_analyzed).toBe(table.length);
    expect(result.parent_segment_count).toBe(result.segments.length);
    expect(new Set(result.segments.map((segment) => segment.id)).size).toBe(result.segments.length);
    expect(result.expert_time_available).toBe(true);
    expect(result.segments.flatMap((segment) => segment.expert_reference_data.map((row) => row.raw_index)))
        .toEqual(table.map((row) => row.index));
    let previousEnd = 0;
    for (const segment of result.segments) {
        expect(segment.start_index).toBe(previousEnd);
        expect(segment.labels).toEqual([{
            label_name: 'EA', start_index: segment.start_index, end_index: segment.end_index,
        }]);
        expect(segment.time_gap).toEqual({
            start_ms: segment.start_index * 10, end_ms: (segment.end_index - 1) * 10,
            delta_ms: (segment.end_index - segment.start_index - 1) * 10,
        });
        previousEnd = segment.end_index;
    }
    expect(previousEnd).toBe(table.length);
    expect(onProgress).toHaveBeenNthCalledWith(1, 0, table.length);
    expect(onProgress).toHaveBeenLastCalledWith(table.length, table.length);
    expect(responses[1].data.segments[0].start_index).toBe(0);
    expect(responses[1].data.segments[0].expert_reference_data[0].raw_index).toBe(0);
    expect(table).toEqual(rows(table.length));
});

it('bounds UTF-8 payload bytes as well as row counts, including request metadata', async () => {
    const payload = '\u8d5b\u8f66\ud83c\udfc1'.repeat(16384);
    const table = rows(60).map((row) => ({ ...row, payload }));
    const result = await analyze(table);
    expect(mockPost.mock.calls.length).toBeGreaterThan(1);
    for (const [, body] of mockPost.mock.calls) {
        expect(Buffer.byteLength(JSON.stringify(body), 'utf8')).toBeLessThanOrEqual(RECORDED_ANALYSIS_MAX_BYTES);
    }
    expect(result.segments.flatMap((segment) => segment.expert_reference_data.map((row) => row.raw_index)))
        .toEqual(table.map((row) => row.index));
    expect(result.samples_analyzed).toBe(table.length);
});

it('fails explicitly if a single sample exceeds the request budget', async () => {
    await expect(analyze([{ payload: 'x'.repeat(RECORDED_ANALYSIS_MAX_BYTES) }]))
        .rejects.toThrow('Telemetry sample 0 exceeds');
    expect(mockPost).not.toHaveBeenCalled();
});

it('stops at a failed chunk without returning or reporting a complete result', async () => {
    const table = rows(RECORDED_ANALYSIS_MAX_ROWS * 3);
    const onProgress = jest.fn();
    mockPost.mockImplementationOnce(async (_url, body) => responseFor(body.records))
        .mockRejectedValueOnce(new Error('Request failed'));
    await expect(analyze(table, { onProgress })).rejects.toThrow('Request failed');
    expect(mockPost).toHaveBeenCalledTimes(2);
    expect(onProgress).not.toHaveBeenCalledWith(table.length, table.length);
});

it('checks cancellation before sending and after receiving a chunk', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(analyze(rows(5), { signal: controller.signal })).rejects.toThrow('cancelled');
    expect(mockPost).not.toHaveBeenCalled();
    const active = new AbortController();
    mockPost.mockImplementationOnce(async (_url, body) => {
        active.abort();
        return responseFor(body.records);
    });
    await expect(analyze(rows(RECORDED_ANALYSIS_MAX_ROWS + 1), { signal: active.signal }))
        .rejects.toThrow('cancelled');
    expect(mockPost).toHaveBeenCalledTimes(1);
});

it.each([undefined, { status: 'error', segments: [] }, { status: 'success' }])(
    'rejects a failed or malformed response instead of silently omitting data: %j', async (data) => {
        mockPost.mockResolvedValueOnce({ data });
        await expect(analyze(rows(5))).rejects.toThrow('AI analysis failed');
    },
);

it('allows successful empty classifications across all chunks', async () => {
    mockPost.mockResolvedValue({ data: { status: 'success', segments: [] } });
    const table = rows(RECORDED_ANALYSIS_MAX_ROWS + 1);
    await expect(analyze(table)).resolves.toMatchObject({
        samples_analyzed: table.length, parent_segment_count: 0, segments: [],
    });
    expect(mockPost).toHaveBeenCalledTimes(2);
});

it('rejects empty input without issuing a request', async () => {
    await expect(analyze([])).rejects.toThrow('No telemetry samples');
    expect(mockPost).not.toHaveBeenCalled();
});
