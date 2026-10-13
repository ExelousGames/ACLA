import { ReactNode } from 'react';
import { act, render, waitFor } from '@testing-library/react';
import apiService from 'services/api.service';
import type { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { parseTelemetryFrames } from 'views/session-shared/visualization/charts/mapTelemetry';
import { RecordedSessionData, RecordedSessionDataProvider, useRecordedSessionData } from './RecordedSessionDataProvider';

jest.mock('services/api.service', () => ({ __esModule: true, default: { post: jest.fn() } }));
const post = apiService.post as jest.Mock;
const session = (id: string, data: any[] = [], local = false): RacingSessionDetailedInfoDto => ({
    SessionId: id, session_name: id, map: 'Spa', car: 'GT3', user_id: '', points: [], data,
    storage: local ? 'local' : 'cloud',
});
const init = (sessionId: string, chunkCount = 1) => ({
    data: { downloadId: 'download-1', sessionMetadata: [{ sessionId, chunkCount, map: 'Spa', car_name: 'GT3' }] },
});
function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((finish) => { resolve = finish; });
    return { promise, resolve };
}

let snapshots: RecordedSessionData[];
let current: RecordedSessionData;
const Reader = () => {
    current = useRecordedSessionData();
    snapshots.push(current);
    return null;
};
const Harness = ({ selected, children = <Reader /> }: {
    selected: RacingSessionDetailedInfoDto | null;
    children?: ReactNode;
}) => <RecordedSessionDataProvider session={selected} map="Spa">{children}</RecordedSessionDataProvider>;

beforeEach(() => {
    post.mockReset();
    snapshots = [];
});

it('shares one complete cloud table across readers and preserves every source row and value', async () => {
    const rows = [
        { time: 9, speed: '001.20', missing: null, extra: { values: [3, null, 1] } },
        { time: 1, speed: -2 },
        { time: 1, speed: -2 },
        {},
    ];
    const secondChunk = deferred<any>();
    post.mockResolvedValueOnce(init('cloud', 2))
        .mockResolvedValueOnce({ data: rows.slice(0, 2) })
        .mockReturnValueOnce(secondChunk.promise);
    const selected = session('cloud');
    const view = render(<Harness selected={selected}><Reader /><Reader /></Harness>);
    await waitFor(() => expect(post).toHaveBeenCalledTimes(3));
    expect(current.status).toBe('loading');
    expect(current.table).toEqual([]);
    await act(async () => secondChunk.resolve({ data: { data: rows.slice(2) } }));
    expect(current).toMatchObject({ sessionId: 'cloud', source: 'cloud', status: 'ready', table: rows });
    rows.forEach((row, index) => expect(current.table[index]).toBe(row));
    const table = current.table;
    expect(snapshots.filter((entry) => entry.status === 'ready').every((entry) => entry.table === table)).toBe(true);
    expect(selected.data).toEqual([]);
    // Reopening visualizations uses the provider's table without downloading again.
    view.rerender(<Harness selected={selected}>{null}</Harness>);
    view.rerender(<Harness selected={selected}><Reader /><Reader /></Harness>);
    expect(current.table).toBe(table);
    expect(post).toHaveBeenCalledTimes(3);
    expect(post).toHaveBeenLastCalledWith('/racing-session/download/chunk', {
        downloadId: 'download-1', sessionId: 'cloud', trackName: 'Spa', carName: 'GT3', chunkIndex: 1,
    }, expect.objectContaining({ signal: expect.anything() }));
});

it('publishes local importer rows unchanged without any cloud request', () => {
    const rows = [{ unusual: '0', value: NaN }, {}, { value: Infinity, nested: [null, -5] }];
    render(<Harness selected={session('local', rows, true)} />);
    expect(current).toMatchObject({ sessionId: 'local', source: 'local-ibt', status: 'ready' });
    expect(current.table).toBe(rows);
    expect(post).not.toHaveBeenCalled();
});

it('keeps original indices and values while a visualization derives drawable frames', () => {
    const rows = Object.freeze([
        Object.freeze({ Graphics_current_time: 50, extra: 'no coordinates' }),
        ...[1000, 500].map((time) => Object.freeze({
            Graphics_current_time: time,
            Graphics_player_car_id: 0,
            Graphics_car_id: Object.freeze([0]),
            Graphics_car_coordinates: Object.freeze([Object.freeze({ x: 10, y: 1, z: 5 })]),
        })),
    ]);
    render(<Harness selected={session('local', rows as any, true)} />);
    const frames = parseTelemetryFrames(current.table);
    expect(frames.map((frame) => frame.sourceIndex)).toEqual([1, 2]);
    expect(frames[1].time).toBeGreaterThan(frames[0].time);
    expect(current.table).toBe(rows);
    expect(current.table[2].Graphics_current_time).toBe(500);
    expect(current.table).toHaveLength(3);
});

it('aborts an old cloud download and ignores its late result after selecting local data', async () => {
    const pending = deferred<any>();
    post.mockResolvedValueOnce(init('old')).mockReturnValueOnce(pending.promise);
    const view = render(<Harness selected={session('old')} />);
    await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
    const signal: AbortSignal = post.mock.calls[1][2].signal;
    const rows = [{ local: true }];
    view.rerender(<Harness selected={session('local', rows, true)} />);
    expect(signal.aborted).toBe(true);
    await act(async () => pending.resolve({ data: [{ stale: true }] }));
    expect(current.sessionId).toBe('local');
    expect(current.table).toBe(rows);
});

it('hides the previous table immediately on selection changes and clearing', async () => {
    post.mockResolvedValueOnce(init('first')).mockResolvedValueOnce({ data: [{ first: true }] });
    const view = render(<Harness selected={session('first')} />);
    await waitFor(() => expect(current.status).toBe('ready'));
    post.mockReturnValueOnce(new Promise(() => {}));
    snapshots = [];
    view.rerender(<Harness selected={session('next')} />);
    expect(snapshots.every((entry) => entry.sessionId === 'next' && entry.table.length === 0)).toBe(true);
    view.rerender(<Harness selected={null} />);
    expect(current).toMatchObject({ sessionId: null, source: null, status: 'idle', table: [] });
    expect(post.mock.calls[2][2].signal.aborted).toBe(true);
});

it('does not start chunk requests after an unmounted initializer completes', async () => {
    const pending = deferred<any>();
    post.mockReturnValueOnce(pending.promise);
    const view = render(<Harness selected={session('cloud')} />);
    view.unmount();
    await act(async () => pending.resolve(init('cloud')));
    expect(post.mock.calls[0][2].signal.aborted).toBe(true);
    expect(post).toHaveBeenCalledTimes(1);
});

it('reports a failed chunk without publishing a partial table, then loads another session', async () => {
    post.mockResolvedValueOnce(init('broken', 2))
        .mockResolvedValueOnce({ data: [{ partial: true }] })
        .mockRejectedValueOnce(new Error('Download failed'));
    const view = render(<Harness selected={session('broken')} />);
    await waitFor(() => expect(current.status).toBe('error'));
    expect(current.message).toBe('Download failed');
    expect(snapshots.every((entry) => entry.table.length === 0)).toBe(true);
    post.mockResolvedValueOnce(init('next')).mockResolvedValueOnce({ data: [{ complete: true }] });
    view.rerender(<Harness selected={session('next')} />);
    await waitFor(() => expect(current.status).toBe('ready'));
    expect(current.table).toEqual([{ complete: true }]);
});

it('reports malformed transport envelopes instead of silently dropping them', async () => {
    post.mockResolvedValueOnce(init('cloud')).mockResolvedValueOnce({ data: { unexpected: [] } });
    render(<Harness selected={session('cloud')} />);
    await waitFor(() => expect(current.status).toBe('error'));
    expect(current.message).toContain('did not return a data table');
    expect(current.table).toEqual([]);
});

it('accepts an empty source table as ready', async () => {
    post.mockResolvedValueOnce(init('empty')).mockResolvedValueOnce({ data: [] });
    render(<Harness selected={session('empty')} />);
    await waitFor(() => expect(current.status).toBe('ready'));
    expect(current.table).toEqual([]);
});
