import { fetchCircuitMapById, fetchCircuitMapList } from 'services/circuitMapService';
import { loadTopLapReferenceModel } from 'services/topLapReferenceService';
import { parseRecordedDrivingSession } from './recorded-driving-session';
import { loadRecordedDrivingReferences, recordedTrackKey } from './recorded-driving-reference';
import { circuitMapFixture, referenceLapFixture } from './training-fixtures';

jest.mock('services/circuitMapService', () => ({ fetchCircuitMapById: jest.fn(), fetchCircuitMapList: jest.fn() }));
jest.mock('services/topLapReferenceService', () => ({ loadTopLapReferenceModel: jest.fn() }));
const list = fetchCircuitMapList as jest.MockedFunction<typeof fetchCircuitMapList>;
const fetchMap = fetchCircuitMapById as jest.MockedFunction<typeof fetchCircuitMapById>;
const loadLaps = loadTopLapReferenceModel as jest.MockedFunction<typeof loadTopLapReferenceModel>;
beforeEach(() => { jest.resetAllMocks(); });

it('loads only distinct matching recorded tracks and scopes map requests to the simulator', async () => {
    const session = parseRecordedDrivingSession([
        { Graphics_game_time: 0, Static_track: 'monza' },
        { Graphics_game_time: 1, Static_track: 'monza' },
        { Graphics_game_time: 2, Static_track: 'spa' },
    ].map((sample) => JSON.stringify(sample)).join('\n'), 'acc');
    const lap = referenceLapFixture(), map = circuitMapFixture(lap);
    loadLaps.mockResolvedValue([lap]);
    list.mockResolvedValue([
        { id: 'wrong-game', game: 'iracing', circuit_name: 'Monza', source_track_key: 'monza' },
        { id: 'monza-map', game: 'acc', circuit_name: 'Monza', source_track_key: 'monza' },
        { id: 'other-map', game: 'acc', circuit_name: 'Brands Hatch', source_track_key: 'brands_hatch' },
    ]);
    fetchMap.mockResolvedValue(map);
    const result = await loadRecordedDrivingReferences(session);
    expect(list).toHaveBeenCalledWith('acc');
    expect(fetchMap).toHaveBeenCalledTimes(1);
    expect(fetchMap).toHaveBeenCalledWith('monza-map', 'acc');
    expect(result.laps).toEqual([lap]);
    expect(result.maps.get('monza')).toBe(map);
    expect(result.maps.has('spa')).toBe(false);
});

it('preserves iRacing layout identities while normalizing ACC aliases', () => {
    expect(recordedTrackKey('iracing', ' spa - grand prix ')).toBe('spa - grand prix');
    expect(recordedTrackKey('iracing', 'spa - endurance')).not.toBe(recordedTrackKey('iracing', 'spa - grand prix'));
    expect(recordedTrackKey('acc', 'Autodromo Nazionale Monza')).toBe('monza');
});

it('propagates a reference download failure and permits a fresh retry', async () => {
    const session = parseRecordedDrivingSession('{"Graphics_game_time":0,"Static_track":"monza"}', 'acc');
    list.mockResolvedValue([]);
    loadLaps.mockRejectedValueOnce(new Error('Reference download failed.')).mockResolvedValueOnce([]);
    await expect(loadRecordedDrivingReferences(session)).rejects.toThrow('Reference download failed.');
    await expect(loadRecordedDrivingReferences(session)).resolves.toEqual({ laps: [], maps: new Map() });
    expect(loadLaps).toHaveBeenCalledTimes(2);
});

it('rejects unsupported recorded simulators before requesting maps', async () => {
    const session = parseRecordedDrivingSession('{"Graphics_game_time":0}', 'ac');
    await expect(loadRecordedDrivingReferences(session)).rejects.toThrow(/supports ACC and iRacing/);
    expect(list).not.toHaveBeenCalled();
    expect(loadLaps).not.toHaveBeenCalled();
});
