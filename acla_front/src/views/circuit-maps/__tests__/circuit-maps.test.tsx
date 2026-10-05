import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CircuitMaps from '../circuit-maps';
import apiService from 'services/api.service';
import type { LiveSessionRuntime, RecordedFileReadEvent } from 'views/live-session/live-session-types';
import {
    OPERATION_COMPONENT_NAMES,
    OperationComponentRefProvider,
    useRegisterOperationComponentRef,
} from 'contexts/OperationComponentRefContext';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { RecordingState } from 'views/live-session/recording-state';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';

const mockRefreshCircuitMaps = jest.fn();
const mockUpsertCachedCircuitMap = jest.fn();
const mockRemoveCachedCircuitMap = jest.fn();

jest.mock('radix-ui/internal', () => jest.requireActual('radix-ui/dist/internal.js'), { virtual: true });

jest.mock('@radix-ui/themes', () => ({
    AlertDialog: jest.requireActual('@radix-ui/themes').AlertDialog,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Box: require('react').forwardRef(({ children, ...props }: any, ref: any) => <div ref={ref} {...props}>{children}</div>),
    Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Flex: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    Heading: ({ children, ...props }: any) => <h2 {...props}>{children}</h2>,
    Spinner: () => <span>Loading</span>,
    Text: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    TextField: {
        Root: ({ value, onChange, placeholder, ...props }: any) => (
            <input aria-label={placeholder} placeholder={placeholder} value={value} onChange={onChange} {...props} />
        ),
    },
    Select: {
        Root: ({ value, onValueChange, children, disabled }: any) => (
            <select value={value} disabled={disabled} onChange={(event) => onValueChange(event.target.value)}>
                {children}
            </select>
        ),
        Trigger: () => null,
        Content: ({ children }: any) => <>{children}</>,
        Item: ({ value, children }: any) => <option value={value}>{children}</option>,
    },
}));

jest.mock('@radix-ui/react-icons', () => ({
    CheckIcon: () => <span>Check</span>,
    Cross2Icon: () => <span>Close</span>,
    PauseIcon: () => <span>Pause</span>,
    PlayIcon: () => <span>Play</span>,
    PlusIcon: () => <span>Plus</span>,
    ReloadIcon: () => <span>Reload</span>,
    TrashIcon: () => <span>Trash</span>,
}));

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: {
        get: jest.fn(),
        post: jest.fn(),
        put: jest.fn(),
        delete: jest.fn(),
    },
}));

jest.mock('contexts/CircuitMapsContext', () => ({
    useCircuitMaps: () => ({
        refreshCircuitMaps: mockRefreshCircuitMaps,
        upsertCachedCircuitMap: mockUpsertCachedCircuitMap,
        removeCachedCircuitMap: mockRemoveCachedCircuitMap,
    }),
}));

const mockedApi = apiService as jest.Mocked<typeof apiService>;
const imported = { filePath: 'C:\\temp\\circuit.jsonl', fileName: 'spa.ibt', rowCount: 2, track: 'spa - grandprix', car: 'GT3' };
let readListener: (event: RecordedFileReadEvent) => void;
const importedRow = {
    Graphics_status: 2, Graphics_normalized_car_position: 0.25,
    Graphics_player_car_id: 63, Graphics_car_id: [63, -1],
    Graphics_car_coordinates: [{ x: 10, y: 2, z: 30 }, { x: 0, y: 0, z: 0 }],
};
const finishImportRead = (rows = [importedRow, importedRow]) => {
    readListener({ type: 'chunk', readId: 'map-read', rows });
    readListener({ type: 'complete', readId: 'map-read', game: 'iracing', format: 'standard-flat', rowCount: rows.length, totalBytes: 200 });
};

const openIRacingImport = async () => {
    await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'iracing');
    await userEvent.click(screen.getByRole('button', { name: /open .ibt file/i }));
    await waitFor(() => expect(window.electronAPI.startRecordedFileRead).toHaveBeenCalledWith({
        filePath: imported.filePath, game: 'iracing', purpose: 'consume',
    }));
};

const baseContext: LiveSessionRuntime = {
    sessionGame: null,
    staticData: {},
    recordingState: RecordingState.CHECKING,
    recordingMetadata: null,
    recordingFileKey: null,
    recordingActive: false,
    recordingGame: null,
    restorationStatus: 'idle',
    restorationError: null,
    recordingFileValidation: null,
    recorderControl: null,
    analysisResultPages: [],
    activeAnalysisResultPageId: null,
    getNextCorner: jest.fn(() => null),
    getLiveSessionSnapshot: jest.fn(() => ({
        status: 'empty',
        track: '',
        car: '',
        current_lap: 0,
        completed_laps: 0,
        normalized_position: 0,
        sample_count: 0,
        live_session_type: 'unknown',
        completed_lap_count: 0,
    })),
    startLiveSession: jest.fn(),
    endLiveSession: jest.fn(),
    setRecordingMetadata: jest.fn(),
    transitionRecordingState: jest.fn(),
    startRecordingSession: jest.fn(),
    stopRecordingSession: jest.fn(),
    streamRecordedTelemetry: jest.fn(),
    clearRecordingSession: jest.fn(),
    clearPersistedDraft: jest.fn(),
    registerRecorderControl: jest.fn(),
    appendAnalysisResultPage: jest.fn(),
    selectAnalysisResultPage: jest.fn(),
    updateActiveAnalysisResultPage: jest.fn(),
};

const LiveSessionReference = ({ snapshot }: { snapshot: LiveSessionRuntime }) => {
    const snapshotRef = React.useRef(snapshot);
    snapshotRef.current = snapshot;
    const componentRef = React.useRef<any>(null);
    if (componentRef.current === null) {
        componentRef.current = {
            getComponentName: () => OPERATION_COMPONENT_NAMES.LIVE_SESSION,
            getAssistantSnapshot: () => snapshotRef.current,
            subscribeAssistantSnapshot: () => () => undefined,
        };
    }
    useRegisterOperationComponentRef(componentRef);
    return null;
};

const renderCircuitMaps = (context: Partial<LiveSessionRuntime> = {}) => (
    render(
        <OperationComponentRefProvider>
            <LiveSessionReference snapshot={{ ...baseContext, ...context }} />
            <CircuitMaps />
        </OperationComponentRefProvider>
    )
);

describe('CircuitMaps', () => {
    beforeEach(() => {
        liveTelemetryStore.resetSession();
        jest.clearAllMocks();
        mockedApi.get.mockResolvedValue({ data: { list: [] }, status: 200 } as any);
        mockedApi.post.mockResolvedValue({ data: { id: 'map-1' }, status: 201 } as any);
        mockedApi.put.mockResolvedValue({ data: {}, status: 200 } as any);
        mockedApi.delete.mockReset().mockResolvedValue({ data: undefined, status: 204 } as any);
        mockRefreshCircuitMaps.mockResolvedValue([]);
        window.electronAPI = {
            importLocalIRacingTelemetry: jest.fn().mockResolvedValue(imported),
            onRecordedFileReadEvent: jest.fn((callback) => { readListener = callback; return jest.fn(); }),
            startRecordedFileRead: jest.fn().mockResolvedValue({ readId: 'map-read' }),
            cancelRecordedFileRead: jest.fn().mockResolvedValue(undefined),
            deleteTempFile: jest.fn().mockResolvedValue({ success: true }),
        } as any;

        (global as any).ResizeObserver = class {
            observe = jest.fn();
            disconnect = jest.fn();
        };

        HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
            setTransform: jest.fn(),
            clearRect: jest.fn(),
            fillRect: jest.fn(),
            save: jest.fn(),
            restore: jest.fn(),
            beginPath: jest.fn(),
            moveTo: jest.fn(),
            lineTo: jest.fn(),
            stroke: jest.fn(),
            arc: jest.fn(),
            fill: jest.fn(),
            fillText: jest.fn(),
        })) as any;
    });

    it.each(['left_boundary', 'middle_line', 'right_boundary', 'pit_lane'])('imports converted iRacing driver coordinates into %s and saves the game and track', async (mode) => {
        renderCircuitMaps();
        await userEvent.selectOptions(screen.getAllByRole('combobox')[1], mode);
        await openIRacingImport();
        expect(screen.getByRole('button', { name: /save/i })).toBeDisabled();
        expect(screen.getByRole('button', { name: /add point/i })).toBeDisabled();
        await act(async () => finishImportRead());
        expect(screen.getByLabelText('Circuit name')).toHaveValue(imported.track);
        expect(screen.getByRole('status')).toHaveTextContent('imported 2 driver samples');
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(imported.filePath);
        await userEvent.click(screen.getByRole('button', { name: /save/i }));
        await waitFor(() => expect(mockedApi.post).toHaveBeenCalledWith('/circuit-map', expect.objectContaining({
            game: 'iracing', circuit_name: imported.track, source_track_key: imported.track,
            samples: expect.objectContaining({ [mode]: [expect.objectContaining({ bin: 250, x: 10, y: 2, z: 30, sample_count: 2 })] }),
        })));
        expect(mockUpsertCachedCircuitMap).toHaveBeenCalledWith(expect.objectContaining({ game: 'iracing' }));
        expect(mockRefreshCircuitMaps).toHaveBeenCalledWith('iracing');
        expect(mockedApi.get).toHaveBeenCalledWith('/circuit-map/list', { game: 'iracing' });
    });

    it('leaves a draft unchanged when file selection is cancelled', async () => {
        (window.electronAPI.importLocalIRacingTelemetry as jest.Mock).mockResolvedValue(null);
        renderCircuitMaps();
        await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'iracing');
        await userEvent.type(screen.getByLabelText('Circuit name'), 'My draft');
        await userEvent.click(screen.getByRole('button', { name: /add point/i }));
        await userEvent.click(screen.getByRole('button', { name: /open .ibt file/i }));
        await waitFor(() => expect(screen.getByRole('button', { name: /open .ibt file/i })).toBeEnabled());
        expect(screen.getByLabelText('Circuit name')).toHaveValue('My draft');
        expect(screen.getByText(/1 samples/)).toBeInTheDocument();
        expect(window.electronAPI.startRecordedFileRead).not.toHaveBeenCalled();
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });

    it('reports unusable coordinates and retains the existing draft', async () => {
        renderCircuitMaps();
        await openIRacingImport();
        await act(async () => finishImportRead([{ ...importedRow, Graphics_player_car_id: -1 }]));
        expect(screen.getByRole('alert')).toHaveTextContent('no usable driver coordinates');
        expect(screen.getByText(/0 samples/)).toBeInTheDocument();
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(imported.filePath);
        expect(screen.getByRole('button', { name: /open .ibt file/i })).toBeEnabled();
    });

    it('reports converter failures and allows retrying', async () => {
        (window.electronAPI.importLocalIRacingTelemetry as jest.Mock).mockRejectedValue(new Error('Invalid iRacing .ibt header.'));
        renderCircuitMaps();
        await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'iracing');
        await userEvent.click(screen.getByRole('button', { name: /open .ibt file/i }));
        expect(await screen.findByRole('alert')).toHaveTextContent('Invalid iRacing .ibt header.');
        expect(screen.getByRole('button', { name: /open .ibt file/i })).toBeEnabled();
    });

    it('discards partial samples if the telemetry reader fails', async () => {
        renderCircuitMaps();
        await openIRacingImport();
        await act(async () => {
            readListener({ type: 'chunk', readId: 'map-read', rows: [importedRow] });
            readListener({ type: 'error', readId: 'map-read', message: 'Read failed' });
        });
        expect(screen.getByRole('alert')).toHaveTextContent('Read failed');
        expect(screen.getByText(/0 samples/)).toBeInTheDocument();
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(imported.filePath);
    });

    it('rejects a recording from a different track without changing the map', async () => {
        renderCircuitMaps();
        await openIRacingImport();
        await act(async () => finishImportRead());
        (window.electronAPI.importLocalIRacingTelemetry as jest.Mock).mockResolvedValue({ ...imported, track: 'monza' });
        await userEvent.click(screen.getByRole('button', { name: /open .ibt file/i }));
        expect(await screen.findByRole('alert')).toHaveTextContent('different circuit');
        expect(screen.getByLabelText('Circuit name')).toHaveValue(imported.track);
        expect(window.electronAPI.startRecordedFileRead).toHaveBeenCalledTimes(1);
    });

    it.each(['game', 'new map', 'unmount'])('cancels an active import on %s', async (action) => {
        const view = renderCircuitMaps();
        await openIRacingImport();
        if (action === 'game') await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'acc');
        else if (action === 'new map') await userEvent.click(screen.getByRole('button', { name: /new map/i }));
        else view.unmount();
        await act(async () => finishImportRead());
        expect(window.electronAPI.cancelRecordedFileRead).toHaveBeenCalledWith('map-read');
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(imported.filePath);
        if (action !== 'unmount') expect(screen.getByText(/0 samples/)).toBeInTheDocument();
    });

    it('cleans up conversion that finishes after resetting the map', async () => {
        let finishConversion!: (value: typeof imported) => void;
        (window.electronAPI.importLocalIRacingTelemetry as jest.Mock).mockImplementation(() => new Promise((resolve) => { finishConversion = resolve; }));
        renderCircuitMaps();
        await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'iracing');
        await userEvent.click(screen.getByRole('button', { name: /open .ibt file/i }));
        await userEvent.click(screen.getByRole('button', { name: /new map/i }));
        await act(async () => finishConversion(imported));
        expect(window.electronAPI.startRecordedFileRead).not.toHaveBeenCalled();
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(imported.filePath);
        expect(screen.getByLabelText('Circuit name')).toHaveValue('');
    });

    it('explains local import availability in a browser', async () => {
        window.electronAPI = undefined as any;
        renderCircuitMaps();
        await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'iracing');
        expect(screen.getByText('Open the desktop app to import local iRacing .ibt files.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /open .ibt file/i })).toBeDisabled();
    });

    it('loads global maps without a user id and disables ACC capture when telemetry is offline', async () => {
        renderCircuitMaps();

        await waitFor(() => expect(mockedApi.get).toHaveBeenCalledWith('/circuit-map/list', { game: 'acc' }));
        expect(JSON.stringify(mockedApi.get.mock.calls)).not.toContain('user_id');
        expect(screen.getByRole('button', { name: /start capture/i })).toBeDisabled();
    });

    it('uses Static_track from the current telemetry row for the circuit name', async () => {
        await act(async () => {
            renderCircuitMaps();
            liveTelemetryStore.publishFrame({
                type: 'frame', game: 'acc', sequence: 1, committedSequence: 1, committedCount: 1,
                sample: { Static_track: 'Canonical Circuit' },
            });
        });

        expect(screen.getByLabelText('Circuit name')).toHaveValue('Canonical Circuit');
    });

    it.each([
        [{ Static_track: 'Canonical Circuit' }, 'Canonical Circuit'],
        [{ track: 'Legacy Circuit' }, ''],
        [{ Static: { track: 'Legacy Circuit' } }, ''],
        [{ Statics: { track: 'Legacy Circuit' } }, ''],
    ])('uses only canonical static fields from the session: %o', async (staticData, expectedName) => {
        renderCircuitMaps({ staticData });

        await waitFor(() => expect(mockedApi.get).toHaveBeenCalled());
        expect(screen.getByLabelText('Circuit name')).toHaveValue(expectedName);
    });

    it('saves a new global map payload without user ownership', async () => {
        renderCircuitMaps();
        act(() => {
            liveTelemetryStore.publishFrame({
                type: 'frame',
                game: 'acc',
                sequence: 1,
                committedSequence: 1,
                committedCount: 1,
                sample: {
                Graphics_status: ACC_STATUS.ACC_LIVE,
                Graphics_normalized_car_position: 0.1,
                Graphics_car_coordinates: Array.from({ length: 60 }, (_, slot) => slot === 0
                    ? { x: 1, y: 0, z: 2 } : { x: 0, y: 0, z: 0 }),
                },
            });
        });

        await userEvent.type(screen.getByLabelText('Circuit name'), 'Global Test Circuit');
        await userEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mockedApi.post).toHaveBeenCalled());
        const [, payload] = mockedApi.post.mock.calls[0];
        expect(mockedApi.post.mock.calls[0][0]).toBe('/circuit-map');
        expect(payload).toMatchObject({
            game: 'acc',
            circuit_name: 'Global Test Circuit',
            resolution: 1000,
        });
        expect(JSON.stringify(payload)).not.toContain('user_id');
        expect(mockUpsertCachedCircuitMap).toHaveBeenCalledWith(expect.objectContaining({
            id: 'map-1',
            game: 'acc',
            circuit_name: 'Global Test Circuit',
            resolution: 1000,
        }));
        expect(mockRefreshCircuitMaps).toHaveBeenCalledWith('acc');
    });

    it.each(['pit_lane', 'middle_line'])('adds and saves manual points in %s mode', async (mode) => {
        renderCircuitMaps();

        await userEvent.type(screen.getByLabelText('Circuit name'), 'Capture Test Circuit');
        await userEvent.selectOptions(screen.getAllByRole('combobox')[1], mode);
        await userEvent.clear(screen.getByLabelText('Normalized position 0-1'));
        await userEvent.type(screen.getByLabelText('Normalized position 0-1'), '0.42');
        await userEvent.clear(screen.getByLabelText('X'));
        await userEvent.type(screen.getByLabelText('X'), '12');
        await userEvent.clear(screen.getByLabelText('Z'));
        await userEvent.type(screen.getByLabelText('Z'), '34');
        await userEvent.click(screen.getByRole('button', { name: /add point/i }));
        await userEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mockedApi.post).toHaveBeenCalled());
        const [, payload] = mockedApi.post.mock.calls[0];
        expect(payload).toMatchObject({
            samples: {
                [mode]: [{
                    bin: 420,
                    normalized_position: 0.42,
                    x: 12,
                    z: 34,
                    locked: true,
                }],
            },
        });
        await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
    });

    it.each(['left_boundary', 'middle_line'])('processes all 120 live capture frames in %s mode in one React batch', async (mode) => {
        renderCircuitMaps();
        await waitFor(() => expect(mockedApi.get).toHaveBeenCalledWith('/circuit-map/list', { game: 'acc' }));
        await userEvent.type(screen.getByLabelText('Circuit name'), 'Live Test Circuit');
        await userEvent.selectOptions(screen.getAllByRole('combobox')[1], mode);
        act(() => {
            liveTelemetryStore.publishFrame({
                type: 'frame',
                game: 'acc',
                sample: {
                    Graphics_status: ACC_STATUS.ACC_LIVE,
                    Graphics_normalized_car_position: 0,
                    Graphics_car_coordinates: Array.from({ length: 60 }, (_, slot) => slot === 0
                        ? { x: 1, y: 1, z: 2 } : { x: 0, y: 0, z: 0 }),
                },
                sequence: 1,
                committedSequence: 1,
                committedCount: 1,
            });
        });
        await userEvent.click(screen.getByRole('button', { name: /start capture/i }));
        act(() => liveTelemetryStore.beginStream());

        act(() => {
            for (let sequence = 1; sequence <= 120; sequence += 1) {
                liveTelemetryStore.publishFrame({
                    type: 'frame',
                    game: 'acc',
                    sample: {
                        Graphics_status: ACC_STATUS.ACC_LIVE,
                        Graphics_normalized_car_position: (sequence - 1) / 1000,
                        Graphics_car_coordinates: Array.from({ length: 60 }, (_, slot) => slot === 0
                            ? { x: sequence + 1, y: 1, z: sequence + 2 }
                            : { x: 0, y: 0, z: 0 }),
                    },
                    sequence,
                    committedSequence: sequence,
                    committedCount: sequence,
                });
            }
        });

        expect(screen.getByText('120 samples')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /pause capture/i }));
        await userEvent.click(screen.getByRole('button', { name: /save/i }));
        await waitFor(() => expect(mockedApi.post).toHaveBeenCalled());
        const [, payload] = mockedApi.post.mock.calls[0];
        expect((payload as any).samples[mode]).toHaveLength(120);
        expect((payload as any).samples[mode === 'middle_line' ? 'left_boundary' : 'middle_line']).toEqual([]);
        await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
    });

    it('loads, edits, and saves a middle line without losing other captured paths', async () => {
        const sample = {
            bin: 420, normalized_position: 0.42, x: 12, y: 0, z: 34,
            sample_count: 3, updated_at: '2026-01-01T00:00:00.000Z',
        };
        const savedMap = {
            id: 'middle-map', game: 'acc', circuit_name: 'Middle Test Circuit', resolution: 1000,
            samples: { middle_line: [sample], left_boundary: [{ ...sample, x: 2 }] },
        };
        mockedApi.get.mockImplementation(async (url: string) => ({
            data: url === '/circuit-map/list' ? { list: [savedMap] } : savedMap,
            status: 200,
        } as any));
        renderCircuitMaps();
        await userEvent.click(await screen.findByRole('button', { name: 'Middle Test Circuit ACC' }));
        expect(await screen.findByText('2 samples')).toBeInTheDocument();
        await userEvent.selectOptions(screen.getAllByRole('combobox')[1], 'middle_line');
        await userEvent.clear(screen.getByLabelText('Normalized position 0-1'));
        await userEvent.type(screen.getByLabelText('Normalized position 0-1'), '0.5');
        await userEvent.click(screen.getByRole('button', { name: /add point/i }));
        expect(screen.getByText('3 samples')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /^close unlock$/i }));
        await userEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/middle-map', expect.objectContaining({
            samples: {
                middle_line: [sample, expect.objectContaining({ bin: 500, locked: false })],
                left_boundary: savedMap.samples.left_boundary,
                right_boundary: [],
                pit_lane: [],
            },
        })));
        expect(mockUpsertCachedCircuitMap).toHaveBeenLastCalledWith(expect.objectContaining({
            sample_count: 3,
            samples: expect.objectContaining({ middle_line: [sample, expect.objectContaining({ bin: 500 })] }),
        }));
        await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
        await userEvent.click(screen.getByRole('button', { name: /^trash delete$/i }));
        expect(screen.getByText('2 samples')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /new map/i }));
        expect(screen.getByText('0 samples')).toBeInTheDocument();
    });

    it('switches Other games into manual edit mode', async () => {
        renderCircuitMaps();
        await screen.findByText('No global maps found.');

        await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'other');

        expect(await screen.findByText('Manual Edit')).toBeInTheDocument();
        await waitFor(() => expect(screen.queryByText('Loading maps')).not.toBeInTheDocument());
        expect(screen.queryByText('ACC Offline')).not.toBeInTheDocument();
    });

    describe('removing a saved map', () => {
        const savedMap = {
            id: 'map/1', game: 'acc', circuit_name: 'Test Circuit', resolution: 1000,
            samples: {
                left_boundary: [{ bin: 1, normalized_position: 0.001, x: 1, y: 0, z: 2, sample_count: 1 }],
            },
        };

        const selectSavedMap = async () => {
            mockedApi.get.mockImplementation(async (url: string) => ({
                data: url === '/circuit-map/list' ? { list: [savedMap] } : savedMap,
                status: 200,
            } as any));
            renderCircuitMaps();
            await userEvent.click(await screen.findByRole('button', { name: 'Test Circuit ACC' }));
            await waitFor(() => expect(screen.getByRole('button', { name: /remove map/i })).toBeEnabled());
        };

        const openDeleteDialog = async () => {
            await userEvent.click(screen.getByRole('button', { name: /remove map/i }));
            return screen.getByRole('alertdialog');
        };

        it('disables removal for an unsaved map', async () => {
            renderCircuitMaps();
            await screen.findByText('No global maps found.');
            expect(screen.getByRole('button', { name: /remove map/i })).toBeDisabled();
        });

        it('requires confirmation and preserves the map and edits on cancel', async () => {
            await selectSavedMap();
            await userEvent.type(screen.getByLabelText('Circuit name'), ' edited');
            const dialog = await openDeleteDialog();
            expect(within(dialog).getByText(/Permanently remove “Test Circuit”/)).toHaveTextContent('removed for everyone');
            expect(mockedApi.delete).not.toHaveBeenCalled();
            await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
            expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
            expect(screen.getByLabelText('Circuit name')).toHaveValue('Test Circuit edited');
            expect(screen.getByText('1 samples')).toBeInTheDocument();
            expect(mockRemoveCachedCircuitMap).not.toHaveBeenCalled();
        });

        it('removes the selected map after success, clears the editor, and prevents duplicate requests', async () => {
            let resolveDelete!: (value: any) => void;
            mockedApi.delete.mockReturnValue(new Promise((resolve) => { resolveDelete = resolve; }));
            await selectSavedMap();
            const dialog = await openDeleteDialog();
            fireEvent.click(within(dialog).getByRole('button', { name: 'Remove map' }));
            expect(mockedApi.delete).toHaveBeenCalledWith('/circuit-map/map%2F1');
            expect(within(dialog).getByRole('button', { name: 'Removing...' })).toBeDisabled();
            expect(within(dialog).getByRole('button', { name: 'Cancel' })).toBeDisabled();
            fireEvent.keyDown(dialog, { key: 'Escape' });
            expect(screen.getByRole('alertdialog')).toBeInTheDocument();
            fireEvent.click(within(dialog).getByRole('button', { name: 'Removing...' }));
            expect(mockedApi.delete).toHaveBeenCalledTimes(1);
            expect(mockRemoveCachedCircuitMap).not.toHaveBeenCalled();

            await act(async () => { resolveDelete({ status: 204 }); });

            expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
            expect(screen.getByText('No global maps found.')).toBeInTheDocument();
            expect(screen.getByLabelText('Circuit name')).toHaveValue('');
            expect(screen.getByText('0 samples')).toBeInTheDocument();
            expect(screen.getByRole('button', { name: /remove map/i })).toBeDisabled();
            expect(mockRemoveCachedCircuitMap).toHaveBeenCalledWith('map/1');

            await userEvent.type(screen.getByLabelText('Circuit name'), 'New Circuit');
            await userEvent.click(screen.getByRole('button', { name: /save/i }));
            await waitFor(() => expect(mockedApi.post).toHaveBeenCalledWith('/circuit-map', expect.objectContaining({ circuit_name: 'New Circuit' })));
            await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
            expect(mockedApi.put).not.toHaveBeenCalled();
        });

        it('ignores an older map load that finishes after the selected map is removed', async () => {
            let resolveFirstMap!: (value: any) => void;
            const firstMap = { ...savedMap, id: 'first-map', circuit_name: 'First Circuit' };
            mockedApi.get.mockImplementation((url: string) => {
                if (url === '/circuit-map/list') return Promise.resolve({ data: { list: [firstMap, savedMap] }, status: 200 } as any);
                if (url === '/circuit-map/first-map') return new Promise((resolve) => { resolveFirstMap = resolve; });
                return Promise.resolve({ data: savedMap, status: 200 } as any);
            });
            renderCircuitMaps();
            await userEvent.click(await screen.findByRole('button', { name: 'First Circuit ACC' }));
            expect(screen.getByRole('button', { name: /remove map/i })).toBeDisabled();
            await userEvent.click(screen.getByRole('button', { name: 'Test Circuit ACC' }));
            await waitFor(() => expect(screen.getByRole('button', { name: /remove map/i })).toBeEnabled());
            const dialog = await openDeleteDialog();
            fireEvent.click(within(dialog).getByRole('button', { name: 'Remove map' }));
            await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
            await act(async () => { resolveFirstMap({ data: firstMap, status: 200 }); });
            expect(screen.getByLabelText('Circuit name')).toHaveValue('');
            expect(screen.getByText('0 samples')).toBeInTheDocument();
            expect(screen.queryByRole('button', { name: 'Test Circuit ACC' })).not.toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'First Circuit ACC' })).toBeInTheDocument();
        });

        it('waits for an active list refresh before allowing removal', async () => {
            await selectSavedMap();
            let resolveList!: (value: any) => void;
            mockedApi.get.mockReturnValueOnce(new Promise((resolve) => { resolveList = resolve; }));
            await userEvent.click(screen.getByRole('button', { name: /refresh/i }));
            expect(screen.getByRole('button', { name: /remove map/i })).toBeDisabled();
            await act(async () => { resolveList({ data: { list: [savedMap] }, status: 200 }); });
            expect(screen.getByRole('button', { name: /remove map/i })).toBeEnabled();
        });

        it('preserves the map on failure and allows retry', async () => {
            mockedApi.delete.mockRejectedValueOnce(new Error('Network unavailable'));
            await selectSavedMap();
            const dialog = await openDeleteDialog();
            fireEvent.click(within(dialog).getByRole('button', { name: 'Remove map' }));
            expect(await within(dialog).findByRole('alert')).toHaveTextContent('Could not remove this map. Please try again.');
            expect(screen.getByLabelText('Circuit name')).toHaveValue('Test Circuit');
            expect(screen.getByText('1 samples')).toBeInTheDocument();
            expect(mockRemoveCachedCircuitMap).not.toHaveBeenCalled();
            fireEvent.click(within(dialog).getByRole('button', { name: 'Remove map' }));
            await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
            expect(mockedApi.delete).toHaveBeenCalledTimes(2);
            expect(screen.getByText('No global maps found.')).toBeInTheDocument();
        });

        it('clears a stale entry when the API confirms the map was already removed', async () => {
            mockedApi.delete.mockRejectedValue({ status: 404, data: { message: 'Circuit map not found' } });
            await selectSavedMap();
            const dialog = await openDeleteDialog();
            fireEvent.click(within(dialog).getByRole('button', { name: 'Remove map' }));
            await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
            expect(mockRemoveCachedCircuitMap).toHaveBeenCalledWith('map/1');
            expect(screen.getByText('No global maps found.')).toBeInTheDocument();
        });

        it('does not treat an unknown endpoint as a successful removal', async () => {
            mockedApi.delete.mockRejectedValue({ status: 404, data: { message: 'Cannot DELETE /circuit-map/map%2F1' } });
            await selectSavedMap();
            const dialog = await openDeleteDialog();
            fireEvent.click(within(dialog).getByRole('button', { name: 'Remove map' }));
            expect(await within(dialog).findByRole('alert')).toBeInTheDocument();
            expect(mockRemoveCachedCircuitMap).not.toHaveBeenCalled();
        });
    });
});
