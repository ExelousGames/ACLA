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
    CheckboxGroup: jest.requireActual('@radix-ui/themes').CheckboxGroup,
    Tabs: jest.requireActual('radix-ui').Tabs,
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
            <select value={value} disabled={disabled} onChange={(event) => onValueChange(event.target.value)}
                aria-label={require('react').Children.toArray(children).find((child: any) => child.props['aria-label'])?.props['aria-label']}>
                {children}
            </select>
        ),
        Trigger: ({ placeholder }: any) => placeholder ? <option value="" disabled>{placeholder}</option> : null,
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
    Static_track: imported.track,
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

const selectCaptureMode = async (mode: string) => {
    await userEvent.click(screen.getByRole('tab', { name: mode === 'middle_line' ? 'Centerline Map' : 'Bounded Map' }));
    await userEvent.selectOptions(screen.getAllByRole('combobox')[1], mode);
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

const canvasPointer = (canvas: HTMLElement, type: string, clientX: number, clientY: number, pointerId = 1, button = 0) => {
    const event = new MouseEvent(type, { bubbles: true, clientX, clientY, button });
    Object.defineProperty(event, 'pointerId', { value: pointerId });
    fireEvent(canvas, event);
};

const clickCanvas = (canvas: HTMLElement, clientX: number, clientY: number) => {
    canvasPointer(canvas, 'pointerdown', clientX, clientY);
    canvasPointer(canvas, 'pointerup', clientX, clientY);
};

const dragCanvas = (canvas: HTMLElement, startX: number, startY: number, endX: number, endY: number) => {
    canvasPointer(canvas, 'pointerdown', startX, startY);
    canvasPointer(canvas, 'pointermove', endX, endY);
    canvasPointer(canvas, 'pointerup', endX, endY);
};

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
            unobserve = jest.fn();
            disconnect = jest.fn();
        };

        HTMLCanvasElement.prototype.setPointerCapture = jest.fn();
        HTMLCanvasElement.prototype.hasPointerCapture = jest.fn(() => true);
        HTMLCanvasElement.prototype.releasePointerCapture = jest.fn();

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
        await selectCaptureMode(mode);
        await openIRacingImport();
        expect(screen.getByRole('button', { name: /save/i })).toBeDisabled();
        expect(screen.getByRole('tab', { name: 'Bounded Map' })).toBeDisabled();
        expect(screen.getByRole('tab', { name: 'Centerline Map' })).toBeDisabled();
        await act(async () => finishImportRead());
        expect(screen.getByLabelText('Circuit name')).toHaveValue(imported.track);
        expect(screen.getByRole('status')).toHaveTextContent('imported 2 driver samples');
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(imported.filePath);
        await userEvent.click(screen.getByRole('button', { name: /save/i }));
        await waitFor(() => expect(mockedApi.post).toHaveBeenCalledWith('/circuit-map', expect.objectContaining({
            game: 'iracing', circuit_name: imported.track, source_track_key: imported.track,
            samples: expect.objectContaining({ [mode]: [expect.objectContaining({ bin: 250, normalized_position: 0.25, x: 10, y: 2, z: 30, sample_count: 2 })] }),
        })));
        expect(mockUpsertCachedCircuitMap).toHaveBeenCalledWith(expect.objectContaining({ game: 'iracing' }));
        expect(mockRefreshCircuitMaps).toHaveBeenCalledWith('iracing');
        expect(mockedApi.get).toHaveBeenCalledWith('/circuit-map/list', { game: 'iracing' });
    });

    it('leaves a draft unchanged when file selection is cancelled', async () => {
        renderCircuitMaps();
        await openIRacingImport();
        await act(async () => finishImportRead());
        (window.electronAPI.importLocalIRacingTelemetry as jest.Mock).mockResolvedValue(null);
        await userEvent.clear(screen.getByLabelText('Circuit name'));
        await userEvent.type(screen.getByLabelText('Circuit name'), 'My draft');
        await userEvent.click(screen.getByRole('button', { name: /open .ibt file/i }));
        await waitFor(() => expect(screen.getByRole('button', { name: /open .ibt file/i })).toBeEnabled());
        expect(screen.getByLabelText('Circuit name')).toHaveValue('My draft');
        expect(screen.getByText(/1 samples/)).toBeInTheDocument();
        expect(window.electronAPI.startRecordedFileRead).toHaveBeenCalledTimes(1);
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
        await userEvent.click(screen.getByRole('button', { name: /open .ibt file/i }));
        await waitFor(() => expect(window.electronAPI.startRecordedFileRead).toHaveBeenCalledTimes(2));
        await act(async () => finishImportRead([{ ...importedRow, Static_track: 'monza' }]));
        expect(await screen.findByRole('alert')).toHaveTextContent('different circuit');
        expect(screen.getByLabelText('Circuit name')).toHaveValue(imported.track);
        expect(screen.getByText(`1 samples / ${imported.track}`)).toBeInTheDocument();
        expect(window.electronAPI.cancelRecordedFileRead).toHaveBeenCalledWith('map-read');
    });

    it('uses converted telemetry track fields even when import metadata disagrees', async () => {
        (window.electronAPI.importLocalIRacingTelemetry as jest.Mock).mockResolvedValue({ ...imported, track: 'Metadata Track' });
        renderCircuitMaps();
        await openIRacingImport();
        await act(async () => finishImportRead());
        expect(screen.getByLabelText('Circuit name')).toHaveValue(importedRow.Static_track);
        await userEvent.click(screen.getByRole('button', { name: /save/i }));
        await waitFor(() => expect(mockedApi.post).toHaveBeenCalledWith('/circuit-map', expect.objectContaining({
            circuit_name: importedRow.Static_track, source_track_key: importedRow.Static_track,
        })));
    });

    it('retains the track from the first chunk when later rows omit static fields', async () => {
        renderCircuitMaps();
        await openIRacingImport();
        await act(async () => {
            readListener({ type: 'chunk', readId: 'map-read', rows: [importedRow] });
            finishImportRead([{ ...importedRow, Static_track: '' }]);
        });
        expect(screen.getByLabelText('Circuit name')).toHaveValue(importedRow.Static_track);
        expect(screen.getByRole('status')).toHaveTextContent('imported 2 driver samples');
    });

    it('rejects telemetry without a track name instead of falling back to import metadata', async () => {
        renderCircuitMaps();
        await openIRacingImport();
        await act(async () => finishImportRead([{ ...importedRow, Static_track: '' }]));
        expect(screen.getByRole('alert')).toHaveTextContent('no track name in its telemetry');
        expect(screen.getByText('0 samples')).toBeInTheDocument();
        expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(imported.filePath);
    });

    it('discards the whole import when the track changes between chunks', async () => {
        renderCircuitMaps();
        await openIRacingImport();
        await act(async () => {
            readListener({ type: 'chunk', readId: 'map-read', rows: [importedRow] });
            finishImportRead([{ ...importedRow, Static_track: 'monza' }]);
        });
        expect(screen.getByRole('alert')).toHaveTextContent('different circuit');
        expect(screen.getByText('0 samples')).toBeInTheDocument();
        expect(screen.getByLabelText('Circuit name')).toHaveValue('');
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

    it('does not label or capture an ACC map using iRacing live telemetry', async () => {
        renderCircuitMaps({ sessionGame: 'iracing', staticData: { Static_track: imported.track } });
        act(() => liveTelemetryStore.publishFrame({
            type: 'frame', game: 'iracing', sequence: 1, committedSequence: 1, committedCount: 1,
            sample: {
                ...importedRow,
                Graphics_car_id: Array.from({ length: 60 }, (_, slot) => slot === 0 ? 63 : -1),
                Graphics_car_coordinates: Array.from({ length: 60 }, (_, slot) => slot === 0
                    ? { x: 10, y: 2, z: 30 } : { x: 0, y: 0, z: 0 }),
            },
        }));
        await screen.findByText('No global maps found.');
        expect(screen.getByLabelText('Circuit name')).toHaveValue('');
        expect(screen.getByRole('button', { name: /start capture/i })).toBeDisabled();
        expect(screen.getByText('0 samples')).toBeInTheDocument();
    });

    it('does not label or capture an ACC map using iRacing live telemetry', async () => {
        renderCircuitMaps({ sessionGame: 'iracing', staticData: { Static_track: imported.track } });
        act(() => liveTelemetryStore.publishFrame({
            type: 'frame', game: 'iracing', sequence: 1, committedSequence: 1, committedCount: 1,
            sample: {
                ...importedRow,
                Graphics_car_id: Array.from({ length: 60 }, (_, slot) => slot === 0 ? 63 : -1),
                Graphics_car_coordinates: Array.from({ length: 60 }, (_, slot) => slot === 0
                    ? { x: 10, y: 2, z: 30 } : { x: 0, y: 0, z: 0 }),
            },
        }));
        await screen.findByText('No global maps found.');
        expect(screen.getByLabelText('Circuit name')).toHaveValue('');
        expect(screen.getByRole('button', { name: /start capture/i })).toBeDisabled();
        expect(screen.getByText('0 samples')).toBeInTheDocument();
    });

    it.each([
        [{ Static_track: 'Canonical Circuit' }, 'Canonical Circuit'],
        [{ track: 'Legacy Circuit' }, ''],
        [{ Static: { track: 'Legacy Circuit' } }, ''],
        [{ Statics: { track: 'Legacy Circuit' } }, ''],
    ])('uses only canonical static fields from the session: %o', async (staticData, expectedName) => {
        renderCircuitMaps({ staticData, sessionGame: 'acc' });

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

    it.each(['left_boundary', 'middle_line'])('processes all 120 live capture frames in %s mode in one React batch', async (mode) => {
        renderCircuitMaps();
        await waitFor(() => expect(mockedApi.get).toHaveBeenCalledWith('/circuit-map/list', { game: 'acc' }));
        await userEvent.type(screen.getByLabelText('Circuit name'), 'Live Test Circuit');
        await selectCaptureMode(mode);
        act(() => {
            liveTelemetryStore.publishFrame({
                type: 'frame',
                game: 'acc',
                sample: {
                    Graphics_status: ACC_STATUS.ACC_LIVE,
                    Graphics_normalized_car_position: 0,
                    Graphics_player_car_id: 42,
                    Graphics_car_id: Array.from({ length: 60 }, (_, slot) => slot === 0 ? 42 : -1),
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
                        Graphics_player_car_id: 42,
                        Graphics_car_id: Array.from({ length: 60 }, (_, slot) => slot === 0 ? 42 : -1),
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
        expect(await screen.findByText('1 samples')).toBeInTheDocument();
        await selectCaptureMode('middle_line');
        clickCanvas(screen.getByLabelText('Centerline map'), 450, 310);
        expect(screen.getByText('Normalized position 0.42')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /^check lock$/i }));
        await userEvent.click(screen.getByRole('button', { name: /^close unlock$/i }));
        await userEvent.click(screen.getByRole('button', { name: /save/i }));

        await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/middle-map', expect.objectContaining({
            samples: {
                middle_line: [expect.objectContaining({ ...sample, locked: false, updated_at: expect.any(String) })],
                left_boundary: savedMap.samples.left_boundary,
                right_boundary: [],
                pit_lane: [],
            },
        })));
        expect(mockUpsertCachedCircuitMap).toHaveBeenLastCalledWith(expect.objectContaining({
            sample_count: 2,
            samples: expect.objectContaining({ middle_line: [expect.objectContaining({ bin: 420, locked: false })] }),
        }));
        await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
        await userEvent.click(screen.getByRole('button', { name: /^trash delete$/i }));
        expect(screen.getByText('0 samples')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /new map/i }));
        expect(screen.getByText('0 samples')).toBeInTheDocument();
    });

    it.each([
        { game: 'acc', startY: 42, endY: 578 },
        { game: 'iracing', startY: 578, endY: 42 },
    ])('renders $game paths with the correct Z orientation and keeps points fixed in each map tab', async ({ game, startY, endY }) => {
        const sample = { bin: 0, normalized_position: 0, x: 0, y: 0, z: 0, sample_count: 1 };
        const savedMap = {
            id: 'two-maps', game, circuit_name: 'Two Maps', resolution: 1000,
            samples: {
                left_boundary: [sample],
                right_boundary: [{ ...sample, x: 100, z: 100 }],
                pit_lane: [{ ...sample, x: 50, z: 50 }],
                middle_line: [
                    { ...sample, x: 10000, z: 10000 },
                    { ...sample, bin: 500, normalized_position: 0.5, x: 10020, z: 10020 },
                ],
            },
        };
        mockedApi.get.mockImplementation(async (url: string) => ({
            data: url === '/circuit-map/list' ? { list: [savedMap] } : savedMap,
            status: 200,
        } as any));
        renderCircuitMaps();
        await userEvent.selectOptions(screen.getAllByRole('combobox')[0], game);
        await userEvent.click(await screen.findByRole('button', { name: `Two Maps ${game.toUpperCase()}` }));
        expect(await screen.findByText('3 samples')).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Bounded Map' })).toHaveAttribute('aria-selected', 'true');
        expect(screen.getByRole('tabpanel', { name: 'Bounded Map' })).toBeInTheDocument();
        expect(within(screen.getAllByRole('combobox')[1]).getAllByRole('option').map((option) => option.textContent))
            .toEqual(['Left Boundary', 'Right Boundary', 'Pit Lane']);
        const contexts = (HTMLCanvasElement.prototype.getContext as jest.Mock).mock.results;
        const boundedContext = contexts[contexts.length - 1].value;
        expect(boundedContext.arc).toHaveBeenCalledTimes(3);
        expect(boundedContext.arc).toHaveBeenNthCalledWith(1, 182, startY, 4, 0, Math.PI * 2);
        expect(boundedContext.arc).toHaveBeenNthCalledWith(2, 718, endY, 4, 0, Math.PI * 2);

        await userEvent.click(screen.getByRole('tab', { name: 'Centerline Map' }));
        expect(screen.getByRole('tabpanel', { name: 'Centerline Map' })).toBeInTheDocument();
        expect(screen.getByText('2 samples')).toBeInTheDocument();
        expect(screen.getAllByRole('combobox')[1]).toHaveValue('middle_line');
        expect(within(screen.getAllByRole('combobox')[1]).getAllByRole('option').map((option) => option.textContent))
            .toEqual(['Middle Line']);
        const centerlineContext = contexts[contexts.length - 1].value;
        expect(centerlineContext.arc).toHaveBeenCalledTimes(3);
        expect(centerlineContext.arc).toHaveBeenNthCalledWith(1, 182, startY, 4, 0, Math.PI * 2);
        expect(centerlineContext.arc).toHaveBeenNthCalledWith(2, 718, endY, 4, 0, Math.PI * 2);
        expect(centerlineContext.arc).toHaveBeenNthCalledWith(3, 182, startY, 10, 0, Math.PI * 2);
        expect(centerlineContext.fillText).toHaveBeenCalledWith('START', 182, startY - 18);

        const centerlineCanvas = screen.getByLabelText('Centerline map');
        clickCanvas(centerlineCanvas, 182, startY);
        expect(screen.getByText('Normalized position 0')).toBeInTheDocument();
        canvasPointer(centerlineCanvas, 'pointerdown', 182, startY);
        canvasPointer(centerlineCanvas, 'pointermove', 450, 310);
        canvasPointer(centerlineCanvas, 'pointerup', 450, 310);

        await userEvent.click(screen.getByRole('tab', { name: 'Bounded Map' }));
        expect(screen.getByText('3 samples')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /save/i }));
        await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/two-maps', expect.objectContaining({ samples: savedMap.samples })));
        await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());

        const canvas = screen.getByLabelText('Bounded map');
        clickCanvas(canvas, 450, 310);
        expect(screen.getByText('Bin 0')).toBeInTheDocument();
        canvasPointer(canvas, 'pointerdown', 450, 310);
        canvasPointer(canvas, 'pointermove', 450, 444);
        canvasPointer(canvas, 'pointerup', 450, 444);
        await userEvent.click(screen.getByRole('button', { name: /save/i }));
        await waitFor(() => expect(mockedApi.put).toHaveBeenLastCalledWith('/circuit-map/two-maps', expect.objectContaining({
            samples: savedMap.samples,
        })));
        await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
    });

    it('keeps edits in both maps and clears the selection when switching tabs', async () => {
        const sample = { bin: 0, normalized_position: 0, x: 0, y: 0, z: 0, sample_count: 1 };
        const savedMap = {
            id: 'both-maps', game: 'acc', circuit_name: 'Both Maps', resolution: 1000,
            samples: { left_boundary: [sample], middle_line: [sample] },
        };
        mockedApi.get.mockImplementation(async (url: string) => ({
            data: url === '/circuit-map/list' ? { list: [savedMap] } : savedMap,
            status: 200,
        } as any));
        renderCircuitMaps();
        await userEvent.click(await screen.findByRole('button', { name: 'Both Maps ACC' }));
        expect(await screen.findByText('1 samples')).toBeInTheDocument();
        clickCanvas(screen.getByLabelText('Bounded map'), 450, 310);
        await userEvent.click(screen.getByRole('button', { name: /^check lock$/i }));
        expect(screen.getByRole('button', { name: /^trash delete$/i })).toBeInTheDocument();
        await userEvent.click(screen.getByRole('tab', { name: 'Centerline Map' }));
        expect(screen.queryByRole('button', { name: /^trash delete$/i })).not.toBeInTheDocument();
        expect(screen.getByText('1 samples')).toBeInTheDocument();
        clickCanvas(screen.getByLabelText('Centerline map'), 450, 310);
        await userEvent.click(screen.getByRole('button', { name: /^check lock$/i }));
        await userEvent.click(screen.getByRole('tab', { name: 'Bounded Map' }));
        expect(screen.queryByRole('button', { name: /^trash delete$/i })).not.toBeInTheDocument();
        expect(screen.getByText('1 samples')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /save/i }));
        await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/both-maps', expect.objectContaining({
            samples: {
                left_boundary: [expect.objectContaining({ bin: 0, locked: true })],
                middle_line: [expect.objectContaining({ bin: 0, normalized_position: 0, locked: true })],
                right_boundary: [], pit_lane: [],
            },
        })));
        await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
    });

    it('pauses live capture on a tab change and can capture the same position into the centerline', async () => {
        renderCircuitMaps();
        await screen.findByText('No global maps found.');
        act(() => liveTelemetryStore.publishFrame({
            type: 'frame', game: 'acc', sequence: 1, committedSequence: 1, committedCount: 1,
            sample: {
                Graphics_status: ACC_STATUS.ACC_LIVE,
                Graphics_normalized_car_position: 0.25,
                Graphics_player_car_id: 42,
                Graphics_car_id: Array.from({ length: 60 }, (_, slot) => slot === 0 ? 42 : -1),
                Graphics_car_coordinates: Array.from({ length: 60 }, (_, slot) => slot === 0
                    ? { x: 10, y: 2, z: 30 } : { x: 0, y: 0, z: 0 }),
                Static_track: 'Live Circuit',
            },
        }));
        await userEvent.click(screen.getByRole('button', { name: /start capture/i }));
        expect(screen.getByText('1 samples / Live Circuit')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('tab', { name: 'Centerline Map' }));
        expect(screen.queryByRole('button', { name: /pause capture/i })).not.toBeInTheDocument();
        expect(screen.getByText('0 samples / Live Circuit')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /start capture/i }));
        expect(screen.getByText('1 samples / Live Circuit')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: /pause capture/i }));
        await userEvent.click(screen.getByRole('button', { name: /save/i }));
        await waitFor(() => expect(mockedApi.post).toHaveBeenCalledWith('/circuit-map', expect.objectContaining({
            samples: {
                left_boundary: [expect.objectContaining({ normalized_position: 0.25 })],
                middle_line: [expect.objectContaining({ normalized_position: 0.25 })],
                right_boundary: [], pit_lane: [],
            },
        })));
        await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
    });

    it('offers only ACC and iRacing games', async () => {
        renderCircuitMaps();
        await screen.findByText('No global maps found.');

        expect(within(screen.getAllByRole('combobox')[0]).getAllByRole('option').map((option) => option.textContent))
            .toEqual(['ACC', 'iRacing']);
    });

    describe('map navigation', () => {
        const samples = [
            { bin: 0, normalized_position: 0, x: 0, y: 0, z: 0, sample_count: 1 },
            { bin: 500, normalized_position: 0.5, x: 100, y: 0, z: 100, sample_count: 1 },
        ];
        const openZoomMap = async (game = 'acc') => {
            const map = {
                id: 'zoom-map', game, circuit_name: 'Zoom Circuit', resolution: 1000,
                samples: { left_boundary: samples, middle_line: samples, right_boundary: [], pit_lane: [] },
            };
            mockedApi.get.mockImplementation(async (url: string) => ({
                data: url === '/circuit-map/list' ? { list: [map] } : map, status: 200,
            } as any));
            renderCircuitMaps();
            await userEvent.selectOptions(screen.getAllByRole('combobox')[0], game);
            await userEvent.click(await screen.findByRole('button', { name: `Zoom Circuit ${game.toUpperCase()}` }));
            expect(await screen.findByText('2 samples')).toBeInTheDocument();
            return map;
        };
        const expectFirstPointAt = (x: number, y: number) => {
            const contexts = (HTMLCanvasElement.prototype.getContext as jest.Mock).mock.results;
            const [screenX, screenY] = contexts[contexts.length - 1].value.arc.mock.calls[0];
            expect(screenX).toBeCloseTo(x);
            expect(screenY).toBeCloseTo(y);
        };

        it.each([
            { game: 'acc', tab: 'Bounded Map', canvas: 'Bounded map', direction: 1 },
            { game: 'acc', tab: 'Centerline Map', canvas: 'Centerline map', direction: 1 },
            { game: 'iracing', tab: 'Bounded Map', canvas: 'Bounded map', direction: -1 },
            { game: 'iracing', tab: 'Centerline Map', canvas: 'Centerline map', direction: -1 },
        ])('zooms $game $tab and selects projected points without changing saved coordinates', async ({ game, tab, canvas, direction }) => {
            const map = await openZoomMap(game);
            await userEvent.click(screen.getByRole('tab', { name: tab }));
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('100%');
            await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('125%');
            expectFirstPointAt(115, 310 - 335 * direction);

            await userEvent.click(screen.getByRole('button', { name: 'Zoom out' }));
            await userEvent.click(screen.getByRole('button', { name: 'Zoom out' }));
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('80%');
            expectFirstPointAt(235.6, 310 - 214.4 * direction);
            clickCanvas(screen.getByLabelText(canvas), 235.6, 310 - 214.4 * direction);
            expect(screen.getByText('Bin 0')).toBeInTheDocument();

            await userEvent.click(screen.getByRole('button', { name: 'Fit map' }));
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('100%');
            expectFirstPointAt(182, 310 - 268 * direction);
            await userEvent.click(screen.getByRole('button', { name: /save/i }));
            await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/zoom-map', expect.objectContaining({ samples: map.samples })));
        });

        it.each([
            { game: 'acc', tab: 'Bounded Map', canvas: 'Bounded map', direction: 1 },
            { game: 'acc', tab: 'Centerline Map', canvas: 'Centerline map', direction: 1 },
            { game: 'iracing', tab: 'Bounded Map', canvas: 'Bounded map', direction: -1 },
            { game: 'iracing', tab: 'Centerline Map', canvas: 'Centerline map', direction: -1 },
        ])('pans the zoomed $game $tab and selects points without changing saved coordinates', async ({ game, tab, canvas: canvasLabel, direction }) => {
            const map = await openZoomMap(game);
            await userEvent.click(screen.getByRole('tab', { name: tab }));
            await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
            const canvas = screen.getByLabelText(canvasLabel);
            dragCanvas(canvas, 450, 310, 500, 310 + 80 * direction);
            expectFirstPointAt(165, 310 - 255 * direction);
            expect(screen.queryByText('Bin 0')).not.toBeInTheDocument();
            expect(canvas.setPointerCapture).toHaveBeenCalledWith(1);
            expect(canvas.releasePointerCapture).toHaveBeenCalledWith(1);
            expect(canvas).not.toHaveClass('circuit-maps__canvas--panning');
            clickCanvas(canvas, 165, 310 - 255 * direction);
            expect(screen.getByText('Bin 0')).toBeInTheDocument();
            await userEvent.click(screen.getByRole('button', { name: /save/i }));
            await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/zoom-map', expect.objectContaining({ samples: map.samples })));

            await userEvent.click(screen.getByRole('button', { name: 'Zoom out' }));
            expectFirstPointAt(232, 310 - 188 * direction);
            await userEvent.click(screen.getByRole('button', { name: 'Fit map' }));
            expectFirstPointAt(182, 310 - 268 * direction);
        });

        it('tolerates click jitter and preserves the selection when dragging a different point', async () => {
            await openZoomMap();
            const canvas = screen.getByLabelText('Bounded map');
            dragCanvas(canvas, 182, 42, 184, 43);
            expectFirstPointAt(182, 42);
            expect(screen.getByText('Bin 0')).toBeInTheDocument();

            canvasPointer(canvas, 'pointerdown', 718, 578);
            expect(canvas).toHaveClass('circuit-maps__canvas--panning');
            canvasPointer(canvas, 'pointerdown', 400, 300, 2);
            canvasPointer(canvas, 'pointermove', 900, 700, 2);
            canvasPointer(canvas, 'pointerup', 900, 700, 2);
            expectFirstPointAt(182, 42);
            canvasPointer(canvas, 'pointermove', 1000, 700);
            expectFirstPointAt(464, 164);
            canvasPointer(canvas, 'pointerup', 1000, 700);
            expect(screen.getByText('Bin 0')).toBeInTheDocument();
            expect(screen.queryByText('Bin 500')).not.toBeInTheDocument();
        });

        it.each(['pointercancel', 'lostpointercapture'])('stops panning after %s without selecting a point', async (eventType) => {
            await openZoomMap();
            const canvas = screen.getByLabelText('Bounded map');
            canvasPointer(canvas, 'pointerdown', 182, 42);
            canvasPointer(canvas, 'pointermove', 232, 62);
            canvasPointer(canvas, eventType, 232, 62);
            canvasPointer(canvas, 'pointermove', 450, 310);
            canvasPointer(canvas, 'pointerup', 450, 310);
            expectFirstPointAt(232, 62);
            expect(canvas).not.toHaveClass('circuit-maps__canvas--panning');
            expect(screen.queryByText('Bin 0')).not.toBeInTheDocument();
            dragCanvas(canvas, 450, 310, 500, 330);
            expectFirstPointAt(282, 82);
        });

        it('bounds the zoom and restores fit when switching tabs, maps, drafts, or games', async () => {
            await openZoomMap();
            for (let index = 0; index < 12; index += 1) fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('600%');
            expect(screen.getByRole('button', { name: 'Zoom in' })).toBeDisabled();
            for (let index = 0; index < 20; index += 1) fireEvent.click(screen.getByRole('button', { name: 'Zoom out' }));
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('35%');
            expect(screen.getByRole('button', { name: 'Zoom out' })).toBeDisabled();

            dragCanvas(screen.getByLabelText('Bounded map'), 450, 310, 500, 330);
            await userEvent.click(screen.getByRole('tab', { name: 'Centerline Map' }));
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('100%');
            expectFirstPointAt(182, 42);
            await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
            dragCanvas(screen.getByLabelText('Centerline map'), 450, 310, 500, 330);
            await userEvent.click(screen.getByRole('button', { name: /new map/i }));
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('100%');
            await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
            dragCanvas(screen.getByLabelText('Centerline map'), 450, 310, 500, 330);
            await userEvent.click(screen.getByRole('button', { name: 'Zoom Circuit ACC' }));
            expect(await screen.findByText('2 samples')).toBeInTheDocument();
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('100%');
            expectFirstPointAt(182, 42);
            await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
            await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'iracing');
            expect(screen.getByLabelText('Zoom level')).toHaveTextContent('100%');
        });
    });

    describe('centerline range tags', () => {
        const tagOptions = ['corner', 'slow', 'fast', 'long straight'];
        const samples = [
            { bin: 0, normalized_position: 0, x: 0, z: 0 },
            { bin: 250, normalized_position: 0.25, x: 100, z: 0 },
            { bin: 500, normalized_position: 0.5, x: 100, z: 100 },
            { bin: 750, normalized_position: 0.75, x: 0, z: 100 },
        ].map((sample) => ({ ...sample, y: 0, sample_count: 1, updated_at: '2026-10-06' }));
        const tag = { id: 'saved-tag', label: 'Final corner', start_position: 0.5, end_position: 0.75 };
        const openTaggedMap = async (
            tags: typeof tag[] = [],
            loadTagOptions = async (): Promise<any> => ({ data: { tags: tagOptions }, status: 200 }),
        ) => {
            let savedMap = {
                id: 'tagged-map', game: 'acc', circuit_name: 'Tagged Circuit', resolution: 1000,
                samples: { middle_line: samples, left_boundary: [samples[0]], right_boundary: [], pit_lane: [] },
                centerline_tags: tags,
            };
            mockedApi.get.mockImplementation(async (url: string) => url === '/circuit-map/centerline-tags'
                ? loadTagOptions()
                : { data: url === '/circuit-map/list' ? { list: [savedMap] } : savedMap, status: 200 } as any);
            mockedApi.put.mockImplementation(async (_url, payload: any) => {
                savedMap = { ...savedMap, ...payload };
                return { data: savedMap, status: 200 } as any;
            });
            renderCircuitMaps();
            await userEvent.click(await screen.findByRole('button', { name: 'Tagged Circuit ACC' }));
            expect(await screen.findByText('1 samples')).toBeInTheDocument();
            await userEvent.click(screen.getByRole('tab', { name: 'Centerline Map' }));
        };
        const clickMap = (clientX: number, clientY: number) => {
            clickCanvas(screen.getByLabelText('Centerline map'), clientX, clientY);
        };

        it('pans while selecting a range and highlights it at zoomed and panned coordinates', async () => {
            await openTaggedMap();
            await waitFor(() => expect(screen.getByRole('button', { name: 'Select range' })).toBeEnabled());
            await userEvent.click(screen.getByRole('button', { name: 'Zoom out' }));
            await userEvent.click(screen.getByRole('button', { name: 'Select range' }));
            dragCanvas(screen.getByLabelText('Centerline map'), 235.6, 95.6, 285.6, 115.6);
            expect(screen.getByRole('status')).toHaveTextContent('Click the range start');
            clickMap(285.6, 115.6);
            clickMap(714.4, 544.4);
            expect(screen.getByRole('status')).toHaveTextContent('0.0% → 50.0%');
            await userEvent.click(screen.getByRole('checkbox', { name: 'corner' }));
            await userEvent.click(screen.getByRole('button', { name: 'Add tags' }));
            expect(screen.getByRole('button', { name: 'corner 0.0% → 50.0%' })).toBeInTheDocument();
            const contexts = (HTMLCanvasElement.prototype.getContext as jest.Mock).mock.results;
            const labelCall = contexts[contexts.length - 1].value.fillText.mock.calls.find(([label]: [string]) => label === 'corner');
            expect(labelCall[1]).toBeCloseTo(714.4);
            expect(labelCall[2]).toBeCloseTo(97.6);
        });

        it.each(tagOptions)('selects a range, highlights it, and saves/reloads %s without changing samples', async (label) => {
            await openTaggedMap();
            await waitFor(() => expect(screen.getByRole('button', { name: 'Select range' })).toBeEnabled());
            await userEvent.click(screen.getByRole('button', { name: 'Select range' }));
            clickMap(450, 310);
            expect(screen.getByRole('status')).toHaveTextContent('Click the range start');
            clickMap(182, 42);
            expect(screen.getByRole('status')).toHaveTextContent('Click a different point');
            clickMap(182, 42);
            expect(screen.queryByRole('button', { name: 'Add tags' })).not.toBeInTheDocument();
            clickMap(718, 578);
            expect(screen.getByRole('status')).toHaveTextContent('0.0% → 50.0%');
            expect(screen.queryByText('Bin 0')).not.toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Add tags' })).toBeDisabled();
            const tagSelect = screen.getByRole('group', { name: 'Range tags' });
            expect(within(tagSelect).getAllByRole('checkbox').map((option) => option.getAttribute('value'))).toEqual(tagOptions);
            expect(mockedApi.get).toHaveBeenCalledWith('/circuit-map/centerline-tags');
            await userEvent.click(within(tagSelect).getByRole('checkbox', { name: label }));
            await userEvent.click(screen.getByRole('button', { name: 'Add tags' }));
            expect(screen.getByRole('button', { name: `${label} 0.0% → 50.0%` })).toHaveAttribute('aria-pressed', 'true');
            const contexts = (HTMLCanvasElement.prototype.getContext as jest.Mock).mock.results;
            expect(contexts[contexts.length - 1].value.fillText).toHaveBeenCalledWith(label, 718, 24, 240);
            await userEvent.click(screen.getByRole('button', { name: /save/i }));
            const expectedTag = expect.objectContaining({ id: expect.any(String), label, start_position: 0, end_position: 0.5 });
            await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/tagged-map', expect.objectContaining({
                samples: { middle_line: samples, left_boundary: [samples[0]], right_boundary: [], pit_lane: [] },
                centerline_tags: [expectedTag],
            })));
            expect(mockUpsertCachedCircuitMap).toHaveBeenLastCalledWith(expect.objectContaining({ centerline_tags: [expectedTag] }));
            await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
            await userEvent.click(screen.getByRole('button', { name: /new map/i }));
            expect(screen.queryByText(label)).not.toBeInTheDocument();
            await userEvent.click(screen.getByRole('button', { name: 'Tagged Circuit ACC' }));
            expect(await screen.findByText(label)).toBeInTheDocument();
        });

        it('adds multiple checked tags to one range and saves/reloads them independently', async () => {
            await openTaggedMap([tag]);
            await waitFor(() => expect(screen.getByRole('button', { name: 'Select range' })).toBeEnabled());
            await userEvent.click(screen.getByRole('button', { name: 'Select range' }));
            clickMap(182, 42);
            clickMap(718, 578);
            await userEvent.click(screen.getByRole('checkbox', { name: 'corner' }));
            await userEvent.click(screen.getByRole('checkbox', { name: 'slow' }));
            await userEvent.click(screen.getByRole('checkbox', { name: 'fast' }));
            await userEvent.click(screen.getByRole('checkbox', { name: 'fast' }));
            expect(screen.getByRole('checkbox', { name: 'corner' })).toBeChecked();
            expect(screen.getByRole('checkbox', { name: 'slow' })).toBeChecked();
            expect(screen.getByRole('checkbox', { name: 'fast' })).not.toBeChecked();
            await userEvent.click(screen.getByRole('button', { name: 'Add tags' }));
            expect(screen.getByRole('button', { name: 'corner 0.0% → 50.0%' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'slow 0.0% → 50.0%' })).toBeInTheDocument();

            await userEvent.click(screen.getByRole('button', { name: 'Select range' }));
            clickMap(182, 42);
            clickMap(718, 578);
            screen.getAllByRole('checkbox').forEach((checkbox) => expect(checkbox).not.toBeChecked());
            expect(screen.getByRole('button', { name: 'Add tags' })).toBeDisabled();
            await userEvent.click(screen.getByRole('checkbox', { name: 'fast' }));
            await userEvent.click(screen.getByRole('checkbox', { name: 'fast' }));
            expect(screen.getByRole('button', { name: 'Add tags' })).toBeDisabled();
            await userEvent.click(screen.getByRole('button', { name: 'Cancel selection' }));

            await userEvent.click(screen.getByRole('button', { name: /save/i }));
            const expectedTags = [tag, ...['corner', 'slow'].map((label) => expect.objectContaining({
                id: expect.any(String), label, start_position: 0, end_position: 0.5,
            }))];
            await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/tagged-map', expect.objectContaining({
                samples: { middle_line: samples, left_boundary: [samples[0]], right_boundary: [], pit_lane: [] },
                centerline_tags: expectedTags,
            })));
            const savedTags = (mockedApi.put.mock.calls[0][1] as any).centerline_tags;
            expect(new Set(savedTags.map((savedTag: typeof tag) => savedTag.id)).size).toBe(3);
            await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
            await userEvent.click(screen.getByRole('button', { name: /new map/i }));
            await userEvent.click(screen.getByRole('button', { name: 'Tagged Circuit ACC' }));
            expect(await screen.findByRole('button', { name: 'corner 0.0% → 50.0%' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'slow 0.0% → 50.0%' })).toBeInTheDocument();
            await userEvent.click(screen.getByRole('button', { name: 'Remove tag slow' }));
            expect(screen.getByRole('button', { name: 'corner 0.0% → 50.0%' })).toBeInTheDocument();
            expect(screen.queryByRole('button', { name: 'slow 0.0% → 50.0%' })).not.toBeInTheDocument();
        });

        it('waits for the backend list and displays the options it returns', async () => {
            let resolveOptions!: (value: any) => void;
            await openTaggedMap([], () => new Promise((resolve) => { resolveOptions = resolve; }));
            expect(screen.getByText('Loading tags...')).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Select range' })).toBeDisabled();
            await act(async () => resolveOptions({ data: { tags: ['backend tag'] }, status: 200 }));
            await userEvent.click(screen.getByRole('button', { name: 'Select range' }));
            clickMap(182, 42);
            clickMap(718, 578);
            const tagSelect = screen.getByRole('group', { name: 'Range tags' });
            expect(within(tagSelect).getAllByRole('checkbox')).toHaveLength(1);
            expect(within(tagSelect).getByRole('checkbox', { name: 'backend tag' })).toBeInTheDocument();
        });

        it('allows retrying a failed tag list request while preserving existing tags', async () => {
            const loadOptions = jest.fn().mockRejectedValueOnce(new Error('Offline'))
                .mockResolvedValue({ data: { tags: tagOptions }, status: 200 });
            await openTaggedMap([tag], loadOptions);
            expect(await screen.findByRole('alert')).toHaveTextContent('Unable to load centerline tags.');
            expect(screen.getByRole('button', { name: 'Select range' })).toBeDisabled();
            expect(screen.getByText('Final corner')).toBeInTheDocument();
            await userEvent.click(screen.getByRole('button', { name: 'Retry loading tags' }));
            await waitFor(() => expect(screen.getByRole('button', { name: 'Select range' })).toBeEnabled());
            expect(screen.queryByRole('alert')).not.toBeInTheDocument();
            expect(loadOptions).toHaveBeenCalledTimes(2);
        });

        it('disables new range tags when the backend list is empty', async () => {
            await openTaggedMap([], async () => ({ data: { tags: [] }, status: 200 }));
            expect(await screen.findByText('No centerline tags available.')).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Select range' })).toBeDisabled();
        });

        it('selects ranges across start/finish, swaps their direction, and keeps tags when saving the bounded map', async () => {
            await openTaggedMap([tag]);
            await waitFor(() => expect(screen.getByRole('button', { name: 'Select range' })).toBeEnabled());
            await userEvent.click(screen.getByRole('button', { name: 'Select range' }));
            clickMap(182, 578);
            clickMap(718, 42);
            expect(screen.getByRole('status')).toHaveTextContent('75.0% → 25.0% (across start/finish)');
            await userEvent.click(screen.getByRole('button', { name: 'Swap start/end' }));
            expect(screen.getByRole('status')).toHaveTextContent('25.0% → 75.0%');
            await userEvent.click(screen.getByRole('button', { name: 'Swap start/end' }));
            await userEvent.click(screen.getByRole('checkbox', { name: 'long straight' }));
            await userEvent.click(screen.getByRole('button', { name: 'Add tags' }));
            await userEvent.click(screen.getByRole('tab', { name: 'Bounded Map' }));
            expect(screen.queryByRole('button', { name: 'Select range' })).not.toBeInTheDocument();
            await userEvent.click(screen.getByRole('button', { name: /save/i }));
            await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/tagged-map', expect.objectContaining({
                centerline_tags: [tag, expect.objectContaining({ label: 'long straight', start_position: 0.75, end_position: 0.25 })],
            })));
            await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
        });

        it('cancels unfinished selections on tab changes and saves tag removal', async () => {
            await openTaggedMap([tag]);
            await waitFor(() => expect(screen.getByRole('button', { name: 'Select range' })).toBeEnabled());
            await userEvent.click(screen.getByRole('button', { name: /Final corner 50.0%/ }));
            expect(screen.getByRole('button', { name: /Final corner 50.0%/ })).toHaveAttribute('aria-pressed', 'true');
            await userEvent.click(screen.getByRole('button', { name: 'Select range' }));
            clickMap(182, 42);
            clickMap(718, 578);
            await userEvent.click(screen.getByRole('checkbox', { name: 'corner' }));
            await userEvent.click(screen.getByRole('tab', { name: 'Bounded Map' }));
            await userEvent.click(screen.getByRole('tab', { name: 'Centerline Map' }));
            expect(screen.queryByRole('button', { name: 'Cancel selection' })).not.toBeInTheDocument();
            await waitFor(() => expect(screen.getByRole('button', { name: 'Select range' })).toBeEnabled());
            await userEvent.click(screen.getByRole('button', { name: 'Select range' }));
            clickMap(182, 42);
            clickMap(718, 578);
            screen.getAllByRole('checkbox').forEach((checkbox) => expect(checkbox).not.toBeChecked());
            expect(screen.getByRole('button', { name: 'Add tags' })).toBeDisabled();
            await userEvent.click(screen.getByRole('button', { name: 'Cancel selection' }));
            await userEvent.click(screen.getByRole('button', { name: 'Remove tag Final corner' }));
            expect(screen.getByText('No range tags yet.')).toBeInTheDocument();
            await userEvent.click(screen.getByRole('button', { name: /save/i }));
            await waitFor(() => expect(mockedApi.put).toHaveBeenCalledWith('/circuit-map/tagged-map', expect.objectContaining({ centerline_tags: [] })));
            await waitFor(() => expect(screen.getByRole('button', { name: /save/i })).toBeEnabled());
        });
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
