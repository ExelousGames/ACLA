import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CircuitMaps from '../circuit-maps';
import apiService from 'services/api.service';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
import { LiveSessionRuntime } from 'views/live-session/live-session-types';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { RecordingState } from 'views/lap-analysis/recording-state';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';

const mockRefreshCircuitMaps = jest.fn();
const mockUpsertCachedCircuitMap = jest.fn();

jest.mock('@radix-ui/themes', () => ({
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
        Root: ({ value, onValueChange, children }: any) => (
            <select value={value} onChange={(event) => onValueChange(event.target.value)}>
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
    },
}));

jest.mock('contexts/CircuitMapsContext', () => ({
    useCircuitMaps: () => ({
        refreshCircuitMaps: mockRefreshCircuitMaps,
        upsertCachedCircuitMap: mockUpsertCachedCircuitMap,
    }),
}));

const mockedApi = apiService as jest.Mocked<typeof apiService>;

const baseContext: LiveSessionRuntime = {
    currentTelemetry: {},
    telemetryStatus: null,
    staticData: {},
    recordingState: RecordingState.CHECKING,
    recordingMetadata: null,
    recordingFileKey: null,
    recordedSampleCount: 0,
    sessionIntelligence: new SessionIntelligence(),
    liveRangeTodoListHandle: null,
    liveRangeTodoListSnapshot: null,
    setCurrentTelemetry: jest.fn(),
    setStaticData: jest.fn(),
    setRecordingMetadata: jest.fn(),
    transitionRecordingState: jest.fn(),
    appendTelemetrySample: jest.fn(),
    readRecordedTelemetry: jest.fn(),
    finalizeRecordingWrites: jest.fn(),
    clearRecordingSession: jest.fn(),
    registerLiveRangeTodoListHandle: jest.fn(),
    publishLiveRangeTodoListSnapshot: jest.fn(),
};

const renderCircuitMaps = (context: Partial<LiveSessionRuntime> = {}) => (
    render(
        <LiveSessionContext.Provider value={{ ...baseContext, ...context }}>
            <CircuitMaps />
        </LiveSessionContext.Provider>
    )
);

describe('CircuitMaps', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        mockedApi.get.mockResolvedValue({ data: { list: [] }, status: 200 } as any);
        mockedApi.post.mockResolvedValue({ data: { id: 'map-1' }, status: 201 } as any);
        mockedApi.put.mockResolvedValue({ data: {}, status: 200 } as any);
        mockRefreshCircuitMaps.mockResolvedValue([]);

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

    it('loads global maps without a user id and disables ACC capture when telemetry is offline', async () => {
        renderCircuitMaps();

        await waitFor(() => expect(mockedApi.get).toHaveBeenCalledWith('/circuit-map/list', { game: 'acc' }));
        expect(JSON.stringify(mockedApi.get.mock.calls)).not.toContain('user_id');
        expect(screen.getByRole('button', { name: /start capture/i })).toBeDisabled();
    });

    it('saves a new global map payload without user ownership', async () => {
        renderCircuitMaps({
            telemetryStatus: ACC_STATUS.ACC_LIVE,
            currentTelemetry: {
                Graphics_status: ACC_STATUS.ACC_LIVE,
                Graphics_normalized_car_position: 0.1,
                Graphics_car_coordinates: JSON.stringify([{ x: 1, y: 0, z: 2 }]),
            },
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

    it('captures pit lane samples as an active circuit map mode', async () => {
        renderCircuitMaps();

        await userEvent.type(screen.getByLabelText('Circuit name'), 'Pit Test Circuit');
        await userEvent.selectOptions(screen.getAllByRole('combobox')[1], 'pit_lane');
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
                pit_lane: [{
                    bin: 420,
                    normalized_position: 0.42,
                    x: 12,
                    z: 34,
                    locked: true,
                }],
            },
        });
    });

    it('switches Other games into manual edit mode', async () => {
        renderCircuitMaps();
        await screen.findByText('No global maps found.');

        await userEvent.selectOptions(screen.getAllByRole('combobox')[0], 'other');

        expect(await screen.findByText('Manual Edit')).toBeInTheDocument();
        await waitFor(() => expect(screen.queryByText('Loading maps')).not.toBeInTheDocument());
        expect(screen.queryByText('ACC Offline')).not.toBeInTheDocument();
    });
});
