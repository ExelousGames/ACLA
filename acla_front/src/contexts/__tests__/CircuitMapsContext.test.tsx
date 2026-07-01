import React, { useEffect } from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import CircuitMapsProvider, { useCircuitMaps } from 'contexts/CircuitMapsContext';
import apiService from 'services/api.service';

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: {
        get: jest.fn(),
    },
}));

const mockedApi = apiService as jest.Mocked<typeof apiService>;

const TestConsumer = ({ onContext }: { onContext: (context: ReturnType<typeof useCircuitMaps>) => void }) => {
    const context = useCircuitMaps();

    useEffect(() => {
        onContext(context);
    }, [context, onContext]);

    return (
        <div>
            <span data-testid="acc-count">{context.mapSummaries.acc.length}</span>
            <span data-testid="error">{context.error || ''}</span>
        </div>
    );
};

const renderProvider = (onContext: (context: ReturnType<typeof useCircuitMaps>) => void) => (
    render(
        <CircuitMapsProvider>
            <TestConsumer onContext={onContext} />
        </CircuitMapsProvider>
    )
);

const mapSummary = {
    id: 'map-1',
    game: 'acc',
    circuit_name: 'Brands Hatch Circuit',
    source_track_key: 'brands_hatch',
    sample_count: 3,
};

const fullMap = {
    ...mapSummary,
    resolution: 1000,
    samples: {
        left_boundary: [{ bin: 1, normalized_position: 0.001, x: 1, y: 0, z: 2, sample_count: 1, updated_at: 'now' }],
        right_boundary: [],
        pit_lane: [],
    },
};

describe('CircuitMapsContext', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        mockedApi.get.mockResolvedValue({ data: { list: [] }, status: 200 } as any);
    });

    it('preloads the ACC circuit map list after mount', async () => {
        mockedApi.get.mockResolvedValueOnce({ data: { list: [mapSummary] }, status: 200 } as any);
        let latestContext: any = null;

        renderProvider((context) => {
            latestContext = context;
        });

        await waitFor(() => expect(screen.getByTestId('acc-count')).toHaveTextContent('1'));
        expect(mockedApi.get).toHaveBeenCalledWith('/circuit-map/list', { game: 'acc' });
        expect(latestContext?.mapSummaries.acc[0]).toMatchObject(mapSummary);
    });

    it('fetches a map by id once, then returns cached data until forced to refresh', async () => {
        mockedApi.get.mockImplementation((url: string) => {
            if (url === '/circuit-map/list') {
                return Promise.resolve({ data: { list: [] }, status: 200 } as any);
            }
            return Promise.resolve({ data: fullMap, status: 200 } as any);
        });
        let latestContext: any = null;

        renderProvider((context) => {
            latestContext = context;
        });

        await waitFor(() => expect(latestContext).not.toBeNull());

        let firstMap = null;
        let secondMap = null;
        let refreshedMap = null;
        await act(async () => {
            firstMap = await latestContext!.getCircuitMapById('map-1');
            secondMap = await latestContext!.getCircuitMapById('map-1');
            refreshedMap = await latestContext!.getCircuitMapById('map-1', { forceRefresh: true });
        });

        expect(firstMap).toMatchObject(fullMap);
        expect(secondMap).toMatchObject(fullMap);
        expect(refreshedMap).toMatchObject(fullMap);
        expect(mockedApi.get.mock.calls.filter(([url]) => url === '/circuit-map/map-1')).toHaveLength(2);
    });

    it('deduplicates concurrent map detail requests for the same id', async () => {
        let resolveMap: (value: any) => void = () => {};
        const detailRequest = new Promise((resolve) => {
            resolveMap = resolve;
        });
        mockedApi.get.mockImplementation((url: string) => {
            if (url === '/circuit-map/list') {
                return Promise.resolve({ data: { list: [] }, status: 200 } as any);
            }
            return detailRequest as any;
        });
        let latestContext: any = null;

        renderProvider((context) => {
            latestContext = context;
        });

        await waitFor(() => expect(latestContext).not.toBeNull());

        let firstRequest: Promise<any> = Promise.resolve(null);
        let secondRequest: Promise<any> = Promise.resolve(null);
        act(() => {
            firstRequest = latestContext!.getCircuitMapById('map-1');
            secondRequest = latestContext!.getCircuitMapById('map-1');
        });

        let firstMap = null;
        let secondMap = null;
        await act(async () => {
            resolveMap({ data: fullMap, status: 200 });
            [firstMap, secondMap] = await Promise.all([firstRequest, secondRequest]);
        });

        expect(firstMap).toMatchObject(fullMap);
        expect(secondMap).toMatchObject(fullMap);
        expect(mockedApi.get.mock.calls.filter(([url]) => url === '/circuit-map/map-1')).toHaveLength(1);
    });

    it('finds a map by game and source track key, and returns null for missing tracks', async () => {
        mockedApi.get.mockImplementation((url: string) => {
            if (url === '/circuit-map/list') {
                return Promise.resolve({ data: { list: [mapSummary] }, status: 200 } as any);
            }
            return Promise.resolve({ data: fullMap, status: 200 } as any);
        });
        let latestContext: any = null;

        renderProvider((context) => {
            latestContext = context;
        });

        await waitFor(() => expect(screen.getByTestId('acc-count')).toHaveTextContent('1'));

        let matchedMap = null;
        let missingMap = null;
        await act(async () => {
            matchedMap = await latestContext!.getCircuitMapByTrack('acc', 'brands_hatch');
            missingMap = await latestContext!.getCircuitMapByTrack('acc', 'spa');
        });

        expect(matchedMap).toMatchObject(fullMap);
        expect(missingMap).toBeNull();
        expect(mockedApi.get.mock.calls.filter(([url]) => url === '/circuit-map/map-1')).toHaveLength(1);
    });

    it('stores errors without blocking future retries', async () => {
        mockedApi.get.mockImplementation((url: string) => {
            if (url === '/circuit-map/list') {
                return Promise.resolve({ data: { list: [] }, status: 200 } as any);
            }
            return Promise.reject({ message: 'Map unavailable' });
        });
        let latestContext: any = null;

        renderProvider((context) => {
            latestContext = context;
        });

        await waitFor(() => expect(latestContext).not.toBeNull());

        let failedMap = null;
        await act(async () => {
            failedMap = await latestContext!.getCircuitMapById('map-1');
        });

        expect(failedMap).toBeNull();
        await waitFor(() => expect(screen.getByTestId('error')).toHaveTextContent('Map unavailable'));

        mockedApi.get.mockImplementation((url: string) => {
            if (url === '/circuit-map/list') {
                return Promise.resolve({ data: { list: [] }, status: 200 } as any);
            }
            return Promise.resolve({ data: fullMap, status: 200 } as any);
        });

        let retriedMap = null;
        await act(async () => {
            retriedMap = await latestContext!.getCircuitMapById('map-1');
        });

        expect(retriedMap).toMatchObject(fullMap);
        await waitFor(() => expect(screen.getByTestId('error')).toHaveTextContent(''));
    });
});
