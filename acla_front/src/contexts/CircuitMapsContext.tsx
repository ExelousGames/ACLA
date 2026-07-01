import React, { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import {
    CircuitMapDto,
    CircuitMapGame,
    CircuitMapSummaryDto
} from 'views/circuit-maps/circuit-map-types';
import {
    fetchCircuitMapById,
    fetchCircuitMapList
} from 'services/circuitMapService';

type CircuitMapRequestOptions = {
    forceRefresh?: boolean;
};

type CircuitMapSummaryState = Record<CircuitMapGame, CircuitMapSummaryDto[]>;
type CircuitMapLoadingState = Record<CircuitMapGame, boolean>;

interface CircuitMapsContextType {
    mapSummaries: CircuitMapSummaryState;
    cachedMaps: Record<string, CircuitMapDto>;
    listLoading: CircuitMapLoadingState;
    error: string | null;
    refreshCircuitMaps: (game?: CircuitMapGame) => Promise<CircuitMapSummaryDto[]>;
    getCircuitMapById: (id: string, options?: CircuitMapRequestOptions) => Promise<CircuitMapDto | null>;
    getCircuitMapByTrack: (
        game: CircuitMapGame,
        sourceTrackKey: string | null | undefined,
        options?: CircuitMapRequestOptions
    ) => Promise<CircuitMapDto | null>;
    clearCircuitMapCache: (id?: string) => void;
    upsertCachedCircuitMap: (map: CircuitMapDto) => void;
}

const DEFAULT_SUMMARIES: CircuitMapSummaryState = {
    acc: [],
    other: []
};

const DEFAULT_LOADING: CircuitMapLoadingState = {
    acc: false,
    other: false
};

const CircuitMapsContext = createContext<CircuitMapsContextType | undefined>(undefined);

const getErrorMessage = (error: unknown): string => {
    if (error instanceof Error) {
        return error.message;
    }

    if (error && typeof error === 'object' && 'message' in error) {
        return String((error as { message?: unknown }).message || 'Unable to load circuit maps');
    }

    return 'Unable to load circuit maps';
};

const upsertSummary = (
    summaries: CircuitMapSummaryState,
    map: CircuitMapDto
): CircuitMapSummaryState => {
    const nextSummary: CircuitMapSummaryDto = {
        id: map.id,
        game: map.game,
        circuit_name: map.circuit_name,
        source_track_key: map.source_track_key ?? null,
        updated_at: map.updated_at ?? null,
        sample_count: map.sample_count
    };
    const currentGameSummaries = summaries[map.game] || [];
    const index = currentGameSummaries.findIndex((summary) => summary.id === map.id);
    const nextGameSummaries = index >= 0
        ? [
            ...currentGameSummaries.slice(0, index),
            nextSummary,
            ...currentGameSummaries.slice(index + 1)
        ]
        : [...currentGameSummaries, nextSummary];

    return {
        ...summaries,
        [map.game]: nextGameSummaries
    };
};

const CircuitMapsProvider = ({ children }: { children: ReactNode }) => {
    const [mapSummaries, setMapSummaries] = useState<CircuitMapSummaryState>(DEFAULT_SUMMARIES);
    const [cachedMaps, setCachedMaps] = useState<Record<string, CircuitMapDto>>({});
    const [listLoading, setListLoading] = useState<CircuitMapLoadingState>(DEFAULT_LOADING);
    const [error, setError] = useState<string | null>(null);
    const cachedMapsRef = useRef<Record<string, CircuitMapDto>>({});
    const pendingMapRequestsRef = useRef<Map<string, Promise<CircuitMapDto | null>>>(new Map());

    const setCachedMap = useCallback((map: CircuitMapDto) => {
        if (!map.id) return;

        cachedMapsRef.current = {
            ...cachedMapsRef.current,
            [map.id]: map
        };
        setCachedMaps(cachedMapsRef.current);
        setMapSummaries((previous) => upsertSummary(previous, map));
    }, []);

    const refreshCircuitMaps = useCallback(async (game: CircuitMapGame = 'acc') => {
        setListLoading((previous) => ({ ...previous, [game]: true }));
        setError(null);

        try {
            const nextSummaries = await fetchCircuitMapList(game);
            setMapSummaries((previous) => ({
                ...previous,
                [game]: nextSummaries
            }));
            return nextSummaries;
        } catch (err) {
            setError(getErrorMessage(err));
            return [];
        } finally {
            setListLoading((previous) => ({ ...previous, [game]: false }));
        }
    }, []);

    const getCircuitMapById = useCallback(async (
        id: string,
        options: CircuitMapRequestOptions = {}
    ): Promise<CircuitMapDto | null> => {
        if (!id) return null;

        if (!options.forceRefresh && cachedMapsRef.current[id]) {
            return cachedMapsRef.current[id];
        }

        if (!options.forceRefresh && pendingMapRequestsRef.current.has(id)) {
            return pendingMapRequestsRef.current.get(id) || null;
        }

        setError(null);
        const request = fetchCircuitMapById(id)
            .then((map) => {
                setCachedMap(map);
                return map;
            })
            .catch((err) => {
                setError(getErrorMessage(err));
                return null;
            })
            .finally(() => {
                pendingMapRequestsRef.current.delete(id);
            });

        pendingMapRequestsRef.current.set(id, request);
        return request;
    }, [setCachedMap]);

    const getCircuitMapByTrack = useCallback(async (
        game: CircuitMapGame,
        sourceTrackKey: string | null | undefined,
        options: CircuitMapRequestOptions = {}
    ): Promise<CircuitMapDto | null> => {
        if (!sourceTrackKey) return null;

        const currentSummaries = mapSummaries[game] || [];
        const summaries = currentSummaries.length > 0
            ? currentSummaries
            : await refreshCircuitMaps(game);
        const summary = summaries.find((item) => item.game === game && item.source_track_key === sourceTrackKey);

        if (!summary) return null;

        return getCircuitMapById(summary.id, options);
    }, [getCircuitMapById, mapSummaries, refreshCircuitMaps]);

    const clearCircuitMapCache = useCallback((id?: string) => {
        if (!id) {
            cachedMapsRef.current = {};
            pendingMapRequestsRef.current.clear();
            setCachedMaps({});
            return;
        }

        const { [id]: _removed, ...nextCache } = cachedMapsRef.current;
        cachedMapsRef.current = nextCache;
        pendingMapRequestsRef.current.delete(id);
        setCachedMaps(nextCache);
    }, []);

    const upsertCachedCircuitMap = useCallback((map: CircuitMapDto) => {
        setCachedMap(map);
    }, [setCachedMap]);

    useEffect(() => {
        void refreshCircuitMaps('acc');
    }, [refreshCircuitMaps]);

    const value = useMemo<CircuitMapsContextType>(() => ({
        mapSummaries,
        cachedMaps,
        listLoading,
        error,
        refreshCircuitMaps,
        getCircuitMapById,
        getCircuitMapByTrack,
        clearCircuitMapCache,
        upsertCachedCircuitMap
    }), [
        cachedMaps,
        clearCircuitMapCache,
        error,
        getCircuitMapById,
        getCircuitMapByTrack,
        listLoading,
        mapSummaries,
        refreshCircuitMaps,
        upsertCachedCircuitMap
    ]);

    return (
        <CircuitMapsContext.Provider value={value}>
            {children}
        </CircuitMapsContext.Provider>
    );
};

export default CircuitMapsProvider;

export const useCircuitMaps = (): CircuitMapsContextType => {
    const context = useContext(CircuitMapsContext);
    if (!context) {
        throw new Error('useCircuitMaps must be used within a CircuitMapsProvider');
    }
    return context;
};
