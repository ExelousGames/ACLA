import React, { createContext, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import { getNextRecordingState, RecordingEvent, RecordingState, StopReason } from 'views/lap-analysis/recording-state';
import {
    detectLiveSessionType,
    getTelemetryCar,
    getTelemetryLap,
    getTelemetryPosition,
    getTelemetryTrack,
} from 'views/lap-analysis/session-intelligence/live-performance-analyst';
import {
    getCornersForTrack,
    getNextCorner as getNextTrackCorner,
} from 'views/lap-analysis/session-intelligence/track-corners';
import type { CornerLookahead } from 'views/lap-analysis/session-intelligence/types';
import {
    LocalTelemetryFileValidation,
    LiveRecordingMetadata,
    LiveSessionRecorderControl,
    LiveSessionRestorationStatus,
    LiveSessionRuntime,
    LiveSessionSnapshot,
    LiveSessionStaticData,
    PERSISTED_LIVE_SESSION_DRAFT_VERSION,
    RecordedFileReadEvent,
    RecordingStartResult,
    RecordingStopResult,
    RecordingViewUpdate,
    StandardTelemetrySample,
} from './live-session-types';
import {
    getPersistedLiveSessionDraft,
    normalizeLiveSessionOwnerEmail,
    removePersistedLiveSessionDraft,
    savePersistedLiveSessionDraft,
} from './live-session-draft-storage';
import {
    AppendLiveSessionAnalysisResultPageInput,
    AppendLiveSessionAnalysisResultPageResult,
    createLiveSessionAnalysisResultPage,
    LiveSessionAnalysisResultPage,
} from './live-session-analysis-results';
import { normalizeAnalysisResultsData } from 'views/lap-analysis/visualization/charts/analysisResultsModel';
import { liveTelemetryStore } from './live-telemetry-store';

const RESTORED_RECORDING_ERROR = 'The local recording file is missing or unreadable. Upload is unavailable; discard this draft to clear it.';

const isAbsoluteFilePath = (filePath: string): boolean =>
    /^(?:[a-zA-Z]:[\\/]|\\\\|\/)/.test(filePath);

const missingProvider = () => console.warn('No provider for LiveSessionContext');
const getEmptyLiveSessionSnapshot = (): LiveSessionSnapshot => ({
    status: 'empty',
    track: '',
    car: '',
    current_lap: 0,
    completed_laps: 0,
    normalized_position: 0,
    sample_count: 0,
    live_session_type: 'unknown',
    completed_lap_count: 0,
});

const defaultRuntime: LiveSessionRuntime = {
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
    getNextCorner: () => null,
    getLiveSessionSnapshot: getEmptyLiveSessionSnapshot,
    startLiveSession: missingProvider,
    endLiveSession: missingProvider,
    setRecordingMetadata: missingProvider,
    transitionRecordingState: missingProvider,
    startRecordingSession: async () => {
        missingProvider();
        return { ok: false, error: { type: 'unsupported-recording-game', message: 'Recording is unavailable.' } };
    },
    stopRecordingSession: async () => {
        missingProvider();
        return null;
    },
    streamRecordedTelemetry: async () => {
        missingProvider();
        return { rowCount: 0, totalBytes: 0 };
    },
    clearRecordingSession: missingProvider,
    clearPersistedDraft: missingProvider,
    registerRecorderControl: missingProvider,
    appendAnalysisResultPage: () => {
        missingProvider();
        return { pageId: '', pageCount: 0 };
    },
    selectAnalysisResultPage: () => {
        missingProvider();
        return false;
    },
    updateActiveAnalysisResultPage: () => {
        missingProvider();
        return false;
    },
};

export const LiveSessionContext = createContext<LiveSessionRuntime>(defaultRuntime);

export const LiveSessionProvider = ({
    children,
    ownerEmail,
}: {
    children: React.ReactNode;
    ownerEmail?: string | null;
}) => {
    const normalizedOwnerEmail = normalizeLiveSessionOwnerEmail(ownerEmail);
    const [sessionGame, setSessionGame] = useState<DesktopGame | null>(null);
    const [staticData, setStaticDataState] = useState<LiveSessionStaticData>({});
    const [recordingState, setRecordingState] = useState(RecordingState.CHECKING);
    const [recordingMetadata, setRecordingMetadataState] = useState<LiveRecordingMetadata | null>(null);
    const [recordingFileKey, setRecordingFileKeyState] = useState<string | null>(null);
    const [recordingActive, setRecordingActive] = useState(false);
    const [recordingGame, setRecordingGame] = useState<DesktopGame | null>(null);
    const [restorationStatus, setRestorationStatus] = useState<LiveSessionRestorationStatus>(
        normalizedOwnerEmail ? 'restoring' : 'idle',
    );
    const [restorationError, setRestorationError] = useState<string | null>(null);
    const [recordingFileValidation, setRecordingFileValidation] = useState<LocalTelemetryFileValidation | null>(null);
    const [recorderControl, setRecorderControl] = useState<LiveSessionRecorderControl | null>(null);
    const [analysisResultPages, setAnalysisResultPages] = useState<LiveSessionAnalysisResultPage[]>([]);
    const [activeAnalysisResultPageId, setActiveAnalysisResultPageId] = useState<string | null>(null);

    const sessionGameRef = useRef<DesktopGame | null>(null);
    const ownerEmailRef = useRef(normalizedOwnerEmail);
    const recordingStateRef = useRef(RecordingState.CHECKING);
    const recordingMetadataRef = useRef<LiveRecordingMetadata | null>(null);
    const staticDataRef = useRef<LiveSessionStaticData>({});
    const recordingFileKeyRef = useRef<string | null>(null);
    const recordingActiveRef = useRef(false);
    const recordingGameRef = useRef<DesktopGame | null>(null);
    const recordingStartPromiseRef = useRef<Promise<RecordingStartResult> | null>(null);
    const recordingStopPromiseRef = useRef<Promise<RecordingStopResult | null> | null>(null);
    const analysisResultPagesRef = useRef<LiveSessionAnalysisResultPage[]>([]);
    const activeAnalysisResultPageIdRef = useRef<string | null>(null);
    const persistDraftRef = useRef<() => void>(() => undefined);
    const draftPersistenceSuppressedRef = useRef(false);
    const registerRecorderControl = useCallback((control: LiveSessionRecorderControl | null) => {
        setRecorderControl(control);
    }, []);

    const clearAnalysisResultPages = useCallback(() => {
        analysisResultPagesRef.current = [];
        activeAnalysisResultPageIdRef.current = null;
        setAnalysisResultPages([]);
        setActiveAnalysisResultPageId(null);
    }, []);

    const appendAnalysisResultPage = useCallback((
        input: AppendLiveSessionAnalysisResultPageInput,
    ): AppendLiveSessionAnalysisResultPageResult => {
        const page = createLiveSessionAnalysisResultPage(input);
        const nextPages = [...analysisResultPagesRef.current, page];
        analysisResultPagesRef.current = nextPages;
        setAnalysisResultPages(nextPages);

        if (activeAnalysisResultPageIdRef.current === null) {
            activeAnalysisResultPageIdRef.current = page.id;
            setActiveAnalysisResultPageId(page.id);
        }

        return { pageId: page.id, pageCount: nextPages.length };
    }, []);

    const selectAnalysisResultPage = useCallback((pageId: string): boolean => {
        if (!analysisResultPagesRef.current.some((page) => page.id === pageId)) return false;
        activeAnalysisResultPageIdRef.current = pageId;
        setActiveAnalysisResultPageId(pageId);
        return true;
    }, []);

    const updateActiveAnalysisResultPage = useCallback((data: unknown): boolean => {
        const activePageId = activeAnalysisResultPageIdRef.current;
        if (
            !activePageId
            || !analysisResultPagesRef.current.some((page) => page.id === activePageId)
        ) return false;
        const normalized = normalizeAnalysisResultsData(data);
        const nextPages = analysisResultPagesRef.current.map((page) => (
            page.id === activePageId ? { ...page, elements: normalized.elements } : page
        ));
        analysisResultPagesRef.current = nextPages;
        setAnalysisResultPages(nextPages);
        return true;
    }, []);

    const persistCurrentDraft = useCallback(() => {
        const currentOwnerEmail = ownerEmailRef.current;
        const currentGame = sessionGameRef.current;
        const currentMetadata = recordingMetadataRef.current;
        const currentFilePath = recordingFileKeyRef.current;
        const currentState = recordingStateRef.current;
        if (
            draftPersistenceSuppressedRef.current
            || !currentOwnerEmail
            || !currentGame
            || !currentMetadata
            || !currentFilePath
            || ![
                RecordingState.RECORDING,
                RecordingState.HOLDING,
                RecordingState.RESUME_READY,
                RecordingState.UPLOAD_READY,
            ].includes(currentState)
        ) {
            return;
        }

        savePersistedLiveSessionDraft({
            version: PERSISTED_LIVE_SESSION_DRAFT_VERSION,
            ownerEmail: currentOwnerEmail,
            sessionGame: currentGame,
            recordingMetadata: currentMetadata,
            telemetryFilePath: currentFilePath,
            recordedSampleCount: liveTelemetryStore.getSnapshot().committedSampleCount,
            lastRuntimeState: currentState,
            updatedAt: new Date().toISOString(),
        });
    }, []);
    persistDraftRef.current = persistCurrentDraft;

    const clearPersistedDraft = useCallback(() => {
        draftPersistenceSuppressedRef.current = true;
        const currentOwnerEmail = ownerEmailRef.current;
        if (currentOwnerEmail) removePersistedLiveSessionDraft(currentOwnerEmail);
        clearAnalysisResultPages();
    }, [clearAnalysisResultPages]);

    const getLatestTelemetrySample = useCallback(() => {
        const latest = liveTelemetryStore.getSnapshot().currentTelemetry;
        return latest && Object.keys(latest).length > 0 ? latest : null;
    }, []);

    const getLiveSessionSnapshot = useCallback((): LiveSessionSnapshot => {
        const latest = getLatestTelemetrySample();
        if (!latest) return getEmptyLiveSessionSnapshot();

        const currentLap = getTelemetryLap(latest);
        return {
            status: 'ready',
            track: getTelemetryTrack(latest),
            car: getTelemetryCar(latest),
            current_lap: currentLap,
            completed_laps: currentLap,
            normalized_position: getTelemetryPosition(latest) ?? 0,
            sample_count: liveTelemetryStore.getSnapshot().committedSampleCount,
            live_session_type: detectLiveSessionType(latest),
            completed_lap_count: currentLap,
        };
    }, [getLatestTelemetrySample]);

    const getNextCorner = useCallback((): CornerLookahead | null => {
        const latest = getLatestTelemetrySample();
        if (!latest) return null;
        const currentPosition = getTelemetryPosition(latest) ?? 0;
        const corner = getNextTrackCorner(
            getCornersForTrack(getTelemetryTrack(latest)),
            currentPosition,
        );
        if (!corner) return null;

        const distanceAhead = corner.from > currentPosition
            ? corner.from - currentPosition
            : 1.0 - currentPosition + corner.from;
        return {
            name: corner.name,
            trackPosition: corner.from,
            distanceAhead,
        };
    }, [getLatestTelemetrySample]);

    const setRecordingMetadata = useCallback((metadata: LiveRecordingMetadata | null) => {
        recordingMetadataRef.current = metadata;
        setRecordingMetadataState(metadata);
        persistDraftRef.current();
    }, []);

    const transitionRecordingState = useCallback((event: RecordingEvent) => {
        const next = getNextRecordingState(recordingStateRef.current, event);
        recordingStateRef.current = next;
        setRecordingState(next);
        persistDraftRef.current();
    }, []);

    const setRecordingFileKey = useCallback((fileKey: string | null) => {
        recordingFileKeyRef.current = fileKey;
        setRecordingFileKeyState(fileKey);
        setRecordingFileValidation(null);
        persistDraftRef.current();
    }, []);

    const resetSampleCount = useCallback(() => {
        liveTelemetryStore.restoreCommittedSampleCount(0);
    }, []);
    const applyRecordingTerminal = useCallback((result: RecordingStopResult) => {
        if (!result || (recordingGameRef.current && result.game !== recordingGameRef.current)) return;
        recordingActiveRef.current = false;
        recordingGameRef.current = null;
        setRecordingActive(false);
        setRecordingGame(null);
        if (typeof result.filePath === 'string' && isAbsoluteFilePath(result.filePath)) {
            setRecordingFileKey(result.filePath);
        }
        if (typeof result.writtenSamples === 'number' && Number.isSafeInteger(result.writtenSamples)) {
            liveTelemetryStore.finalizeCommittedSampleCount(result.writtenSamples);
        }
        const hasPublishedFile = Boolean(result.filePath || recordingFileKeyRef.current);
        const nextState = hasPublishedFile ? RecordingState.UPLOAD_READY : RecordingState.READY;
        recordingStateRef.current = nextState;
        setRecordingState(nextState);
        persistDraftRef.current();
    }, [setRecordingFileKey]);

    const startRecordingSession = useCallback(async (game: DesktopGame): Promise<RecordingStartResult> => {
        if (sessionGameRef.current !== game) throw new Error('Recording game must match the active live session.');
        if (recordingActiveRef.current) throw new Error('A recording session is already active.');
        if (typeof window.electronAPI?.startRecordingSession !== 'function') {
            return Promise.resolve({
                ok: false,
                error: { type: 'unsupported-recording-game', message: 'Desktop recording is unavailable.' },
            });
        }
        recordingActiveRef.current = true;
        recordingGameRef.current = game;
        setRecordingActive(true);
        setRecordingGame(game);
        liveTelemetryStore.beginStream();

        const startPromise = Promise.resolve()
            .then(() => window.electronAPI.startRecordingSession({ game }))
            .then((result) => {
                if (!result.ok) {
                    recordingActiveRef.current = false;
                    recordingGameRef.current = null;
                    setRecordingActive(false);
                    setRecordingGame(null);
                    return result;
                }
                if (result.game !== game || !isAbsoluteFilePath(result.filePath)) {
                    throw new Error('Recording startup returned mismatched session data.');
                }
                setRecordingFileKey(result.filePath);
                resetSampleCount();
                transitionRecordingState({ type: 'recordingStarted' });
                return result;
            })
            .catch((error) => {
                recordingActiveRef.current = false;
                recordingGameRef.current = null;
                setRecordingActive(false);
                setRecordingGame(null);
                throw error;
            })
            .finally(() => {
                if (recordingStartPromiseRef.current === startPromise) {
                    recordingStartPromiseRef.current = null;
                }
            });
        recordingStartPromiseRef.current = startPromise;
        return startPromise;
    }, [resetSampleCount, setRecordingFileKey, transitionRecordingState]);

    const stopRecordingSession = useCallback(async (reason: StopReason = 'manual'): Promise<RecordingStopResult | null> => {
        if (recordingStopPromiseRef.current) return recordingStopPromiseRef.current;
        const startPromise = recordingStartPromiseRef.current;
        if (!recordingActiveRef.current && !startPromise) return null;
        if (typeof window.electronAPI?.stopRecordingSession !== 'function') {
            throw new Error('Desktop recording stop is unavailable.');
        }
        recordingStopPromiseRef.current = Promise.resolve(startPromise)
            .catch(() => null)
            .then(() => {
                if (!recordingActiveRef.current) return null;
                return window.electronAPI.stopRecordingSession();
            })
            .then((result) => {
                if (!result) return null;
                applyRecordingTerminal(result);
                if (reason === 'error' && !recordingFileKeyRef.current) {
                    transitionRecordingState({ type: 'recordingStopped', reason: 'error' });
                }
                return result;
            })
            .catch((error) => {
                const game = recordingGameRef.current;
                if (game) {
                    applyRecordingTerminal({
                        game,
                        error: error instanceof Error ? error.message : String(error),
                    });
                }
                throw error;
            })
            .finally(() => {
                recordingStopPromiseRef.current = null;
            });
        return recordingStopPromiseRef.current;
    }, [applyRecordingTerminal, transitionRecordingState]);

    useEffect(() => {
        const removeView = window.electronAPI?.onRecordingViewUpdate?.((update: RecordingViewUpdate) => {
            if (update.game !== sessionGameRef.current) return;
            let nextStaticData = staticDataRef.current;
            Object.entries(update.sample).forEach(([key, value]) => {
                if (!key.startsWith('Static_') || Object.prototype.hasOwnProperty.call(nextStaticData, key)) {
                    return;
                }
                if (nextStaticData === staticDataRef.current) nextStaticData = { ...staticDataRef.current };
                nextStaticData[key] = value;
            });
            if (!liveTelemetryStore.publishFrame(update, nextStaticData)) return;
            if (nextStaticData !== staticDataRef.current) {
                staticDataRef.current = nextStaticData;
                setStaticDataState(nextStaticData);
            }
        });
        const removeEnded = window.electronAPI?.onRecordingSessionEnded?.((result: RecordingStopResult) => {
            applyRecordingTerminal(result);
        });
        return () => {
            removeView?.();
            removeEnded?.();
        };
    }, [applyRecordingTerminal]);

    const runRecordedFileRead = useCallback(async (
        filePath: string,
        game: DesktopGame,
        purpose: 'validate' | 'consume',
        onChunk?: (rows: StandardTelemetrySample[]) => void | Promise<void>,
        onProgress?: (rowsRead: number, totalRows: number | null, bytesRead: number, totalBytes: number) => void,
    ): Promise<{ rowCount: number; totalBytes: number }> => {
        if (!window.electronAPI?.startRecordedFileRead || !window.electronAPI?.onRecordedFileReadEvent) {
            throw new Error('Recorded-file reader is unavailable.');
        }
        let readId: string | null = null;
        const queued: RecordedFileReadEvent[] = [];
        return new Promise((resolve, reject) => {
            let settled = false;
            let removeListener: () => void = () => undefined;
            const cleanup = () => removeListener();
            const finishError = (error: Error) => {
                if (settled) return;
                settled = true;
                cleanup();
                reject(error);
            };
            const handleEvent = (event: RecordedFileReadEvent): void | Promise<void> => {
                if (!readId) {
                    queued.push(event);
                    return;
                }
                if (event.readId !== readId || settled) return;
                if (event.type === 'chunk') {
                    try {
                        return Promise.resolve(onChunk?.(event.rows)).catch((error) => {
                            finishError(error instanceof Error ? error : new Error(String(error)));
                            if (readId) void window.electronAPI.cancelRecordedFileRead?.(readId).catch(() => undefined);
                        });
                    } catch (error) {
                        finishError(error instanceof Error ? error : new Error(String(error)));
                        if (readId) void window.electronAPI.cancelRecordedFileRead?.(readId).catch(() => undefined);
                        return;
                    }
                }
                else if (event.type === 'progress') {
                    onProgress?.(event.rowsRead, null, event.bytesRead, event.totalBytes);
                } else if (event.type === 'complete') {
                    settled = true;
                    cleanup();
                    resolve({ rowCount: event.rowCount, totalBytes: event.totalBytes });
                } else if (event.type === 'error') {
                    finishError(new Error(event.message));
                }
            };
            removeListener = window.electronAPI.onRecordedFileReadEvent(handleEvent);
            void window.electronAPI.startRecordedFileRead({ filePath, game, purpose }).then((result) => {
                readId = result.readId;
                for (const event of queued.splice(0)) void handleEvent(event);
            }).catch((error) => finishError(error instanceof Error ? error : new Error(String(error))));
        });
    }, []);

    const streamRecordedTelemetry = useCallback(async (
        onChunk: (rows: StandardTelemetrySample[]) => void | Promise<void>,
        onProgress?: (rowsRead: number, totalRows: number | null, bytesRead: number, totalBytes: number) => void,
    ) => {
        const filePath = recordingFileKeyRef.current;
        const game = sessionGameRef.current;
        if (!filePath || !game) return { rowCount: 0, totalBytes: 0 };
        if (recordingMetadataRef.current?.gameRecordedFrom !== game) {
            throw new Error('Recorded-file game does not match the authoritative session metadata.');
        }
        return runRecordedFileRead(filePath, game, 'consume', onChunk, onProgress);
    }, [runRecordedFileRead]);

    const clearRecordingSession = useCallback(() => {
        setRecordingFileKey(null);
        setRecordingMetadata(null);
        resetSampleCount();
        liveTelemetryStore.beginStream();
        clearAnalysisResultPages();
    }, [clearAnalysisResultPages, resetSampleCount, setRecordingFileKey, setRecordingMetadata]);

    const resetLiveSession = useCallback((nextGame: DesktopGame | null) => {
        sessionGameRef.current = nextGame;
        setSessionGame(nextGame);

        liveTelemetryStore.resetSession();
        staticDataRef.current = {};
        setStaticDataState({});
        recordingStateRef.current = RecordingState.CHECKING;
        setRecordingState(RecordingState.CHECKING);
        setRecordingMetadata(null);
        setRecordingFileKey(null);
        resetSampleCount();
        clearAnalysisResultPages();
        setRestorationError(null);
    }, [clearAnalysisResultPages, resetSampleCount, setRecordingFileKey, setRecordingMetadata]);

    const startLiveSession = useCallback((game: DesktopGame) => {
        if (sessionGameRef.current) return;
        const beginSession = () => {
            if (sessionGameRef.current) return;
            draftPersistenceSuppressedRef.current = false;
            resetLiveSession(game);
            setRestorationStatus('not-found');
        };
        if (recordingActiveRef.current || recordingStopPromiseRef.current) {
            void stopRecordingSession('complete').then(beginSession, beginSession);
        } else {
            beginSession();
        }
    }, [resetLiveSession, stopRecordingSession]);

    const endLiveSession = useCallback(() => {
        const finishSession = () => {
            resetLiveSession(null);
            setRestorationStatus(normalizedOwnerEmail ? 'not-found' : 'idle');
        };
        if (recordingActiveRef.current || recordingStopPromiseRef.current) {
            void stopRecordingSession('complete').then(finishSession, finishSession);
        } else {
            finishSession();
        }
    }, [normalizedOwnerEmail, resetLiveSession, stopRecordingSession]);

    useEffect(() => {
        let cancelled = false;
        persistDraftRef.current();
        const initializeOwner = () => {
            if (cancelled) return;
            ownerEmailRef.current = normalizedOwnerEmail;
            draftPersistenceSuppressedRef.current = false;
            resetLiveSession(null);

            if (!normalizedOwnerEmail) {
                setRestorationStatus('idle');
                return;
            }

            setRestorationStatus('restoring');
            const draft = getPersistedLiveSessionDraft(normalizedOwnerEmail);
            if (!draft) {
                setRestorationStatus('not-found');
                return;
            }

            const restoreDraft = async () => {
                let validation: LocalTelemetryFileValidation;
                try {
                    const summary = await runRecordedFileRead(
                        draft.telemetryFilePath,
                        draft.sessionGame,
                        'validate',
                    );
                    validation = {
                        exists: true,
                        readable: true,
                        hasData: summary.rowCount > 0,
                        size: summary.totalBytes,
                    };
                } catch (error) {
                    validation = {
                        exists: false,
                        readable: false,
                        hasData: false,
                        size: 0,
                        error: error instanceof Error ? error.message : String(error),
                    };
                }
                if (cancelled || ownerEmailRef.current !== normalizedOwnerEmail) return;

                resetLiveSession(draft.sessionGame);
                setRecordingMetadata(draft.recordingMetadata);
                setRecordingFileKey(draft.telemetryFilePath);
                liveTelemetryStore.restoreCommittedSampleCount(draft.recordedSampleCount);
                const restoredStaticData = {
                    Static_track: draft.recordingMetadata.mapName,
                    Static_car_model: draft.recordingMetadata.carName,
                };
                staticDataRef.current = restoredStaticData;
                setStaticDataState(restoredStaticData);
                recordingStateRef.current = RecordingState.UPLOAD_READY;
                setRecordingState(RecordingState.UPLOAD_READY);
                setRecordingFileValidation(validation);

                if (!validation.exists || !validation.readable) {
                    setRestorationStatus('error');
                    setRestorationError(RESTORED_RECORDING_ERROR);
                } else {
                    setRestorationStatus('restored');
                    setRestorationError(null);
                }
            };
            void restoreDraft();
        };

        if (recordingActiveRef.current || recordingStopPromiseRef.current) {
            void stopRecordingSession('complete').then(initializeOwner, initializeOwner);
        } else {
            initializeOwner();
        }
        return () => {
            cancelled = true;
        };
    }, [normalizedOwnerEmail, resetLiveSession, runRecordedFileRead, setRecordingFileKey, setRecordingMetadata, stopRecordingSession]);

    useEffect(() => {
        persistCurrentDraft();
    }, [
        persistCurrentDraft,
        recordingFileKey,
        recordingMetadata,
        recordingState,
        sessionGame,
    ]);

    useEffect(() => {
        return liveTelemetryStore.subscribeSelector(
            (snapshot) => snapshot.committedSampleCount,
            () => persistDraftRef.current(),
        );
    }, []);

    useEffect(() => {
        const persistBeforeUnload = () => persistDraftRef.current();
        window.addEventListener('beforeunload', persistBeforeUnload);
        return () => window.removeEventListener('beforeunload', persistBeforeUnload);
    }, []);

    useEffect(() => () => {
        persistDraftRef.current();
        if (recordingActiveRef.current) void stopRecordingSession('complete');
    }, [stopRecordingSession]);

    const value = useMemo<LiveSessionRuntime>(() => ({
        sessionGame,
        staticData,
        recordingState,
        recordingMetadata,
        recordingFileKey,
        recordingActive,
        recordingGame,
        restorationStatus,
        restorationError,
        recordingFileValidation,
        recorderControl,
        analysisResultPages,
        activeAnalysisResultPageId,
        getNextCorner,
        getLiveSessionSnapshot,
        startLiveSession,
        endLiveSession,
        setRecordingMetadata,
        transitionRecordingState,
        startRecordingSession,
        stopRecordingSession,
        streamRecordedTelemetry,
        clearRecordingSession,
        clearPersistedDraft,
        registerRecorderControl,
        appendAnalysisResultPage,
        selectAnalysisResultPage,
        updateActiveAnalysisResultPage,
    }), [
        clearRecordingSession,
        clearPersistedDraft,
        endLiveSession,
        getLiveSessionSnapshot,
        getNextCorner,
        restorationStatus,
        restorationError,
        recordingFileValidation,
        recorderControl,
        analysisResultPages,
        activeAnalysisResultPageId,
        recordingFileKey,
        recordingActive,
        recordingGame,
        recordingMetadata,
        recordingState,
        setRecordingMetadata,
        sessionGame,
        startLiveSession,
        startRecordingSession,
        staticData,
        stopRecordingSession,
        streamRecordedTelemetry,
        transitionRecordingState,
        registerRecorderControl,
        appendAnalysisResultPage,
        selectAnalysisResultPage,
        updateActiveAnalysisResultPage,
    ]);

    return <LiveSessionContext.Provider value={value}>{children}</LiveSessionContext.Provider>;
};
