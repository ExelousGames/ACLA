import React, { createContext, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { PythonShellOptions } from 'services/pythonService';
import { createPythonStreamSession, PythonStreamEvent, PythonStreamSession } from 'services/pythonStreaming';
import { getNextRecordingState, RecordingEvent, RecordingState } from 'views/lap-analysis/recording-state';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import {
    LocalTelemetryFileValidation,
    LiveRecordingMetadata,
    LiveSessionRecorderControl,
    LiveSessionRestorationStatus,
    LiveSessionRuntime,
    LiveSessionStaticData,
    LiveTelemetry,
    PERSISTED_LIVE_SESSION_DRAFT_VERSION,
} from './live-session-types';
import {
    getPersistedLiveSessionDraft,
    normalizeLiveSessionOwnerEmail,
    removePersistedLiveSessionDraft,
    savePersistedLiveSessionDraft,
} from './live-session-draft-storage';
import type {
    LiveRangeTodoListHandle,
    LiveRangeTodoListSnapshot,
} from './live-range-todo-list-types';

const TELEMETRY_WRITE_TIMEOUT_MS = 6000;
const LIVE_TELEMETRY_UI_UPDATE_MS = 100;
const LIVE_SAMPLE_COUNT_UI_UPDATE_MS = 250;

const RESTORED_RECORDING_ERROR = 'The local recording file is missing or unreadable. Upload is unavailable; discard this draft to clear it.';

const isAbsoluteFilePath = (filePath: string): boolean =>
    /^(?:[a-zA-Z]:[\\/]|\\\\|\/)/.test(filePath);

type TelemetryWriterEvent = {
    status?: string;
    request_id?: string;
    message?: string;
    written?: number;
    [key: string]: unknown;
};

type PendingTelemetryWrite = {
    resolve: () => void;
    reject: (error: Error) => void;
    timeoutId: number;
};

const normalizeAccStatus = (value: unknown): ACC_STATUS | null => {
    const numeric = typeof value === 'string' ? Number(value) : value;
    if (typeof numeric !== 'number' || Number.isNaN(numeric)) return null;
    return ACC_STATUS[numeric as ACC_STATUS] !== undefined ? numeric as ACC_STATUS : null;
};

const missingProvider = () => console.warn('No provider for LiveSessionContext');

const defaultRuntime: LiveSessionRuntime = {
    sessionGame: null,
    currentTelemetry: {},
    telemetryStatus: null,
    staticData: {},
    recordingState: RecordingState.CHECKING,
    recordingMetadata: null,
    recordingFileKey: null,
    recordedSampleCount: 0,
    restorationStatus: 'idle',
    restorationError: null,
    recordingFileValidation: null,
    sessionIntelligence: new SessionIntelligence(),
    liveRangeTodoListHandle: null,
    liveRangeTodoListSnapshot: null,
    recorderControl: null,
    startLiveSession: missingProvider,
    endLiveSession: missingProvider,
    setCurrentTelemetry: missingProvider,
    setStaticData: missingProvider,
    setRecordingMetadata: missingProvider,
    transitionRecordingState: missingProvider,
    appendTelemetrySample: async () => missingProvider(),
    readRecordedTelemetry: async () => {
        missingProvider();
        return [];
    },
    finalizeRecordingWrites: async () => missingProvider(),
    clearRecordingSession: missingProvider,
    clearPersistedDraft: missingProvider,
    registerLiveRangeTodoListHandle: missingProvider,
    publishLiveRangeTodoListSnapshot: missingProvider,
    registerRecorderControl: missingProvider,
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
    const [currentTelemetry, setCommittedTelemetry] = useState<LiveTelemetry>({});
    const [telemetryStatus, setTelemetryStatus] = useState<ACC_STATUS | null>(null);
    const [staticData, setStaticDataState] = useState<LiveSessionStaticData>({});
    const [recordingState, setRecordingState] = useState(RecordingState.CHECKING);
    const [recordingMetadata, setRecordingMetadataState] = useState<LiveRecordingMetadata | null>(null);
    const [recordingFileKey, setRecordingFileKeyState] = useState<string | null>(null);
    const [recordedSampleCount, setRecordedSampleCount] = useState(0);
    const [restorationStatus, setRestorationStatus] = useState<LiveSessionRestorationStatus>(
        normalizedOwnerEmail ? 'restoring' : 'idle',
    );
    const [restorationError, setRestorationError] = useState<string | null>(null);
    const [recordingFileValidation, setRecordingFileValidation] = useState<LocalTelemetryFileValidation | null>(null);
    const [liveRangeTodoListHandle, setLiveRangeTodoListHandle] = useState<LiveRangeTodoListHandle | null>(null);
    const [liveRangeTodoListSnapshot, setLiveRangeTodoListSnapshot] = useState<LiveRangeTodoListSnapshot | null>(null);
    const [recorderControl, setRecorderControl] = useState<LiveSessionRecorderControl | null>(null);

    const sessionIntelligenceRef = useRef(new SessionIntelligence());
    const sessionGameRef = useRef<DesktopGame | null>(null);
    const ownerEmailRef = useRef(normalizedOwnerEmail);
    const recordingStateRef = useRef(RecordingState.CHECKING);
    const recordingMetadataRef = useRef<LiveRecordingMetadata | null>(null);
    const sessionGenerationRef = useRef(0);
    const latestTelemetryRef = useRef<LiveTelemetry>({});
    const committedTelemetryRef = useRef<LiveTelemetry>({});
    const telemetryFlushTimeoutRef = useRef<number | null>(null);
    const recordingFileKeyRef = useRef<string | null>(null);
    const writeQueueRef = useRef<Promise<void>>(Promise.resolve());
    const telemetryWriterSessionRef = useRef<PythonStreamSession<TelemetryWriterEvent> | null>(null);
    const telemetryWriterCleanupRef = useRef<(() => void) | null>(null);
    const telemetryWriterFileKeyRef = useRef<string | null>(null);
    const telemetryWriterPendingRef = useRef<Map<string, PendingTelemetryWrite>>(new Map());
    const telemetryWriterSequenceRef = useRef(0);
    const sampleCountRef = useRef(0);
    const committedSampleCountRef = useRef(0);
    const sampleCountFlushTimeoutRef = useRef<number | null>(null);
    const persistDraftRef = useRef<() => void>(() => undefined);
    const draftPersistenceSuppressedRef = useRef(false);
    const registerLiveRangeTodoListHandle = useCallback((handle: LiveRangeTodoListHandle | null) => {
        setLiveRangeTodoListHandle(handle);
    }, []);

    const publishLiveRangeTodoListSnapshot = useCallback((snapshot: LiveRangeTodoListSnapshot | null) => {
        setLiveRangeTodoListSnapshot(snapshot);
    }, []);

    const registerRecorderControl = useCallback((control: LiveSessionRecorderControl | null) => {
        setRecorderControl(control);
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
            recordedSampleCount: sampleCountRef.current,
            lastRuntimeState: currentState,
            updatedAt: new Date().toISOString(),
        });
    }, []);
    persistDraftRef.current = persistCurrentDraft;

    const clearPersistedDraft = useCallback(() => {
        draftPersistenceSuppressedRef.current = true;
        const currentOwnerEmail = ownerEmailRef.current;
        if (currentOwnerEmail) removePersistedLiveSessionDraft(currentOwnerEmail);
    }, []);

    const flushCurrentTelemetry = useCallback(() => {
        if (telemetryFlushTimeoutRef.current !== null) {
            window.clearTimeout(telemetryFlushTimeoutRef.current);
            telemetryFlushTimeoutRef.current = null;
        }

        const nextTelemetry = latestTelemetryRef.current && typeof latestTelemetryRef.current === 'object'
            ? latestTelemetryRef.current
            : {};
        if (committedTelemetryRef.current === nextTelemetry) return;
        if (Object.keys(nextTelemetry).length === 0 && Object.keys(committedTelemetryRef.current).length === 0) return;

        committedTelemetryRef.current = nextTelemetry;
        setCommittedTelemetry(nextTelemetry);
    }, []);

    const setCurrentTelemetry = useCallback((data: LiveTelemetry) => {
        const nextTelemetry = data && typeof data === 'object' ? data : {};
        latestTelemetryRef.current = nextTelemetry;

        if (telemetryFlushTimeoutRef.current === null) {
            telemetryFlushTimeoutRef.current = window.setTimeout(flushCurrentTelemetry, LIVE_TELEMETRY_UI_UPDATE_MS);
        }
        if (Object.keys(nextTelemetry).length > 0) {
            sessionIntelligenceRef.current.tick(nextTelemetry);
        }
    }, [flushCurrentTelemetry]);

    const setStaticData = useCallback((data: LiveSessionStaticData) => {
        setStaticDataState(data && typeof data === 'object' ? data : {});
    }, []);

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

    const flushSampleCount = useCallback(() => {
        if (sampleCountFlushTimeoutRef.current !== null) {
            window.clearTimeout(sampleCountFlushTimeoutRef.current);
            sampleCountFlushTimeoutRef.current = null;
        }
        if (committedSampleCountRef.current === sampleCountRef.current) return;
        committedSampleCountRef.current = sampleCountRef.current;
        setRecordedSampleCount(sampleCountRef.current);
    }, []);

    const incrementSampleCount = useCallback(() => {
        sampleCountRef.current += 1;
        if (sampleCountFlushTimeoutRef.current === null) {
            sampleCountFlushTimeoutRef.current = window.setTimeout(flushSampleCount, LIVE_SAMPLE_COUNT_UI_UPDATE_MS);
        }
    }, [flushSampleCount]);

    const resetSampleCount = useCallback(() => {
        if (sampleCountFlushTimeoutRef.current !== null) {
            window.clearTimeout(sampleCountFlushTimeoutRef.current);
            sampleCountFlushTimeoutRef.current = null;
        }
        sampleCountRef.current = 0;
        committedSampleCountRef.current = 0;
        setRecordedSampleCount(0);
    }, []);

    const disposeTelemetryWriter = useCallback(async ({ force = false }: { force?: boolean } = {}) => {
        telemetryWriterCleanupRef.current?.();
        telemetryWriterCleanupRef.current = null;
        const session = telemetryWriterSessionRef.current;
        telemetryWriterSessionRef.current = null;
        telemetryWriterFileKeyRef.current = null;

        for (const [requestId, pending] of Array.from(telemetryWriterPendingRef.current.entries())) {
            telemetryWriterPendingRef.current.delete(requestId);
            window.clearTimeout(pending.timeoutId);
            pending.reject(new Error('Telemetry writer disposed'));
        }

        if (session) {
            try {
                await session.dispose({ force });
            } catch (error) {
                console.warn('Failed to dispose telemetry writer session', error);
            }
        }
    }, []);

    const handleTelemetryWriterEvent = useCallback((event: PythonStreamEvent<TelemetryWriterEvent>) => {
        if (!event) return;
        const requestId = typeof event.request_id === 'string' ? event.request_id : undefined;
        const pending = requestId ? telemetryWriterPendingRef.current.get(requestId) : undefined;

        if (event.status === 'ok' && requestId && pending) {
            telemetryWriterPendingRef.current.delete(requestId);
            pending.resolve();
            return;
        }
        if (event.status === 'error') {
            const error = new Error(typeof event.message === 'string' ? event.message : 'Telemetry writer error');
            if (requestId && pending) {
                telemetryWriterPendingRef.current.delete(requestId);
                pending.reject(error);
            } else {
                for (const [pendingId, item] of Array.from(telemetryWriterPendingRef.current.entries())) {
                    telemetryWriterPendingRef.current.delete(pendingId);
                    item.reject(error);
                }
            }
            return;
        }
        if (event.status === 'shutdown') {
            if (requestId && pending) {
                telemetryWriterPendingRef.current.delete(requestId);
                pending.resolve();
            }
            void disposeTelemetryWriter({ force: true });
        }
    }, [disposeTelemetryWriter]);

    const ensureTelemetryWriter = useCallback(async (fileKey: string, sessionGeneration: number) => {
        if (telemetryWriterSessionRef.current && telemetryWriterFileKeyRef.current === fileKey) {
            await telemetryWriterSessionRef.current.waitUntilReady();
            return telemetryWriterSessionRef.current;
        }

        await disposeTelemetryWriter({ force: true });
        try {
            const session = await createPythonStreamSession<TelemetryWriterEvent>({
                scriptName: 'append_telemetry_data.py',
                pythonOptions: {
                    mode: 'text',
                    pythonOptions: ['-u'],
                    scriptPath: 'src/py-scripts',
                    args: [fileKey],
                },
                readyTimeoutMs: 8000,
            });
            if (
                sessionGeneration !== sessionGenerationRef.current
                || sessionGameRef.current === null
            ) {
                await session.dispose({ force: true });
                throw new Error('Live session ended');
            }
            telemetryWriterSessionRef.current = session;
            telemetryWriterFileKeyRef.current = fileKey;
            telemetryWriterCleanupRef.current = session.onMessage(handleTelemetryWriterEvent);
            await session.waitUntilReady();
            return session;
        } catch (error) {
            await disposeTelemetryWriter({ force: true });
            throw error;
        }
    }, [disposeTelemetryWriter, handleTelemetryWriterEvent]);

    const finalizeRecordingWrites = useCallback(async () => {
        try {
            await writeQueueRef.current;
        } catch (error) {
            console.warn('Telemetry write queue rejected during finalization', error);
        } finally {
            writeQueueRef.current = Promise.resolve();
        }
        flushSampleCount();
        await disposeTelemetryWriter({ force: false });
    }, [disposeTelemetryWriter, flushSampleCount]);

    const appendTelemetrySample = useCallback(async (data: LiveTelemetry) => {
        const sessionGeneration = sessionGenerationRef.current;
        const enqueueWrite = async () => {
            if (
                sessionGeneration !== sessionGenerationRef.current
                || sessionGameRef.current === null
            ) {
                return;
            }
            let fileKey = recordingFileKeyRef.current;
            if (!fileKey) {
                const created = await window.electronAPI.writeTempFile({
                    content: '',
                    prefix: 'telemetry_live',
                    extension: '.jsonl',
                });
                if (!created.success || !created.path || !isAbsoluteFilePath(created.path)) {
                    throw new Error(created.error || 'Unable to create persistent telemetry file');
                }
                fileKey = created.path;
                setRecordingFileKey(fileKey);
                resetSampleCount();
            }

            const session = await ensureTelemetryWriter(fileKey, sessionGeneration);
            if (
                sessionGeneration !== sessionGenerationRef.current
                || sessionGameRef.current === null
            ) {
                return;
            }
            const requestId = `telemetry-append-${Date.now()}-${++telemetryWriterSequenceRef.current}`;
            let resolveAck!: () => void;
            let rejectAck!: (error: Error) => void;
            const ackPromise = new Promise<void>((resolve, reject) => {
                resolveAck = resolve;
                rejectAck = reject;
            });
            const timeoutId = window.setTimeout(() => {
                const pending = telemetryWriterPendingRef.current.get(requestId);
                if (pending) {
                    telemetryWriterPendingRef.current.delete(requestId);
                    pending.reject(new Error('Telemetry writer append timed out'));
                }
            }, TELEMETRY_WRITE_TIMEOUT_MS);

            telemetryWriterPendingRef.current.set(requestId, {
                resolve: () => {
                    window.clearTimeout(timeoutId);
                    resolveAck();
                },
                reject: (error) => {
                    window.clearTimeout(timeoutId);
                    rejectAck(error);
                },
                timeoutId,
            });

            try {
                await session.send('append', { data }, requestId);
                await ackPromise;
                incrementSampleCount();
            } catch (error) {
                const pending = telemetryWriterPendingRef.current.get(requestId);
                if (pending) {
                    telemetryWriterPendingRef.current.delete(requestId);
                    pending.reject(error instanceof Error ? error : new Error(String(error)));
                }
                throw error;
            }
        };

        const nextWrite = writeQueueRef.current.then(enqueueWrite);
        writeQueueRef.current = nextWrite.catch((error) => {
            if (!(error instanceof Error && (
                error.message === 'Telemetry writer disposed'
                || error.message === 'Live session ended'
            ))) {
                console.error('Telemetry write failed', error);
            }
        });
        return nextWrite.catch((error) => {
            if (error instanceof Error && (
                error.message === 'Telemetry writer disposed'
                || error.message === 'Live session ended'
            )) return;
            throw error;
        });
    }, [ensureTelemetryWriter, incrementSampleCount, resetSampleCount, setRecordingFileKey]);

    const readRecordedTelemetry = useCallback(async (
        onProgress?: (read: number, total: number | null, bytesRead?: number, totalBytes?: number) => void,
    ): Promise<LiveTelemetry[]> => {
        const fileKey = recordingFileKeyRef.current;
        if (!fileKey) return [];

        try {
            const options: PythonShellOptions = {
                mode: 'text',
                pythonOptions: ['-u'],
                scriptPath: 'src/py-scripts',
                args: [fileKey],
            };
            const { shellId } = await window.electronAPI.runPythonScript('read_telemetry_data.py', options);
            return new Promise((resolve) => {
                let completeReceived = false;
                const allData: LiveTelemetry[] = [];
                let removeMessageListener: (() => void) | null = null;
                let removeEndListener: (() => void) | null = null;
                const cleanup = () => {
                    removeMessageListener?.();
                    removeEndListener?.();
                    removeMessageListener = null;
                    removeEndListener = null;
                };

                removeMessageListener = window.electronAPI.onPythonMessage((returnedShellId: number, message: string) => {
                    if (returnedShellId !== shellId) return;
                    try {
                        const parsed = JSON.parse(message);
                        if (parsed.type === 'progress') {
                            onProgress?.(parsed.read, parsed.total ?? null, parsed.bytesRead, parsed.totalBytes);
                        } else if (parsed.type === 'chunk' && Array.isArray(parsed.data)) {
                            allData.push(...parsed.data);
                        } else if (parsed.type === 'complete') {
                            completeReceived = true;
                            if (Array.isArray(parsed.data)) allData.push(...parsed.data);
                            resolve(allData);
                            cleanup();
                        }
                    } catch {
                        // Ignore non-JSON process output.
                    }
                });
                removeEndListener = window.electronAPI.onPythonEnd('live-session-recording-reader', (returnedShellId: number) => {
                    if (returnedShellId !== shellId || completeReceived) return;
                    resolve(allData);
                    cleanup();
                });
            });
        } catch (error) {
            console.error('Error reading live recording data', error);
            return [];
        }
    }, []);

    const clearRecordingSession = useCallback(() => {
        setRecordingFileKey(null);
        setRecordingMetadata(null);
        resetSampleCount();
        writeQueueRef.current = Promise.resolve();
        sessionIntelligenceRef.current.reset();
        void disposeTelemetryWriter({ force: true });
    }, [disposeTelemetryWriter, resetSampleCount, setRecordingFileKey, setRecordingMetadata]);

    const resetLiveSession = useCallback((nextGame: DesktopGame | null) => {
        sessionGenerationRef.current += 1;
        sessionGameRef.current = nextGame;
        setSessionGame(nextGame);

        if (telemetryFlushTimeoutRef.current !== null) {
            window.clearTimeout(telemetryFlushTimeoutRef.current);
            telemetryFlushTimeoutRef.current = null;
        }
        latestTelemetryRef.current = {};
        committedTelemetryRef.current = {};
        setCommittedTelemetry({});
        setTelemetryStatus(null);
        setStaticDataState({});
        recordingStateRef.current = RecordingState.CHECKING;
        setRecordingState(RecordingState.CHECKING);
        setRecordingMetadata(null);
        setRecordingFileKey(null);
        resetSampleCount();
        writeQueueRef.current = Promise.resolve();
        sessionIntelligenceRef.current.reset();
        setLiveRangeTodoListSnapshot(null);
        setRestorationError(null);
        void disposeTelemetryWriter({ force: true });
    }, [disposeTelemetryWriter, resetSampleCount, setRecordingFileKey, setRecordingMetadata]);

    const startLiveSession = useCallback((game: DesktopGame) => {
        if (sessionGameRef.current !== null) return;
        draftPersistenceSuppressedRef.current = false;
        resetLiveSession(game);
        setRestorationStatus('not-found');
    }, [resetLiveSession]);

    const endLiveSession = useCallback(() => {
        resetLiveSession(null);
        setRestorationStatus(normalizedOwnerEmail ? 'not-found' : 'idle');
    }, [normalizedOwnerEmail, resetLiveSession]);

    useEffect(() => {
        let cancelled = false;
        persistDraftRef.current();
        ownerEmailRef.current = normalizedOwnerEmail;
        draftPersistenceSuppressedRef.current = false;
        resetLiveSession(null);

        if (!normalizedOwnerEmail) {
            setRestorationStatus('idle');
            return () => {
                cancelled = true;
            };
        }

        setRestorationStatus('restoring');
        const draft = getPersistedLiveSessionDraft(normalizedOwnerEmail);
        if (!draft) {
            setRestorationStatus('not-found');
            return () => {
                cancelled = true;
            };
        }

        const restoreDraft = async () => {
            let validation: LocalTelemetryFileValidation;
            try {
                if (!window.electronAPI?.validateTelemetryFile) {
                    throw new Error('Local telemetry validation is unavailable');
                }
                validation = await window.electronAPI.validateTelemetryFile(draft.telemetryFilePath);
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
            sampleCountRef.current = draft.recordedSampleCount;
            committedSampleCountRef.current = draft.recordedSampleCount;
            setRecordedSampleCount(draft.recordedSampleCount);
            setStaticDataState({
                track: draft.recordingMetadata.mapName,
                car_model: draft.recordingMetadata.carName,
            });
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
        return () => {
            cancelled = true;
        };
    }, [normalizedOwnerEmail, resetLiveSession, setRecordingFileKey, setRecordingMetadata]);

    useEffect(() => {
        persistCurrentDraft();
    }, [
        persistCurrentDraft,
        recordedSampleCount,
        recordingFileKey,
        recordingMetadata,
        recordingState,
        sessionGame,
    ]);

    useEffect(() => {
        const nextStatus = normalizeAccStatus(
            currentTelemetry?.Graphics_status ?? currentTelemetry?.Graphics?.status,
        );
        if (Object.keys(currentTelemetry).length === 0) {
            setTelemetryStatus(null);
        } else if (nextStatus !== null) {
            setTelemetryStatus(nextStatus);
        }
    }, [currentTelemetry]);

    useEffect(() => {
        const persistBeforeUnload = () => persistDraftRef.current();
        window.addEventListener('beforeunload', persistBeforeUnload);
        return () => window.removeEventListener('beforeunload', persistBeforeUnload);
    }, []);

    useEffect(() => () => {
        persistDraftRef.current();
        if (telemetryFlushTimeoutRef.current !== null) window.clearTimeout(telemetryFlushTimeoutRef.current);
        if (sampleCountFlushTimeoutRef.current !== null) window.clearTimeout(sampleCountFlushTimeoutRef.current);
        void disposeTelemetryWriter({ force: true });
    }, [disposeTelemetryWriter]);

    const value = useMemo<LiveSessionRuntime>(() => ({
        sessionGame,
        currentTelemetry,
        telemetryStatus,
        staticData,
        recordingState,
        recordingMetadata,
        recordingFileKey,
        recordedSampleCount,
        restorationStatus,
        restorationError,
        recordingFileValidation,
        sessionIntelligence: sessionIntelligenceRef.current,
        liveRangeTodoListHandle,
        liveRangeTodoListSnapshot,
        recorderControl,
        startLiveSession,
        endLiveSession,
        setCurrentTelemetry,
        setStaticData,
        setRecordingMetadata,
        transitionRecordingState,
        appendTelemetrySample,
        readRecordedTelemetry,
        finalizeRecordingWrites,
        clearRecordingSession,
        clearPersistedDraft,
        registerLiveRangeTodoListHandle,
        publishLiveRangeTodoListSnapshot,
        registerRecorderControl,
    }), [
        appendTelemetrySample,
        clearRecordingSession,
        clearPersistedDraft,
        currentTelemetry,
        endLiveSession,
        finalizeRecordingWrites,
        readRecordedTelemetry,
        recordedSampleCount,
        restorationStatus,
        restorationError,
        recordingFileValidation,
        liveRangeTodoListHandle,
        liveRangeTodoListSnapshot,
        recorderControl,
        recordingFileKey,
        recordingMetadata,
        recordingState,
        setRecordingMetadata,
        setCurrentTelemetry,
        setStaticData,
        sessionGame,
        startLiveSession,
        staticData,
        telemetryStatus,
        transitionRecordingState,
        registerLiveRangeTodoListHandle,
        publishLiveRangeTodoListSnapshot,
        registerRecorderControl,
    ]);

    return <LiveSessionContext.Provider value={value}>{children}</LiveSessionContext.Provider>;
};
