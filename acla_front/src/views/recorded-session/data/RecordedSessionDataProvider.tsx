import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react';
import type { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { loadCloudTelemetry } from './load-cloud-telemetry';

export type RecordedSessionTable = readonly Readonly<Record<string, unknown>>[];

export interface RecordedSessionData {
    readonly sessionId: string | null;
    readonly source: 'cloud' | 'local-ibt' | null;
    readonly status: 'idle' | 'loading' | 'ready' | 'error';
    readonly table: RecordedSessionTable;
    readonly message?: string;
}

const EMPTY_TABLE: RecordedSessionTable = Object.freeze([]);
const IDLE_DATA: RecordedSessionData = { sessionId: null, source: null, status: 'idle', table: EMPTY_TABLE };
const RecordedSessionDataContext = createContext<RecordedSessionData>(IDLE_DATA);

/** The table is source data. Consumers derive display/analysis data separately. */
export const useRecordedSessionData = (): RecordedSessionData => useContext(RecordedSessionDataContext);

export function RecordedSessionDataProvider({ session, map, children }: {
    session: RacingSessionDetailedInfoDto | null;
    map: string | null;
    children: ReactNode;
}) {
    const input = useMemo<RecordedSessionData>(() => {
        if (!session) return IDLE_DATA;
        if (session.storage === 'local') {
            return { sessionId: session.SessionId, source: 'local-ibt', status: 'ready', table: session.data };
        }
        return {
            sessionId: session.SessionId,
            source: 'cloud',
            status: 'loading',
            table: EMPTY_TABLE,
            message: 'Loading recorded telemetry from backend...',
        };
    }, [session]);
    const [download, setDownload] = useState<{
        input: RecordedSessionData;
        map: string | null;
        data: RecordedSessionData;
    } | null>(null);

    useEffect(() => {
        if (!session || input.source !== 'cloud') {
            setDownload(null);
            return;
        }
        const controller = new AbortController();
        const publish = (data: RecordedSessionData) => {
            if (!controller.signal.aborted) setDownload({ input, map, data });
        };
        publish(input);
        void loadCloudTelemetry(session, map, controller.signal, (message) => {
            publish({ ...input, message });
        }).then((table) => {
            publish({ ...input, status: 'ready', table, message: undefined });
        }).catch((error) => {
            publish({ ...input, status: 'error', message: error?.message || 'Failed to load recorded telemetry.' });
        });
        return () => controller.abort();
    }, [input, map, session]);

    // Never expose a previous session's table, including the render before effect cleanup.
    const data = download?.input === input && download.map === map ? download.data : input;
    return <RecordedSessionDataContext.Provider value={data}>{children}</RecordedSessionDataContext.Provider>;
}
