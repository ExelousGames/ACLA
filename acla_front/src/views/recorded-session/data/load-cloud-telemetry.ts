import apiService from 'services/api.service';
import type { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';

const RECORDED_TELEMETRY_TIMEOUT_MS = 120000;

/** Transport only: concatenate source chunks without interpreting their rows. */
export async function loadCloudTelemetry(
    session: RacingSessionDetailedInfoDto,
    map: string | null,
    signal: AbortSignal,
    onProgress: (message: string) => void,
): Promise<Record<string, unknown>[]> {
    const config = { timeout: RECORDED_TELEMETRY_TIMEOUT_MS, signal };
    const initResponse = await apiService.post<any>('/racing-session/download/init', {
        sessionId: session.SessionId,
    }, config);
    if (signal.aborted) throw new Error('Recorded telemetry loading was cancelled.');
    const initData = initResponse.data;
    const metadata = Array.isArray(initData?.sessionMetadata)
        ? initData.sessionMetadata.find((entry: any) => entry.sessionId === session.SessionId)
        : null;
    if (!metadata) {
        throw new Error('Selected session was not returned by the backend download initializer.');
    }

    const chunkCount = Math.max(1, Number(metadata.chunkCount) || 1);
    const rows: Record<string, unknown>[] = [];
    for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex += 1) {
        if (signal.aborted) throw new Error('Recorded telemetry loading was cancelled.');
        const response = await apiService.post<any>('/racing-session/download/chunk', {
            downloadId: initData.downloadId,
            sessionId: session.SessionId,
            trackName: map || metadata.map || session.map || '',
            carName: metadata.car_name || session.car || '',
            chunkIndex,
        }, config);
        if (signal.aborted) throw new Error('Recorded telemetry loading was cancelled.');
        const body = response.data;
        const chunkRows = Array.isArray(body) ? body : body?.data;
        if (!Array.isArray(chunkRows)) {
            throw new Error('Recorded telemetry download did not return a data table.');
        }
        // Do not sort, filter, coerce values, fill missing fields, or derive columns.
        for (const row of chunkRows) rows.push(row);
        onProgress(`Loading recorded telemetry ${chunkIndex + 1}/${chunkCount}...`);
    }
    return rows;
}
