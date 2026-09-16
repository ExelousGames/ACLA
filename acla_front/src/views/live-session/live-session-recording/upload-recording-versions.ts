import type { UploadReacingSessionInitDto, UploadRacingSessionInitReturnDto, RecordedSessionSource } from 'data/live-analysis/live-analysis-type';
import type { LiveSessionRuntime } from '../live-session-types';
import apiService from 'services/api.service';

type UploadOptions = {
    metadata: UploadReacingSessionInitDto;
    filePath: string;
    sampleCount: number;
    stream: LiveSessionRuntime['streamRecordedTelemetry'];
    completedVersions: Set<string>;
    onProgress: (progress: number, status: string) => void;
};

// Inspection and upload may request the same conversion at the same time.
const pendingIRacingPreparations = new Map<string, Promise<{ filePath: string }>>();

export function prepareIRacingRecording(filePath: string): Promise<{ filePath: string }> {
    const pending = pendingIRacingPreparations.get(filePath);
    if (pending) return pending;
    const preparation = Promise.resolve().then(() => {
        if (!window.electronAPI?.prepareIRacingRecordedTelemetry) throw new Error('iRacing .ibt import is unavailable. Restart the updated desktop app.');
        return window.electronAPI.prepareIRacingRecordedTelemetry(filePath);
    }).finally(() => { pendingIRacingPreparations.delete(filePath); });
    pendingIRacingPreparations.set(filePath, preparation);
    return preparation;
}

export async function waitForIRacingPreparation(filePath: string): Promise<void> {
    await pendingIRacingPreparations.get(filePath)?.catch(() => undefined);
}

export async function uploadRecordingVersions({ metadata, filePath, sampleCount, stream, completedVersions, onProgress }: UploadOptions) {
    const versions: { source: RecordedSessionSource; filePath?: string }[] = metadata.game_recorded_from === 'iracing'
        ? [{ source: 'iracing_live' }, { source: 'iracing_recorded' }]
        : [{ source: metadata.game_recorded_from }];
    const convertedFiles: string[] = [];
    if (versions.length === 2) {
        onProgress(2, 'Finding and converting iRacing .ibt telemetry...');
        const converted = await prepareIRacingRecording(filePath);
        versions[1].filePath = converted.filePath;
        convertedFiles.push(converted.filePath);
    }
    for (let versionIndex = 0; versionIndex < versions.length; versionIndex += 1) {
        const version = versions[versionIndex];
        const checkpoint = JSON.stringify([metadata.userId, filePath, sampleCount, version.source]);
        if (completedVersions.has(checkpoint)) continue;
        const label = versions.length === 2 ? `${version.source}: ` : '';
        const progress = (percent: number, status: string) => onProgress(
            5 + Math.floor((versionIndex + percent / 100) / versions.length * 90), `${label}${status}`,
        );
        progress(0, 'Initializing upload...');
        const initResp = await apiService.post('/racing-session/upload/init', {
            ...metadata,
            sessionName: versions.length === 2 ? `${metadata.sessionName} (${version.source})` : metadata.sessionName,
            game_recorded_from: version.source,
        });
        const { uploadId } = (initResp.data || {}) as UploadRacingSessionInitReturnDto;
        if (!uploadId) throw new Error('Failed to initialize upload');
        let chunkIndex = 0;
        let uploadedRows = 0;
        const summary = await stream(async (rows) => {
            const index = chunkIndex++;
            const params = new URLSearchParams({ uploadId });
            for (let attempt = 0; attempt < 3; attempt += 1) {
                try {
                    await apiService.post(`/racing-session/upload/chunk?${params.toString()}`, { chunk: rows, chunkIndex: index });
                    break;
                } catch (error) {
                    if (attempt === 2) throw error;
                    await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 2)));
                }
            }
            uploadedRows += rows.length;
        }, (_rowsRead, _totalRows, bytesRead, totalBytes) => {
            progress(totalBytes > 0 ? Math.min(85, bytesRead / totalBytes * 85) : 5,
                `Uploaded ${uploadedRows.toLocaleString()} telemetry points...`);
        }, version.filePath);
        if (summary.rowCount === 0) throw new Error(`${label}No telemetry data found to upload`);
        progress(92, 'Finalizing upload...');
        const params = new URLSearchParams({ uploadId });
        await apiService.post(`/racing-session/upload/complete?${params.toString()}`, {});
        completedVersions.add(checkpoint);
    }
    return convertedFiles;
}
