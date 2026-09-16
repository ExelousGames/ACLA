import type { UploadReacingSessionInitDto, UploadRacingSessionInitReturnDto } from 'data/live-analysis/live-analysis-type';
import type { LiveSessionRuntime } from '../live-session-types';
import apiService from 'services/api.service';

type UploadOptions = {
    metadata: UploadReacingSessionInitDto;
    filePath: string;
    stream: LiveSessionRuntime['streamRecordedTelemetry'];
    onProgress: (progress: number, status: string) => void;
};

export async function uploadRecording({ metadata, filePath, stream, onProgress }: UploadOptions) {
    const progress = (percent: number, status: string) => onProgress(5 + Math.floor(percent * 0.9), status);
    progress(0, 'Initializing upload...');
    const initResp = await apiService.post('/racing-session/upload/init', {
        ...metadata,
        game_recorded_from: metadata.game_recorded_from === 'iracing' ? 'iracing_live' : metadata.game_recorded_from,
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
    }, filePath);
    if (summary.rowCount === 0) throw new Error('No telemetry data found to upload');
    progress(92, 'Finalizing upload...');
    const params = new URLSearchParams({ uploadId });
    await apiService.post(`/racing-session/upload/complete?${params.toString()}`, {});
}
