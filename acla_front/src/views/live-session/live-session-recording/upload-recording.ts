import type { UploadReacingSessionInitDto, UploadRacingSessionInitReturnDto } from 'data/live-analysis/live-analysis-type';
import type { LiveSessionRuntime } from '../live-session-types';
import apiService from 'services/api.service';
import type { SessionVideoClip } from './session-video';

type UploadOptions = {
    metadata: UploadReacingSessionInitDto;
    filePath: string;
    stream: LiveSessionRuntime['streamRecordedTelemetry'];
    onProgress: (progress: number, status: string) => void;
    videos?: SessionVideoClip[];
};

export async function uploadRecording({ metadata, filePath, stream, onProgress, videos = [] }: UploadOptions) {
    if (videos.length > 1) throw new Error('Only one video is allowed per session.');
    const progress = (percent: number, status: string) => onProgress(5 + Math.floor(percent * 0.9), status);
    progress(0, 'Initializing upload...');
    const initResp = await apiService.post('/racing-session/upload/init', {
        ...metadata,
        game_recorded_from: metadata.game_recorded_from === 'iracing' ? 'iracing_live' : metadata.game_recorded_from,
        ...(videos.length ? { videos: videos.map(({ name, size, mimeType, startedAt }) => ({ name, size, mimeType, startedAt })) } : {}),
    });
    const { uploadId } = (initResp.data || {}) as UploadRacingSessionInitReturnDto;
    if (!uploadId) throw new Error('Failed to initialize upload');
    try {
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
            progress(totalBytes > 0 ? Math.min(videos.length ? 45 : 85, bytesRead / totalBytes * (videos.length ? 45 : 85)) : 5,
                `Uploaded ${uploadedRows.toLocaleString()} telemetry points...`);
        }, filePath);
        if (summary.rowCount === 0) throw new Error('No telemetry data found to upload');
        const totalVideoBytes = videos.reduce((total, video) => total + video.size, 0);
        let uploadedVideoBytes = 0;
        for (let videoIndex = 0; videoIndex < videos.length; videoIndex += 1) {
            const video = videos[videoIndex];
            let offset = 0;
            let videoChunkIndex = 0;
            while (offset < video.size) {
                if (!window.sessionVideo) throw new Error('Local video files are unavailable.');
                const { data, bytesRead } = await window.sessionVideo.read(video.filePath, offset);
                if (!bytesRead || bytesRead > video.size - offset) throw new Error('The local video changed during upload.');
                const params = new URLSearchParams({ uploadId });
                for (let attempt = 0; attempt < 3; attempt += 1) {
                    try {
                        await apiService.post(`/racing-session/upload/video-chunk?${params}`, {
                            videoIndex, chunkIndex: videoChunkIndex, data,
                        }, { timeout: 60000 });
                        break;
                    } catch (error) {
                        if (attempt === 2) throw error;
                        await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
                    }
                }
                offset += bytesRead;
                uploadedVideoBytes += bytesRead;
                videoChunkIndex += 1;
                progress(45 + 45 * uploadedVideoBytes / totalVideoBytes, 'Uploading video...');
            }
        }
        progress(92, 'Finalizing upload...');
        const params = new URLSearchParams({ uploadId });
        await apiService.post(`/racing-session/upload/complete?${params.toString()}`, {});
    } catch (error) {
        if (videos.length) {
            await apiService.delete(`/racing-session/upload/${encodeURIComponent(uploadId)}`).catch(() => undefined);
        }
        throw error;
    }
}
