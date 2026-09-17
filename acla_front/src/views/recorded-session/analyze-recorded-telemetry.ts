import apiService from 'services/api.service';
import {
    normalizeSegmentClassificationResult,
    SegmentClassificationResult,
} from 'views/session-shared/segment-classification';
import type { RecordedSessionTable } from './data/RecordedSessionDataProvider';

export const RECORDED_ANALYSIS_MAX_ROWS = 5000;
export const RECORDED_ANALYSIS_MAX_BYTES = 4 * 1024 * 1024;
const CONTEXT_ROWS = 256;
const REQUEST_TIMEOUT_MS = 120000;

const checkCancelled = (signal: AbortSignal) => {
    if (signal.aborted) throw new Error('Recorded analysis was cancelled.');
};

/** Size JSON as UTF-8 without allocating an encoded copy of the full request. */
const jsonBytes = (value: unknown): number => {
    const json = JSON.stringify(value);
    let bytes = 0;
    for (let index = 0; index < json.length; index += 1) {
        const code = json.charCodeAt(index);
        if (code < 0x80) bytes += 1;
        else if (code < 0x800) bytes += 2;
        else if (code >= 0xd800 && code <= 0xdbff) {
            bytes += 4;
            index += 1;
        } else bytes += 3;
    }
    return bytes;
};

/** Analyze every row with bounded, sequential requests; indices remain source indices. */
export async function analyzeRecordedTelemetry({
    sessionId, track, car, table, signal, onProgress,
}: {
    sessionId: string;
    track: string | null;
    car?: string;
    table: RecordedSessionTable;
    signal: AbortSignal;
    onProgress: (completed: number, total: number) => void;
}): Promise<SegmentClassificationResult> {
    checkCancelled(signal);
    if (!table.length) throw new Error('No telemetry samples are available for analysis.');
    const envelopeBytes = jsonBytes({ track, car, records: [] });
    const result: SegmentClassificationResult = {
        status: 'success', session_id: sessionId, samples_analyzed: 0,
        parent_segment_count: 0, segments: [],
    };
    let requestStart = 0;
    let ownedStart = 0;
    onProgress(0, table.length);

    while (ownedStart < table.length) {
        checkCancelled(signal);
        let requestEnd = requestStart;
        let bytes = envelopeBytes;
        while (requestEnd < table.length && requestEnd - requestStart < RECORDED_ANALYSIS_MAX_ROWS) {
            const rowBytes = jsonBytes(table[requestEnd]);
            if (envelopeBytes + rowBytes > RECORDED_ANALYSIS_MAX_BYTES) {
                throw new Error(`Telemetry sample ${requestEnd} exceeds the AI request size limit.`);
            }
            const nextBytes = bytes + rowBytes + (requestEnd > requestStart ? 1 : 0);
            if (nextBytes > RECORDED_ANALYSIS_MAX_BYTES) break;
            bytes = nextBytes;
            requestEnd += 1;
        }

        // Very wide rows can exhaust the budget with only previous context. Drop
        // that context, never a new source row, so every iteration makes progress.
        if (requestEnd <= ownedStart) {
            requestStart = ownedStart;
            continue;
        }
        const overlap = requestEnd === table.length ? 0 : Math.min(
            CONTEXT_ROWS, Math.floor((requestEnd - ownedStart) / 4),
        );
        const ownedEnd = requestEnd - overlap;
        const response = await apiService.post<SegmentClassificationResult>(
            '/racing-session/analyze-live-recorded-analysis',
            { track, car, records: table.slice(requestStart, requestEnd) },
            { timeout: REQUEST_TIMEOUT_MS, signal },
        );
        checkCancelled(signal);
        if (!Array.isArray(response.data?.segments)
            || (response.data.status && response.data.status !== 'success')) {
            throw new Error(`AI analysis failed for samples ${ownedStart}-${ownedEnd}.`);
        }
        const batch = normalizeSegmentClassificationResult(response.data, sessionId);
        for (const segment of batch.segments) {
            const start = Math.max(ownedStart, requestStart + segment.start_index);
            const end = Math.min(ownedEnd, requestStart + segment.end_index);
            if (end <= start) continue;
            const references = segment.expert_reference_data
                .filter((row) => row.raw_index + requestStart >= start && row.raw_index + requestStart < end)
                .map((row) => ({ ...row, raw_index: row.raw_index + requestStart }));
            const clipped = start !== requestStart + segment.start_index || end !== requestStart + segment.end_index;
            // A clipped range needs its own timing, not the whole context window's gap.
            const timeGap = clipped
                ? references.length ? {
                    start_ms: references[0].expert_time_difference,
                    end_ms: references[references.length - 1].expert_time_difference,
                    delta_ms: references[references.length - 1].expert_time_difference - references[0].expert_time_difference,
                } : undefined
                : segment.time_gap;
            result.segments.push({
                ...segment,
                id: `${segment.track_section || segment.id || 'segment'}:${start}-${end}`,
                start_index: start,
                end_index: end,
                labels: segment.labels.flatMap((label) => {
                    const labelStart = Math.max(start, requestStart + label.start_index);
                    const labelEnd = Math.min(end, requestStart + label.end_index);
                    return labelEnd > labelStart ? [{
                        ...label, start_index: labelStart, end_index: labelEnd,
                    }] : [];
                }),
                expert_reference_data: references,
                time_gap: timeGap,
            });
        }
        if (typeof batch.expert_time_available === 'boolean') {
            result.expert_time_available = (result.expert_time_available ?? true) && batch.expert_time_available;
        }
        result.samples_analyzed = ownedEnd;
        result.parent_segment_count = result.segments.length;
        onProgress(ownedEnd, table.length);
        ownedStart = ownedEnd;
        requestStart = ownedEnd - overlap;
    }
    return result;
}
