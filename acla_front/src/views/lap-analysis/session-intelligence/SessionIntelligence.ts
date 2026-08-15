import { SessionEvent, TelemetrySample, TelemetryQuery, QueryResult, QueryScope, CornerLookahead, ReduceOp } from './types';
import { TelemetryBuffer } from './TelemetryBuffer';
import { EventLog, EventSearchParams } from './EventLog';
import { SensorManager } from './SensorManager';
import { executeQuery, resolveScope } from './telemetry-query';
import { getCornersForTrack, getNextCorner } from './track-corners';
import {
    detectLiveSessionType,
    getTelemetryCar,
    getTelemetryLap,
    getTelemetryPosition,
    getTelemetryTrack,
    LiveSessionType,
} from './live-performance-analyst';

type IndexedTelemetrySample = {
    index: number;
    sample: TelemetrySample;
};

export type NormalizedRangeTelemetryWindow = {
    status: 'ready' | 'empty';
    startPosition: number;
    endPosition: number;
    lap: number;
    startSampleIdx?: number;
    endSampleIdx?: number;
    rows: TelemetrySample[];
};

type LiveSessionSnapshot = {
    status: 'ready' | 'empty';
    track: string;
    car: string;
    current_lap: number;
    completed_laps: number;
    normalized_position: number;
    sample_count: number;
    live_session_type: LiveSessionType;
    baseline_ready: boolean;
    baseline_collection_started: boolean;
    baseline_progress_percent: number;
    baseline_lap: number | null;
    completed_lap_count: number;
};

const BASELINE_START_POSITION_EPSILON = 0.005;

export class SessionIntelligence {
    private buffer = new TelemetryBuffer();
    private log = new EventLog();
    private sensors = new SensorManager();
    private currentLap: number = 0;
    private currentTrack: string = '';
    private currentPosition: number = 0;
    private baselineStartLap: number | null = null;
    private baselineStartPending: boolean = false;

    // Called every telemetry tick from AnalysisContext.
    tick(sample: TelemetrySample): void {
        // Update track if statics have arrived
        const track: string = getTelemetryTrack(sample);
        if (track && track !== this.currentTrack) {
            this.currentTrack = track;
            this.sensors.setTrack(track);
        }

        this.currentLap = getTelemetryLap(sample);
        this.currentPosition = getTelemetryPosition(sample) ?? this.currentPosition;

        if (
            this.baselineStartPending
            && this.currentPosition <= BASELINE_START_POSITION_EPSILON
        ) {
            this.baselineStartLap = this.currentLap;
            this.baselineStartPending = false;
        }

        const sampleIdx = this.buffer.push(sample);
        this.sensors.tick(sample, sampleIdx, this.log);
    }

    // ── Tool API (called by ai-command-registry handlers) ─────────────────────

    query<TReduce extends ReduceOp>(q: TelemetryQuery<TReduce>): QueryResult<TReduce> {
        return executeQuery(q, this.buffer, this.log, this.currentLap);
    }

    findEvents(params: EventSearchParams): SessionEvent[] {
        return this.log.find({ ...params, currentLap: this.currentLap });
    }

    getAllEvents(): SessionEvent[] {
        return this.log.all();
    }

    getNextCorner(): CornerLookahead | null {
        const corners = getCornersForTrack(this.currentTrack);
        const corner = getNextCorner(corners, this.currentPosition);
        if (!corner) return null;

        // Wrap-around: if corner is behind current pos, it's on the next lap
        const distanceAhead = corner.from > this.currentPosition
            ? corner.from - this.currentPosition
            : 1.0 - this.currentPosition + corner.from;

        return {
            name: corner.name,
            trackPosition: corner.from,
            distanceAhead,
        };
    }

    getRowsForScope(scope: QueryScope): TelemetrySample[] {
        return resolveScope(scope, this.buffer, this.log, this.currentLap);
    }

    getLiveSessionSnapshot(): LiveSessionSnapshot {
        const latest = this.getLatestSample();
        if (!latest) {
            return {
                status: 'empty',
                track: '',
                car: '',
                current_lap: 0,
                completed_laps: 0,
                normalized_position: 0,
                sample_count: 0,
                live_session_type: 'unknown',
                baseline_ready: false,
                baseline_collection_started: false,
                baseline_progress_percent: 0,
                baseline_lap: null,
                completed_lap_count: 0,
            };
        }

        const completedLapNumbers = this.getCompletedLapNumbers();
        return {
            status: 'ready',
            track: this.currentTrack || getTelemetryTrack(latest),
            car: getTelemetryCar(latest),
            current_lap: this.currentLap,
            completed_laps: this.currentLap,
            normalized_position: this.currentPosition,
            sample_count: this.buffer.length,
            live_session_type: detectLiveSessionType(latest),
            baseline_ready: this.hasCompletedBaselineLap(),
            baseline_collection_started: this.hasBaselineCollectionStarted(),
            baseline_progress_percent: this.getBaselineProgressPercent(),
            baseline_lap: this.getBaselineLap(),
            completed_lap_count: completedLapNumbers.length,
        };
    }

    startBaselineCollectionAtLapStart(): void {
        if (this.currentPosition <= BASELINE_START_POSITION_EPSILON) {
            this.baselineStartLap = this.currentLap;
            this.baselineStartPending = false;
            return;
        }

        this.baselineStartLap = null;
        this.baselineStartPending = true;
    }

    hasBaselineCollectionStarted(): boolean {
        return this.baselineStartLap !== null;
    }

    hasCompletedBaselineLap(): boolean {
        return this.baselineStartLap !== null
            && this.currentLap > this.baselineStartLap
            && this.getRowsForLap(this.baselineStartLap).length > 0;
    }

    getBaselineLap(): number | null {
        return this.baselineStartLap;
    }

    getBaselineProgressPercent(): number {
        if (this.hasCompletedBaselineLap()) {
            return 100;
        }

        if (this.baselineStartLap === null) {
            return 0;
        }

        if (this.currentLap > this.baselineStartLap) {
            return 100;
        }

        return Math.max(0, Math.min(99, Math.round(this.currentPosition * 100)));
    }

    getCompletedLapNumbers(): number[] {
        const laps = new Set<number>();
        this.getIndexedRows().forEach(({ sample }) => {
            const lap = getTelemetryLap(sample);
            if (lap < this.currentLap) {
                laps.add(lap);
            }
        });
        return Array.from(laps).sort((a, b) => a - b);
    }

    getLastCompletedLapRows(): TelemetrySample[] {
        if (!this.hasCompletedBaselineLap() || this.baselineStartLap === null) return [];
        return this.getRowsForLap(this.baselineStartLap);
    }

    getRowsForLap(lap: number): TelemetrySample[] {
        return this.getIndexedRows()
            .filter(({ sample }) => getTelemetryLap(sample) === lap)
            .map(({ sample }) => sample);
    }

    getTelemetryWindowForNormalizedRange(args: {
        start_position?: number;
        startPosition?: number;
        end_position?: number;
        endPosition?: number;
        lap?: 'current' | 'last' | number;
    }): NormalizedRangeTelemetryWindow {
        const startPosition = this.clampNormalizedRangePosition(args.start_position ?? args.startPosition);
        const endPosition = this.clampNormalizedRangePosition(args.end_position ?? args.endPosition);
        const lap = this.resolveLap(args.lap ?? 'current');

        const rows = this.getIndexedRows().filter(({ sample }) => {
            const position = getTelemetryPosition(sample);
            if (position === undefined) return false;

            const sampleLap = getTelemetryLap(sample);
            if (startPosition <= endPosition) {
                return sampleLap === lap && position >= startPosition && position <= endPosition;
            }

            return (
                (sampleLap === lap - 1 && position >= startPosition)
                || (sampleLap === lap && position <= endPosition)
            );
        });

        if (rows.length === 0) {
            return {
                status: 'empty',
                startPosition,
                endPosition,
                lap,
                rows: [],
            };
        }

        return {
            status: 'ready',
            startPosition,
            endPosition,
            lap,
            startSampleIdx: rows[0].index,
            endSampleIdx: rows[rows.length - 1].index,
            rows: rows.map(({ sample }) => sample),
        };
    }

    reset(): void {
        this.buffer.reset();
        this.log.reset();
        this.sensors.reset();
        this.currentLap = 0;
        this.currentTrack = '';
        this.currentPosition = 0;
        this.baselineStartLap = null;
        this.baselineStartPending = false;
    }

    private getLatestSample(): TelemetrySample | null {
        return this.buffer.length > 0 ? this.buffer.get(this.buffer.length - 1) : null;
    }

    private getIndexedRows(): IndexedTelemetrySample[] {
        const rows: IndexedTelemetrySample[] = [];
        for (let index = 0; index < this.buffer.length; index += 1) {
            const sample = this.buffer.get(index);
            if (sample) {
                rows.push({ index, sample });
            }
        }
        return rows;
    }

    private resolveLap(lap: 'current' | 'last' | number | undefined): number {
        if (lap === 'current') return this.currentLap;
        if (lap === 'last' || lap === undefined) return Math.max(0, this.currentLap - 1);
        return Math.max(0, Math.floor(Number(lap) || 0));
    }

    private clampNormalizedRangePosition(value: unknown): number {
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return 0;
        return Math.max(0, Math.min(1, parsed));
    }

}
