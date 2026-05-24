import { SessionEvent, TelemetrySample, TelemetryQuery, QueryResult, QueryScope, CornerLookahead } from './types';
import { TelemetryBuffer } from './TelemetryBuffer';
import { EventLog, EventSearchParams } from './EventLog';
import { SensorManager } from './SensorManager';
import { executeQuery, getSchemaInfo, resolveScope } from './telemetry-query';
import { getCornersForTrack, getNextCorner } from './track-corners';

export class SessionIntelligence {
    private buffer = new TelemetryBuffer();
    private log = new EventLog();
    private sensors = new SensorManager();
    private currentLap: number = 0;
    private currentTrack: string = '';
    private currentPosition: number = 0;
    private onEvent: ((event: SessionEvent) => void) | null = null;

    // Optional callback fired on every new event — used to push WS observations.
    onEventEmitted(cb: (event: SessionEvent) => void): void {
        this.onEvent = cb;
        this.sensors.onEventEmitted(cb);
    }

    // Called every telemetry tick from AnalysisContext.
    tick(sample: TelemetrySample): void {
        // Update track if statics have arrived
        const track: string = sample['Static_track'] ?? '';
        if (track && track !== this.currentTrack) {
            this.currentTrack = track;
            this.sensors.setTrack(track);
        }

        this.currentLap = sample['Graphics_completed_laps'] ?? this.currentLap;
        this.currentPosition = sample['Graphics_normalized_car_position'] ?? this.currentPosition;

        const sampleIdx = this.buffer.push(sample);
        this.sensors.tick(sample, sampleIdx, this.log);
    }

    // ── Tool API (called by ai-command-registry handlers) ─────────────────────

    query(q: TelemetryQuery): QueryResult {
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

    getSchema(): { groups: string[]; fields: string[] } {
        return getSchemaInfo();
    }

    getRowsForScope(scope: QueryScope): TelemetrySample[] {
        return resolveScope(scope, this.buffer, this.log, this.currentLap);
    }

    reset(): void {
        this.buffer.reset();
        this.log.reset();
        this.sensors.reset();
        this.currentLap = 0;
        this.currentTrack = '';
        this.currentPosition = 0;
    }
}
