import { SessionEvent, TelemetrySample, TelemetryQuery, QueryResult, QueryScope, CornerLookahead } from './types';
import { TelemetryBuffer } from './TelemetryBuffer';
import { EventLog, EventSearchParams } from './EventLog';
import { SensorManager } from './SensorManager';
import { executeQuery, resolveScope } from './telemetry-query';
import { getCornersForTrack, getNextCorner } from './track-corners';
import {
    chooseLiveFocusSection,
    compareLiveSectionPerformance,
    createLiveTrackSection,
    detectLiveSessionType,
    estimateSecondsToSection,
    getTelemetryCar,
    getTelemetryLap,
    getTelemetryPosition,
    getTelemetryTrack,
    isPositionInWrappedRange,
    LivePerformanceComparison,
    LiveSectionClassification,
    LiveSectionFocus,
    LiveSessionType,
    LiveTrackSection,
    normalizeLiveSectionClassification,
    normalizedDistanceAhead,
} from './live-performance-analyst';

type IndexedTelemetrySample = {
    index: number;
    sample: TelemetrySample;
};

type SectionTelemetryWindow = {
    status: 'ready' | 'no_live_session' | 'section_not_found' | 'empty';
    section?: LiveTrackSection;
    lap?: number;
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
    completed_lap_count: number;
    section_count: number;
};

const BASELINE_START_POSITION_EPSILON = 0.005;

export class SessionIntelligence {
    private buffer = new TelemetryBuffer();
    private log = new EventLog();
    private sensors = new SensorManager();
    private currentLap: number = 0;
    private currentTrack: string = '';
    private currentPosition: number = 0;
    private baselineStartLap: number | null = 0;
    private baselineStartPendingFromLap: number | null = null;
    private onEvent: ((event: SessionEvent) => void) | null = null;
    private sectionHistory: LiveSectionClassification[] = [];
    private focusSection: LiveSectionFocus | null = null;

    // Optional callback fired on every new event — used to push WS observations.
    onEventEmitted(cb: (event: SessionEvent) => void): void {
        this.onEvent = cb;
        this.sensors.onEventEmitted(cb);
    }

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
            this.baselineStartPendingFromLap !== null
            && (
                this.currentPosition <= BASELINE_START_POSITION_EPSILON
                || this.currentLap > this.baselineStartPendingFromLap
            )
        ) {
            this.baselineStartLap = this.currentLap;
            this.baselineStartPendingFromLap = null;
        }

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
                completed_lap_count: 0,
                section_count: 0,
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
            completed_lap_count: completedLapNumbers.length,
            section_count: this.getKnownTrackSections().length,
        };
    }

    startBaselineCollectionAtLapStart(): void {
        this.sectionHistory = [];
        this.focusSection = null;

        if (this.currentPosition <= BASELINE_START_POSITION_EPSILON) {
            this.baselineStartLap = this.currentLap;
            this.baselineStartPendingFromLap = null;
            return;
        }

        this.baselineStartLap = null;
        this.baselineStartPendingFromLap = this.currentLap;
    }

    hasBaselineCollectionStarted(): boolean {
        return this.baselineStartLap !== null;
    }

    hasCompletedBaselineLap(): boolean {
        return this.baselineStartLap !== null
            && this.currentLap > this.baselineStartLap
            && this.getRowsForLap(this.baselineStartLap).length > 0;
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

    getKnownTrackSections(): LiveTrackSection[] {
        return getCornersForTrack(this.currentTrack)
            .map((corner) => createLiveTrackSection(this.currentTrack, corner));
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

    getSectionTelemetryWindow(args: {
        section_id?: string;
        section_name?: string;
        lap?: 'current' | 'last' | number;
    }): SectionTelemetryWindow {
        const section = this.resolveSection(args.section_id, args.section_name);
        if (!section) {
            return { status: 'section_not_found', rows: [] };
        }

        const lap = this.resolveLap(args.lap);
        const rows = this.getIndexedRows().filter(({ sample }) => (
            getTelemetryLap(sample) === lap
            && isPositionInWrappedRange(getTelemetryPosition(sample) ?? -1, section.from, section.to)
        ));

        if (rows.length === 0) {
            return { status: 'empty', section, lap, rows: [] };
        }

        return {
            status: 'ready',
            section,
            lap,
            startSampleIdx: rows[0].index,
            endSampleIdx: rows[rows.length - 1].index,
            rows: rows.map(({ sample }) => sample),
        };
    }

    recordSectionClassification(raw: Record<string, any>): LiveSectionClassification | null {
        const section = this.resolveSection(raw.section_id || raw.sectionId, raw.section_name || raw.sectionName);
        if (!section) return null;

        const classification = normalizeLiveSectionClassification(raw, section, this.currentLap);
        this.sectionHistory = [
            ...this.sectionHistory.filter((item) => !(
                item.sectionId === classification.sectionId
                && item.lap === classification.lap
                && item.startSampleIdx === classification.startSampleIdx
                && item.endSampleIdx === classification.endSampleIdx
            )),
            classification,
        ].slice(-80);

        this.focusSection = chooseLiveFocusSection(
            this.sectionHistory,
            this.getKnownTrackSections(),
            this.currentPosition,
            {
                estimateSeconds: (candidate) => this.estimateSecondsToSection(candidate),
            },
        ) ?? this.focusSection;

        return classification;
    }

    getSectionHistory(limit = 20): LiveSectionClassification[] {
        return this.sectionHistory
            .slice()
            .sort((a, b) => b.observedAt - a.observedAt)
            .slice(0, Math.max(1, Math.min(limit, 80)));
    }

    getFocusSection(): LiveSectionFocus | null {
        if (this.focusSection) {
            return this.focusSection;
        }

        this.focusSection = chooseLiveFocusSection(
            this.sectionHistory,
            this.getKnownTrackSections(),
            this.currentPosition,
            {
                estimateSeconds: (candidate) => this.estimateSecondsToSection(candidate),
            },
        );
        return this.focusSection;
    }

    clearFocusSection(): void {
        this.focusSection = null;
    }

    compareFocusedSection(latest?: LiveSectionClassification | null): LivePerformanceComparison {
        const focus = this.getFocusSection();
        return compareLiveSectionPerformance(focus?.baseline, latest);
    }

    getSectionTiming(section: LiveTrackSection): { distanceAhead: number; secondsAhead?: number } {
        const target = section.guideFrom ?? section.from;
        return {
            distanceAhead: normalizedDistanceAhead(this.currentPosition, target),
            secondsAhead: this.estimateSecondsToSection(section),
        };
    }

    reset(): void {
        this.buffer.reset();
        this.log.reset();
        this.sensors.reset();
        this.currentLap = 0;
        this.currentTrack = '';
        this.currentPosition = 0;
        this.baselineStartLap = 0;
        this.baselineStartPendingFromLap = null;
        this.sectionHistory = [];
        this.focusSection = null;
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

    private resolveSection(sectionId?: string, sectionName?: string): LiveTrackSection | null {
        const sections = this.getKnownTrackSections();
        return sections.find((section) => (
            (sectionId && section.id === sectionId)
            || (sectionName && section.name.toLowerCase() === sectionName.toLowerCase())
        )) ?? null;
    }

    private estimateSecondsToSection(section: LiveTrackSection): number | undefined {
        return estimateSecondsToSection(
            this.buffer.last(80),
            this.currentPosition,
            section.guideFrom ?? section.from,
        );
    }
}
