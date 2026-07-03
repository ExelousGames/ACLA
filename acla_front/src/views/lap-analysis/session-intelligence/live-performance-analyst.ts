import { CornerDefinition, FieldStats, TelemetrySample } from './types';

export type LiveSessionType = 'solo_practice' | 'traffic_or_race' | 'unknown';

export type LiveTrackSection = {
    id: string;
    name: string;
    from: number;
    to: number;
    guideFrom?: number;
};

export type LiveSectionClassification = {
    sectionId: string;
    sectionName: string;
    lap: number;
    startSampleIdx: number;
    endSampleIdx: number;
    mistakeCount: number;
    expertAdherenceCount: number;
    severity: number;
    confidence: number;
    parentLabel?: string | null;
    childLabels: string[];
    telemetryStats?: Record<string, FieldStats>;
    observedAt: number;
};

export type LiveSectionFocus = {
    section: LiveTrackSection;
    baseline: LiveSectionClassification;
    selectedAt: number;
    reason: string;
    score: number;
};

export type LivePerformanceComparison = {
    status: 'improved' | 'regressed' | 'similar' | 'insufficient_data';
    mistakeDelta: number;
    severityDelta: number;
    expertAdherenceDelta: number;
    confidence: number;
};

export type LiveAnalystToolStatus = Record<string, unknown> & {
    source: 'live_performance_analyst';
    agent_mode: 'live_performance_analyst';
    event: string;
};

export const buildBaselineClassifierRequestReadyToolStatus = (
    snapshot: Record<string, unknown>,
): LiveAnalystToolStatus => ({
    source: 'live_performance_analyst',
    agent_mode: 'live_performance_analyst',
    event: 'baseline_classifier_request_ready',
    snapshot,
    message: 'Baseline complete. Request the recorded-session classifier through the frontend before choosing a focus.',
});

export type LiveAnalystRecordedAnalysisError =
    | 'baseline_lap_record_required'
    | 'recorded_analysis_unavailable'
    | 'recorded_analysis_failed';

export const buildLiveAnalysisPlanStartedToolStatus = (
    snapshot: Record<string, unknown>,
): LiveAnalystToolStatus => ({
    source: 'live_performance_analyst',
    agent_mode: 'live_performance_analyst',
    event: 'live_analysis_plan_started',
    snapshot,
    message: 'Live analysis procedure started. Collect a baseline first, then use recorded-session analysis to choose a focus.',
});

export const buildRecordedAnalysisErrorToolStatus = (
    error: LiveAnalystRecordedAnalysisError,
    message: string,
    snapshot?: Record<string, unknown> | null,
): LiveAnalystToolStatus => ({
    source: 'live_performance_analyst',
    agent_mode: 'live_performance_analyst',
    event: error,
    ...(snapshot ? { snapshot } : {}),
    message,
});

export const buildRecordedAnalysisReadyToolStatus = (
    analysis: unknown,
    snapshot?: Record<string, unknown> | null,
): LiveAnalystToolStatus => ({
    source: 'live_performance_analyst',
    agent_mode: 'live_performance_analyst',
    event: 'recorded_analysis_ready',
    ...(snapshot ? { snapshot } : {}),
    analysis,
});

export const buildLiveAnalysisWindowToolStatus = (
    snapshot: Record<string, unknown>,
    focus: unknown,
): LiveAnalystToolStatus => ({
    source: 'live_performance_analyst',
    agent_mode: 'live_performance_analyst',
    event: 'live_analysis_window',
    snapshot,
    focus,
});

export const DEFAULT_ANALYST_MIN_LEAD_SECONDS = 8;
export const DEFAULT_ANALYST_MIN_DISTANCE = 0.04;
export const DEFAULT_ANALYST_COOLDOWN_MS = 20000;

const toFiniteNumber = (value: unknown): number | undefined => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
};

const countActiveCars = (value: unknown): number | undefined => {
    if (Array.isArray(value)) {
        const count = value.filter((item) => {
            const parsed = Number(item);
            return Number.isFinite(parsed) && parsed !== 0;
        }).length;
        return count > 0 ? count : undefined;
    }

    return toFiniteNumber(value);
};

export const detectLiveSessionType = (sample: TelemetrySample | null | undefined): LiveSessionType => {
    if (!sample) return 'unknown';

    const counts = [
        countActiveCars(sample.Graphics_active_cars_count),
        countActiveCars(sample.Graphics_active_cars),
        countActiveCars(sample.Graphics?.active_cars),
        countActiveCars(sample.Static_num_cars),
        countActiveCars(sample.Static?.num_cars),
    ].filter((count): count is number => count !== undefined);

    if (counts.length === 0) return 'unknown';
    if (counts.some((count) => count <= 1)) return 'solo_practice';
    return 'traffic_or_race';
};

export const getTelemetryLap = (sample: TelemetrySample | null | undefined): number => (
    toFiniteNumber(sample?.Graphics_completed_laps)
    ?? toFiniteNumber(sample?.Graphics_completed_lap)
    ?? 0
);

export const getTelemetryPosition = (sample: TelemetrySample | null | undefined): number | undefined => {
    const value = toFiniteNumber(
        sample?.Graphics_normalized_car_position
        ?? sample?.graphics_normalized_car_position
        ?? sample?.normalized_car_position
        ?? sample?.car_position
    );

    if (value === undefined) return undefined;
    return Math.max(0, Math.min(1, value));
};

export const getTelemetryTrack = (sample: TelemetrySample | null | undefined): string => (
    typeof sample?.Static_track === 'string' && sample.Static_track
        ? sample.Static_track
        : typeof sample?.Static?.track === 'string'
            ? sample.Static.track
            : ''
);

export const getTelemetryCar = (sample: TelemetrySample | null | undefined): string => (
    typeof sample?.Static_car_model === 'string' && sample.Static_car_model
        ? sample.Static_car_model
        : typeof sample?.Static?.car_model === 'string'
            ? sample.Static.car_model
            : ''
);

export const createLiveTrackSection = (
    track: string,
    corner: CornerDefinition,
): LiveTrackSection => ({
    id: `${track || 'unknown'}:${corner.name}:${corner.from}-${corner.to}`,
    name: corner.name,
    from: corner.from,
    to: corner.to,
    guideFrom: corner.guideFrom,
});

export const isPositionInWrappedRange = (
    position: number,
    start: number,
    end: number,
): boolean => (
    start <= end
        ? position >= start && position <= end
        : position >= start || position <= end
);

export const normalizedDistanceAhead = (
    currentPosition: number,
    targetPosition: number,
): number => (
    targetPosition >= currentPosition
        ? targetPosition - currentPosition
        : 1 - currentPosition + targetPosition
);

const getTimestampSeconds = (sample: TelemetrySample): number | undefined => {
    const raw = toFiniteNumber(sample.Physics_timestamp ?? sample.timestamp);
    if (raw === undefined) return undefined;
    return raw > 100000 ? raw / 1000 : raw;
};

export const estimateSecondsToSection = (
    recentRows: TelemetrySample[],
    currentPosition: number,
    sectionStart: number,
): number | undefined => {
    if (recentRows.length < 2) return undefined;

    const first = recentRows.find((row) => getTelemetryPosition(row) !== undefined && getTimestampSeconds(row) !== undefined);
    const last = [...recentRows].reverse().find((row) => getTelemetryPosition(row) !== undefined && getTimestampSeconds(row) !== undefined);
    if (!first || !last || first === last) return undefined;

    const firstPosition = getTelemetryPosition(first);
    const lastPosition = getTelemetryPosition(last);
    const firstTs = getTimestampSeconds(first);
    const lastTs = getTimestampSeconds(last);
    if (firstPosition === undefined || lastPosition === undefined || firstTs === undefined || lastTs === undefined) return undefined;

    const elapsedSeconds = lastTs - firstTs;
    if (elapsedSeconds <= 0) return undefined;

    const progress = normalizedDistanceAhead(firstPosition, lastPosition);
    const normalizedPerSecond = progress / elapsedSeconds;
    if (!Number.isFinite(normalizedPerSecond) || normalizedPerSecond <= 0) return undefined;

    return normalizedDistanceAhead(currentPosition, sectionStart) / normalizedPerSecond;
};

export const hasEnoughCoachingLead = (
    distanceAhead: number,
    secondsAhead?: number,
    minDistance = DEFAULT_ANALYST_MIN_DISTANCE,
    minSeconds = DEFAULT_ANALYST_MIN_LEAD_SECONDS,
): boolean => (
    distanceAhead >= minDistance
    && (secondsAhead === undefined || secondsAhead >= minSeconds)
);

export const chooseLiveFocusSection = (
    history: LiveSectionClassification[],
    sections: LiveTrackSection[],
    currentPosition: number,
    options: {
        minDistance?: number;
        minLeadSeconds?: number;
        estimateSeconds?: (section: LiveTrackSection) => number | undefined;
        now?: number;
    } = {},
): LiveSectionFocus | null => {
    const bySection = new Map(sections.map((section) => [section.id, section]));
    const now = options.now ?? Date.now();

    const candidates = history
        .map((classification) => {
            const section = bySection.get(classification.sectionId);
            if (!section) return null;

            const distanceAhead = normalizedDistanceAhead(currentPosition, section.guideFrom ?? section.from);
            const secondsAhead = options.estimateSeconds?.(section);
            if (!hasEnoughCoachingLead(
                distanceAhead,
                secondsAhead,
                options.minDistance,
                options.minLeadSeconds,
            )) {
                return null;
            }

            const repeatedCount = history.filter((item) => item.sectionId === classification.sectionId && item.mistakeCount > 0).length;
            const score = (classification.mistakeCount * 3)
                + (classification.severity * 2)
                + classification.confidence
                + repeatedCount
                + Math.max(0, 1 - ((now - classification.observedAt) / 120000));

            return { section, classification, score, repeatedCount };
        })
        .filter((candidate): candidate is {
            section: LiveTrackSection;
            classification: LiveSectionClassification;
            score: number;
            repeatedCount: number;
        } => Boolean(candidate))
        .sort((a, b) => b.score - a.score || b.classification.observedAt - a.classification.observedAt);

    const best = candidates[0];
    if (!best || best.classification.mistakeCount <= 0) return null;

    return {
        section: best.section,
        baseline: best.classification,
        selectedAt: now,
        reason: best.repeatedCount > 1 ? 'repeated_mistake' : 'highest_priority_mistake',
        score: best.score,
    };
};

export const compareLiveSectionPerformance = (
    baseline?: LiveSectionClassification | null,
    latest?: LiveSectionClassification | null,
): LivePerformanceComparison => {
    if (!baseline || !latest) {
        return {
            status: 'insufficient_data',
            mistakeDelta: 0,
            severityDelta: 0,
            expertAdherenceDelta: 0,
            confidence: 0,
        };
    }

    const mistakeDelta = latest.mistakeCount - baseline.mistakeCount;
    const severityDelta = latest.severity - baseline.severity;
    const expertAdherenceDelta = latest.expertAdherenceCount - baseline.expertAdherenceCount;
    const score = (-mistakeDelta * 2) + (-severityDelta) + expertAdherenceDelta;

    return {
        status: score > 0.5 ? 'improved' : score < -0.5 ? 'regressed' : 'similar',
        mistakeDelta,
        severityDelta,
        expertAdherenceDelta,
        confidence: Math.min(baseline.confidence, latest.confidence),
    };
};

export const normalizeLiveSectionClassification = (
    raw: Record<string, any>,
    section: LiveTrackSection,
    fallbackLap: number,
): LiveSectionClassification => ({
    sectionId: raw.section_id || raw.sectionId || section.id,
    sectionName: raw.section_name || raw.sectionName || section.name,
    lap: Math.max(0, Math.floor(toFiniteNumber(raw.lap) ?? fallbackLap)),
    startSampleIdx: Math.max(0, Math.floor(toFiniteNumber(raw.start_sample_idx ?? raw.startSampleIdx) ?? 0)),
    endSampleIdx: Math.max(0, Math.floor(toFiniteNumber(raw.end_sample_idx ?? raw.endSampleIdx) ?? 0)),
    mistakeCount: Math.max(0, Math.floor(toFiniteNumber(raw.mistake_count ?? raw.mistakeCount) ?? 0)),
    expertAdherenceCount: Math.max(0, Math.floor(toFiniteNumber(raw.expert_adherence_count ?? raw.expertAdherenceCount) ?? 0)),
    severity: Math.max(0, toFiniteNumber(raw.severity) ?? 0),
    confidence: Math.max(0, Math.min(1, toFiniteNumber(raw.confidence) ?? 0)),
    parentLabel: raw.parent_label ?? raw.parentLabel ?? null,
    childLabels: Array.isArray(raw.child_labels)
        ? raw.child_labels.map(String)
        : Array.isArray(raw.childLabels)
            ? raw.childLabels.map(String)
            : [],
    telemetryStats: raw.telemetry_stats ?? raw.telemetryStats,
    observedAt: toFiniteNumber(raw.observed_at ?? raw.observedAt) ?? Date.now(),
});
