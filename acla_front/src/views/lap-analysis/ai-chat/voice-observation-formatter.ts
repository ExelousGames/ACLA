const asRecord = (value: unknown): Record<string, unknown> => (
    value && typeof value === 'object' && !Array.isArray(value)
        ? value as Record<string, unknown>
        : {}
);

const isNumber = (value: unknown): value is number => (
    typeof value === 'number' && Number.isFinite(value)
);

const roundToTenth = (value: number): number => Math.round(value * 10) / 10;

const formatLabels = (value: unknown): string => (
    Array.isArray(value) ? `[${value.map(String).join(', ')}]` : String(value)
);

const summarizeRecordedAnalysisSegments = (value: unknown): string => {
    if (!Array.isArray(value)) return '[]';

    const summaries = value.slice(0, 5).map((item) => {
        const segment = asRecord(item);
        const labels = Array.isArray(segment.child_labels) && segment.child_labels.length
            ? segment.child_labels
            : segment.label_ids;
        return `${segment.id || 'segment'}:${segment.parent_label || 'unknown'} labels=${formatLabels(labels)}`;
    });

    return `[${summaries.join('; ')}]`;
};

export const formatObservationForLlm = (data: Record<string, unknown>): string => {
    const event = typeof data.event === 'string' ? data.event : 'event';

    if (data.source === 'live_performance_analyst' || data.agent_mode === 'live_performance_analyst') {
        const snapshot = asRecord(data.snapshot);
        const sessionType = snapshot.live_session_type || snapshot.session_type || 'unknown';
        const completedLaps = snapshot.completed_laps;
        const currentLap = snapshot.current_lap;
        const track = snapshot.track || 'current track';

        if (event === 'live_analysis_plan_started') {
            return (
                'live_performance_analyst plan started: '
                + `track=${track}, current_lap=${currentLap}, completed_laps=${completedLaps}, `
                + `session_type=${sessionType}, baseline_ready=${snapshot.baseline_ready}.`
            );
        }

        if (event === 'collecting_baseline') {
            return (
                'live_performance_analyst collecting baseline: '
                + `track=${track}, current_lap=${currentLap}, completed_laps=${completedLaps}, `
                + `session_type=${sessionType}.`
            );
        }

        if (event === 'baseline_classifier_request_ready') {
            return (
                'live_performance_analyst baseline classifier request ready: '
                + `track=${track}, current_lap=${currentLap}, completed_laps=${completedLaps}, `
                + `session_type=${sessionType}, baseline_ready=${snapshot.baseline_ready}.`
            );
        }

        if (event === 'recorded_analysis_ready') {
            const analysisContext = asRecord(data.analysis);
            const analysis = asRecord(analysisContext.analysis);
            return (
                'live_performance_analyst recorded classifier analysis ready. '
                + `status=${analysisContext.status}; `
                + `session_id=${analysisContext.session_id}; `
                + `samples_analyzed=${analysis.samples_analyzed}; `
                + `segment_count=${analysis.segment_count}; `
                + `returned_segment_count=${analysis.returned_segment_count}; `
                + `segments=${summarizeRecordedAnalysisSegments(analysis.segments)}, `
                + `session_type=${sessionType}.`
            );
        }

        if ([
            'baseline_lap_record_required',
            'recorded_analysis_unavailable',
            'recorded_analysis_failed',
        ].includes(event)) {
            return (
                'live_performance_analyst cannot complete recorded classifier analysis. '
                + `reason=${event}; message=${data.message}.`
            );
        }

        if (event === 'live_analysis_window') {
            const focus = asRecord(data.focus);
            const section = asRecord(focus.section);
            const baseline = asRecord(focus.baseline);
            const timing = asRecord(focus.timing);
            return (
                'live_performance_analyst coaching window. '
                + `section=${section.id}:${section.name} `
                + `range=[${section.from},${section.to}], `
                + `mistakes=${baseline.mistakeCount}, severity=${baseline.severity}, `
                + `labels=${formatLabels(baseline.childLabels)}, seconds_ahead=${timing.secondsAhead}, `
                + `distance_ahead=${timing.distanceAhead}, session_type=${sessionType}.`
            );
        }
    }

    if (event === 'attack_window' || event === 'defense_threat') {
        const nextCorner = asRecord(data.next_corner);
        const location = data.projected_section || nextCorner.name || 'the next section';
        const opponent = data.opponent_id ?? data.opponent_slot;
        const details: string[] = [];

        if (isNumber(data.time_to_overlap_seconds)) {
            details.push(`arriving in ${roundToTenth(data.time_to_overlap_seconds)}s`);
        }
        if (isNumber(data.closing_speed_mps)) {
            details.push(`closing speed ${roundToTenth(data.closing_speed_mps)} m/s`);
        }
        if (isNumber(data.distance_m)) {
            details.push(`distance ${roundToTenth(data.distance_m)}m`);
        }
        if (opponent !== undefined && opponent !== null) {
            details.push(`opponent ${opponent}`);
        }

        const detailText = details.length ? details.join(', ') : 'coordinate-derived relative motion';
        if (event === 'attack_window') {
            return (
                `overtake_agent attack_window at ${location}: ${detailText}. `
                + 'Tell the driver an attack is opening and give one short action.'
            );
        }
        return (
            `overtake_agent defense_threat at ${location}: ${detailText}. `
            + 'Tell the driver to defend and give one short action.'
        );
    }

    if (event === 'opportunity_forecast') {
        const selected = asRecord(data.selected_opportunity);
        const opportunities = Array.isArray(data.opportunities) ? data.opportunities : [];
        const labels = opportunities.slice(0, 3).flatMap((item) => {
            const opportunity = asRecord(item);
            if (!Object.keys(opportunity).length) return [];
            let label = String(opportunity.label_name || opportunity.label_id || 'opportunity');
            if (isNumber(opportunity.probability)) {
                label = `${label} ${Math.round(opportunity.probability * 100)}%`;
            }
            if (opportunity.circuit_section_name) {
                label = `${label} at ${opportunity.circuit_section_name}`;
            }
            return [label];
        });
        const horizonText = data.horizon_seconds !== undefined && data.horizon_seconds !== null
            ? `next ${data.horizon_seconds}s`
            : 'upcoming';
        const nextCorner = asRecord(data.next_corner);
        const sectionMatch = asRecord(data.circuit_section_match);
        const bestMatch = asRecord(sectionMatch.best_match);
        const selectedSection = selected.circuit_section_name || bestMatch.name;
        const locationBits: string[] = [];
        if (selectedSection) locationBits.push(`forecast section ${selectedSection}`);
        if (nextCorner.name) locationBits.push(`next corner ${nextCorner.name}`);
        const locationText = locationBits.length ? ` (${locationBits.join('; ')})` : '';

        if (labels.length) {
            if (data.mode === 'agent' || data.source === 'overtake_agent') {
                return (
                    `overtake_agent alert ${horizonText}${locationText}: ${labels.join(', ')}. `
                    + 'Tell the driver the next possible pass window relative to the circuit section and give one short action.'
                );
            }
            return (
                `opportunity_forecast ${horizonText}${locationText}: ${labels.join(', ')}. `
                + 'Explain what it means and what the driver should do next in one short engineer radio message.'
            );
        }

        return (
            `opportunity_forecast ${horizonText}${locationText}: no strong opportunity labels. `
            + 'If useful, tell the driver to keep building the setup in one short radio message.'
        );
    }

    const bits = [event];
    ['section', 'lap', 'lap_number'].forEach((key) => {
        const value = data[key];
        if (value !== undefined && value !== null) bits.push(`${key}=${value}`);
    });
    const telemetryRows = Array.isArray(data.telemetry_rows) ? data.telemetry_rows.length : 0;
    if (telemetryRows) bits.push(`telemetry_rows=${telemetryRows}`);
    return `${bits.join(' ')}. Respond with one short engineer suggestion.`;
};

export const buildFormattedObservationFrame = (data: Record<string, unknown>) => ({
    type: 'observation' as const,
    data: {
        ...data,
        text: formatObservationForLlm(data),
    },
});
