import {
    buildFormattedObservationFrame,
    formatObservationForLlm,
} from '../voice-observation-formatter';

describe('formatObservationForLlm', () => {
    it('formats live analyst observations without backend-only teaching actions', () => {
        const collectingMsg = formatObservationForLlm({
            source: 'live_performance_analyst',
            agent_mode: 'live_performance_analyst',
            event: 'collecting_baseline',
            snapshot: {
                track: 'brands_hatch',
                current_lap: 1,
                completed_laps: 0,
                live_session_type: 'solo_practice',
            },
        });
        const analysisMsg = formatObservationForLlm({
            source: 'live_performance_analyst',
            agent_mode: 'live_performance_analyst',
            event: 'recorded_analysis_ready',
            snapshot: { track: 'brands_hatch', live_session_type: 'solo_practice' },
            analysis: {
                status: 'ready',
                session_id: 'session-1',
                analysis: {
                    samples_analyzed: 120,
                    segment_count: 1,
                    returned_segment_count: 1,
                    segments: [
                        {
                            id: 'segment-1',
                            parent_label: 'Paddock Hill',
                            child_labels: ['Initiate brake too late'],
                        },
                    ],
                },
            },
        });
        const baselineMsg = formatObservationForLlm({
            source: 'live_performance_analyst',
            agent_mode: 'live_performance_analyst',
            event: 'recorded_session_required',
            message: 'Recorded-session AI analysis is required.',
            snapshot: { track: 'brands_hatch', live_session_type: 'solo_practice' },
        });
        const coachingMsg = formatObservationForLlm({
            source: 'live_performance_analyst',
            agent_mode: 'live_performance_analyst',
            event: 'live_analysis_window',
            snapshot: { live_session_type: 'traffic_or_race' },
            focus: {
                section: { id: 'brands_hatch2', name: 'Paddock Hill', from: 0.1, to: 0.2 },
                baseline: { mistakeCount: 2, severity: 2, childLabels: ['Initiate brake too late'] },
                timing: { secondsAhead: 9.5, distanceAhead: 0.06 },
            },
        });

        expect(collectingMsg).toContain('track=brands_hatch');
        expect(collectingMsg).toContain('completed_laps=0');
        expect(collectingMsg).not.toContain('get_live_focus_section');
        expect(collectingMsg).not.toContain('classify_live_section');
        expect(collectingMsg).not.toContain('Do not');
        expect(analysisMsg).toContain('recorded classifier analysis ready');
        expect(analysisMsg).toContain('segment_count=1');
        expect(analysisMsg).toContain('Paddock Hill');
        expect(analysisMsg).not.toContain('goal=');
        expect(analysisMsg).not.toContain('focus=');
        expect(analysisMsg).not.toContain('plan=');
        expect(baselineMsg).toContain('Recorded-session AI analysis is required.');
        expect(baselineMsg).not.toContain('Explain briefly');
        expect(baselineMsg).not.toContain('classify_live_section');
        expect(coachingMsg).not.toContain('Call show_map');
        expect(coachingMsg).not.toContain('Give one short correction');
        expect(coachingMsg).toContain('traffic_or_race');
    });

    it('formats overtake and generic observations', () => {
        expect(formatObservationForLlm({
            source: 'overtake_agent',
            agent_mode: 'overtake',
            event: 'attack_window',
            projected_section: 'Druids',
            time_to_overlap_seconds: 4.24,
            closing_speed_mps: 3.04,
            distance_m: 12.02,
            opponent_slot: 7,
        })).toBe(
            'overtake_agent attack_window at Druids: arriving in 4.2s, closing speed 3 m/s, distance 12m, opponent 7. '
            + 'Tell the driver an attack is opening and give one short action.',
        );

        expect(formatObservationForLlm({
            event: 'custom_alert',
            section: 'T1',
            lap: 3,
            telemetry_rows: [{}, {}],
        })).toBe('custom_alert section=T1 lap=3 telemetry_rows=2. Respond with one short engineer suggestion.');
    });

    it('builds the formatted websocket frame sent to the backend', () => {
        expect(buildFormattedObservationFrame({
            event: 'custom_alert',
            section: 'T1',
        })).toEqual({
            type: 'observation',
            data: {
                event: 'custom_alert',
                section: 'T1',
                text: 'custom_alert section=T1. Respond with one short engineer suggestion.',
            },
        });
    });
});
