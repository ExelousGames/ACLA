import {
    buildFormattedToolResultFrame,
    formatToolResultForLlm,
} from '../voice-tool-result-formatter';

describe('formatToolResultForLlm', () => {
    it('formats live analyst tool statuses without backend-only teaching actions', () => {
        const planStartedMsg = formatToolResultForLlm({
            source: 'live_performance_analyst',
            agent_mode: 'live_performance_analyst',
            event: 'live_analysis_plan_started',
            snapshot: {
                track: 'brands_hatch',
                current_lap: 1,
                completed_laps: 0,
                baseline_ready: false,
                live_session_type: 'solo_practice',
            },
        });
        const classifierReadyMsg = formatToolResultForLlm({
            source: 'live_performance_analyst',
            agent_mode: 'live_performance_analyst',
            event: 'baseline_classifier_request_ready',
            snapshot: {
                track: 'brands_hatch',
                current_lap: 2,
                completed_laps: 1,
                baseline_ready: true,
                live_session_type: 'solo_practice',
            },
        });
        const analysisMsg = formatToolResultForLlm({
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
        const baselineMsg = formatToolResultForLlm({
            source: 'live_performance_analyst',
            agent_mode: 'live_performance_analyst',
            event: 'baseline_lap_record_required',
            message: 'Cached baseline lap records are required.',
            snapshot: { track: 'brands_hatch', live_session_type: 'solo_practice' },
        });
        const coachingMsg = formatToolResultForLlm({
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

        expect(planStartedMsg).toContain('plan started');
        expect(planStartedMsg).toContain('baseline_ready=false');
        expect(planStartedMsg).not.toContain('Respond with one short');
        expect(classifierReadyMsg).toContain('baseline classifier request ready');
        expect(classifierReadyMsg).toContain('baseline_ready=true');
        expect(classifierReadyMsg).not.toContain('Respond with one short');
        expect(analysisMsg).toContain('recorded classifier analysis ready');
        expect(analysisMsg).toContain('segment_count=1');
        expect(analysisMsg).toContain('Paddock Hill');
        expect(analysisMsg).not.toContain('goal=');
        expect(analysisMsg).not.toContain('focus=');
        expect(analysisMsg).not.toContain('plan=');
        expect(baselineMsg).toContain('Cached baseline lap records are required.');
        expect(baselineMsg).not.toContain('Explain briefly');
        expect(baselineMsg).not.toContain('classify_live_section');
        expect(coachingMsg).not.toContain('Call show_map');
        expect(coachingMsg).not.toContain('Give one short correction');
        expect(coachingMsg).toContain('traffic_or_race');
    });

    it('formats overtake and generic tool statuses', () => {
        expect(formatToolResultForLlm({
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

        expect(formatToolResultForLlm({
            event: 'custom_alert',
            section: 'T1',
            lap: 3,
            telemetry_rows: [{}, {}],
        })).toBe('custom_alert section=T1 lap=3 telemetry_rows=2. Respond with one short engineer suggestion.');
    });

    it('builds the formatted websocket tool_result frame sent to the backend', () => {
        expect(buildFormattedToolResultFrame({
            event: 'custom_alert',
            section: 'T1',
        })).toEqual({
            type: 'tool_result',
            id: undefined,
            name: 'custom_alert',
            result: {
                event: 'custom_alert',
                section: 'T1',
                status: 'complete',
                text: 'custom_alert section=T1. Respond with one short engineer suggestion.',
            },
        });
    });
});
