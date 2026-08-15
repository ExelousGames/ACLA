import { buildFormattedToolResultFrame } from '../voice-tool-result-formatter';

describe('buildFormattedToolResultFrame', () => {
    it('builds the formatted websocket tool_result frame sent to the backend', () => {
        expect(buildFormattedToolResultFrame({
            event: 'custom_alert',
            section: 'T1',
            telemetry_rows: [{}, {}],
        }, 'workflow-test')).toEqual({
            type: 'tool_result',
            id: 'workflow-test',
            name: 'custom_alert',
            final: false,
            result: {
                event: 'custom_alert',
                section: 'T1',
                telemetry_row_count: 2,
                status: 'complete',
            },
        });
    });

    it('does not duplicate the run id in a nested native tool message', () => {
        const frame = buildFormattedToolResultFrame({
            run_id: 'tool-7',
            event: 'custom_alert',
            section: 'T1',
        });

        expect(frame).toEqual(expect.objectContaining({
            type: 'tool_result',
            id: 'tool-7',
            name: 'custom_alert',
        }));
        expect((frame as any).messages).toBeUndefined();
    });

    it('does not expose classifier analysis in live analyst status frames', () => {
        expect(buildFormattedToolResultFrame({
            source: 'live_performance_analyst',
            agent_mode: 'live_performance_analyst',
            event: 'recorded_analysis_ready',
            analysis: {
                analysis: {
                    segments: [{ id: 'segment-1', labels: ['late brake'] }],
                },
            },
        }, 'workflow-test')).toEqual({
            type: 'tool_result',
            id: 'workflow-test',
            name: 'live_performance_analyst',
            final: false,
            result: {
                source: 'live_performance_analyst',
                agent_mode: 'live_performance_analyst',
                event: 'recorded_analysis_ready',
                status: 'complete',
            },
        });
    });
});
