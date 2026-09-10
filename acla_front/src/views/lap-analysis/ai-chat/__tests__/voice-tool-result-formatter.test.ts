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
            result: {
                event: 'custom_alert',
                section: 'T1',
                telemetry_row_count: 2,
            },
        });
    });

    it.each(['working', 'complete', 'failed', 'custom-status'])(
        'preserves the supplied status %s without rewriting it',
        (status) => {
            const data = { name: 'background_tool', status, progress: 50 };
            expect(buildFormattedToolResultFrame(data, 'call-1')).toEqual({
                type: 'tool_result', id: 'call-1', name: 'background_tool', result: data,
            });
        },
    );

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
                    segments: [{ id: 'segment-1', labels: [{ label_name: 'late brake', start_index: 0, end_index: 1 }] }],
                },
            },
        }, 'workflow-test')).toEqual({
            type: 'tool_result',
            id: 'workflow-test',
            name: 'live_performance_analyst',
            result: {
                source: 'live_performance_analyst',
                agent_mode: 'live_performance_analyst',
                event: 'recorded_analysis_ready',
            },
        });
    });
});
