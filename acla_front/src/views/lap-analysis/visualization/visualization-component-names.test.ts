import {
    deriveTelemetryMetricFamilies,
    getTelemetryComponentName,
    getVisualizationComponentName,
} from './visualization-component-names';

describe('semantic visualization component names', () => {
    it('sorts and deduplicates metric families deterministically', () => {
        expect(deriveTelemetryMetricFamilies({ fields: ['Physics_speed_kmh', 'Physics_brake', 'speed'] }))
            .toEqual(['brake', 'speed']);
        expect(getTelemetryComponentName(['speed', 'brake', 'speed'])).toBe('telemetry:brake+speed');
    });

    it('derives families from nested tool data instead of generic container keys', () => {
        expect(getVisualizationComponentName('telemetry-overview', { data: { metrics: ['speed'] } }))
            .toBe('telemetry:speed');
        expect(getVisualizationComponentName('telemetry-overview', { data: [{ Physics_brake: 0.4 }] }))
            .toBe('telemetry:brake');
    });

    it('uses one stable type-based name for non-telemetry charts', () => {
        expect(getVisualizationComponentName('analysis-results', { metrics: ['speed'] }))
            .toBe('visualization:analysis-results');
    });
});
