import { SessionIntelligence } from '../SessionIntelligence';
import {
    chooseLiveFocusSection,
    compareLiveSectionPerformance,
    createLiveTrackSection,
    detectLiveSessionType,
    isPositionInWrappedRange,
} from '../live-performance-analyst';
import { getCornersForTrack } from '../track-corners';

describe('live performance analyst helpers', () => {
    it('detects live session type from ACC car-count fields', () => {
        expect(detectLiveSessionType({ Static_num_cars: 1 })).toBe('solo_practice');
        expect(detectLiveSessionType({ Graphics_active_cars_count: 1 })).toBe('solo_practice');
        expect(detectLiveSessionType({ Graphics_active_cars_count: 8 })).toBe('traffic_or_race');
        expect(detectLiveSessionType({})).toBe('unknown');
    });

    it('matches wrap-aware track sections across start finish', () => {
        expect(isPositionInWrappedRange(0.98, 0.97, 0.09)).toBe(true);
        expect(isPositionInWrappedRange(0.03, 0.97, 0.09)).toBe(true);
        expect(isPositionInWrappedRange(0.5, 0.97, 0.09)).toBe(false);
    });

    it('selects high-priority mistakes only when there is enough coaching lead', () => {
        const [t1, t2] = getCornersForTrack('brands_hatch').map((corner) => createLiveTrackSection('brands_hatch', corner));
        const now = 100000;
        const history = [
            {
                sectionId: t1.id,
                sectionName: t1.name,
                lap: 0,
                startSampleIdx: 0,
                endSampleIdx: 20,
                mistakeCount: 4,
                expertAdherenceCount: 0,
                severity: 3,
                confidence: 0.9,
                parentLabel: 'Mistake',
                childLabels: ['wide exit'],
                observedAt: now - 1000,
            },
            {
                sectionId: t2.id,
                sectionName: t2.name,
                lap: 0,
                startSampleIdx: 30,
                endSampleIdx: 60,
                mistakeCount: 2,
                expertAdherenceCount: 1,
                severity: 1,
                confidence: 0.8,
                parentLabel: 'Mistake',
                childLabels: ['late brake'],
                observedAt: now,
            },
        ];

        const focus = chooseLiveFocusSection(history, [t1, t2], 0.5, { now });

        expect(focus?.section.id).toBe(t1.id);
        expect(focus?.reason).toBe('highest_priority_mistake');
    });

    it('compares focused section performance between passes', () => {
        const baseline = {
            sectionId: 'section',
            sectionName: 'Section',
            lap: 0,
            startSampleIdx: 0,
            endSampleIdx: 20,
            mistakeCount: 3,
            expertAdherenceCount: 0,
            severity: 3,
            confidence: 0.9,
            parentLabel: 'Mistake',
            childLabels: [],
            observedAt: 1,
        };
        const latest = {
            ...baseline,
            lap: 1,
            mistakeCount: 1,
            expertAdherenceCount: 2,
            severity: 1,
            observedAt: 2,
        };

        expect(compareLiveSectionPerformance(baseline, latest)).toMatchObject({
            status: 'improved',
            mistakeDelta: -2,
            severityDelta: -2,
            expertAdherenceDelta: 2,
        });
    });
});

describe('SessionIntelligence live analyst section state', () => {
    it('does not infer a baseline before the analyst starts collection', () => {
        const intelligence = new SessionIntelligence();
        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.5,
        });

        expect(intelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: false,
            baseline_collection_started: false,
            baseline_progress_percent: 0,
            baseline_lap: null,
            completed_laps: 0,
            live_session_type: 'solo_practice',
        });
    });

    it('starts analyst baseline collection at the next lap start', () => {
        const intelligence = new SessionIntelligence();
        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.45,
        });

        intelligence.startBaselineCollectionAtLapStart();

        expect(intelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: false,
            baseline_collection_started: false,
            baseline_progress_percent: 0,
        });

        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 1,
            Graphics_normalized_car_position: 0.01,
        });

        expect(intelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: false,
            baseline_collection_started: true,
            baseline_progress_percent: 1,
            baseline_lap: 1,
        });

        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 2,
            Graphics_normalized_car_position: 0.01,
        });

        expect(intelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: true,
            baseline_progress_percent: 100,
            baseline_lap: 1,
        });
    });

    it('extracts wrap-aware section telemetry after a baseline lap exists', () => {
        const intelligence = new SessionIntelligence();
        intelligence.startBaselineCollectionAtLapStart();
        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.98,
        });
        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.03,
        });
        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 1,
            Graphics_normalized_car_position: 0.1,
        });

        expect(intelligence.hasCompletedBaselineLap()).toBe(true);
        expect(intelligence.getBaselineProgressPercent()).toBe(100);
        expect(intelligence.getSectionTelemetryWindow({
            section_name: 'T1 Paddock Hill Bend',
            lap: 0,
        })).toMatchObject({
            status: 'ready',
            rows: expect.any(Array),
            startSampleIdx: 0,
            endSampleIdx: 1,
        });
    });

    it('does not complete a new baseline from laps driven before collection started', () => {
        const intelligence = new SessionIntelligence();
        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 0,
            Graphics_normalized_car_position: 0.98,
        });
        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 1,
            Graphics_normalized_car_position: 0.45,
        });

        intelligence.startBaselineCollectionAtLapStart();

        expect(intelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: false,
            baseline_collection_started: false,
            baseline_progress_percent: 0,
            baseline_lap: null,
        });

        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 2,
            Graphics_normalized_car_position: 0.01,
        });

        expect(intelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: false,
            baseline_collection_started: true,
            baseline_progress_percent: 1,
            baseline_lap: 2,
        });
    });
});
