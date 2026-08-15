import { SessionIntelligence } from '../SessionIntelligence';
import {
    detectLiveSessionType,
} from '../live-performance-analyst';

describe('live performance analyst helpers', () => {
    it('detects live session type from ACC car-count fields', () => {
        expect(detectLiveSessionType({ Static_num_cars: 1 })).toBe('solo_practice');
        expect(detectLiveSessionType({ Graphics_active_cars_count: 1 })).toBe('solo_practice');
        expect(detectLiveSessionType({ Graphics_active_cars_count: 8 })).toBe('traffic_or_race');
        expect(detectLiveSessionType({})).toBe('unknown');
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
            Graphics_normalized_car_position: 0.45,
        });

        expect(intelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: false,
            baseline_collection_started: false,
            baseline_progress_percent: 0,
            baseline_lap: null,
        });

        intelligence.tick({
            Static_track: 'brands_hatch',
            Static_num_cars: 1,
            Graphics_completed_laps: 1,
            Graphics_normalized_car_position: 0.005,
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
            Graphics_normalized_car_position: 0.001,
        });

        expect(intelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: true,
            baseline_progress_percent: 100,
            baseline_lap: 1,
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
            Graphics_normalized_car_position: 0.005,
        });

        expect(intelligence.getLiveSessionSnapshot()).toMatchObject({
            baseline_ready: false,
            baseline_collection_started: true,
            baseline_progress_percent: 1,
            baseline_lap: 2,
        });
    });
});
