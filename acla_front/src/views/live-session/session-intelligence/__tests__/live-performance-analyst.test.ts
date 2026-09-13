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
