import { detectOvertakeTacticalState } from '../overtake-agent-detector';

const row = (
    timeSeconds: number,
    playerX: number,
    opponentX: number,
    opponentY = 0,
) => ({
    Graphics_current_time: timeSeconds * 1000,
    Graphics_player_car_id: 10,
    Graphics_car_id: [10, 20],
    Graphics_car_coordinates: [
        { x: playerX, y: 0, z: 0 },
        { x: opponentX, y: opponentY, z: 0 },
    ],
    Graphics_normalized_car_position: 0.5 + timeSeconds * 0.01,
    Static_track: 'brands_hatch',
});

describe('detectOvertakeTacticalState', () => {
    it('detects attack from coordinate-derived closing motion ahead', () => {
        const result = detectOvertakeTacticalState([
            row(0, 100, 150),
            row(1, 110, 155),
            row(2, 120, 160),
            row(3, 130, 165),
            row(4, 140, 170),
        ]);

        expect(result.status).toBe('actionable');
        if (result.status === 'actionable') {
            expect(result.event).toBe('attack_window');
            expect(result.mode).toBe('attack');
            expect(result.time_to_overlap_seconds).toBe(6);
        }
    });

    it('stays neutral when the car ahead is pulling away', () => {
        const result = detectOvertakeTacticalState([
            row(0, 100, 130),
            row(1, 105, 140),
            row(2, 110, 150),
            row(3, 115, 160),
        ]);

        expect(result.status).toBe('neutral');
    });

    it('detects defense from a closing car behind', () => {
        const result = detectOvertakeTacticalState([
            row(0, 100, 70),
            row(1, 110, 85),
            row(2, 120, 100),
            row(3, 130, 115),
            row(4, 140, 130),
        ]);

        expect(result.status).toBe('actionable');
        if (result.status === 'actionable') {
            expect(result.event).toBe('defense_threat');
            expect(result.mode).toBe('defense');
            expect(result.time_to_overlap_seconds).toBe(2);
        }
    });

    it('does not trigger for a lateral-only nearby car', () => {
        const result = detectOvertakeTacticalState([
            row(0, 100, 150, 25),
            row(1, 110, 155, 25),
            row(2, 120, 160, 25),
            row(3, 130, 165, 25),
            row(4, 140, 170, 25),
        ]);

        expect(result.status).toBe('neutral');
    });

    it('returns insufficient data when coordinates are unavailable', () => {
        const result = detectOvertakeTacticalState([
            { Graphics_current_time: 0, Graphics_player_car_id: 10 },
            { Graphics_current_time: 1000, Graphics_player_car_id: 10 },
        ]);

        expect(result.status).toBe('insufficient_data');
    });
});
