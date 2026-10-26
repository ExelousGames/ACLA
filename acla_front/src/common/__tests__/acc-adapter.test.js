/** @jest-environment node */
const { AccAdapter } = require('../../../electron/recording/readers/acc/acc-adapter');

const capture = (sharedMemory, broadcasting = {}) => ({ type: 'sample', sharedMemory, broadcasting });

describe('ACC telemetry adapter', () => {
    const adapter = new AccAdapter();

    it('flattens SDK objects, swaps vector axes and cleans strings without changing the capture', () => {
        const coordinates = Array.from({ length: 60 }, (_, index) => ({ x: index, y: index + 1, z: -index - 1 }));
        const carIds = Array.from({ length: 60 }, (_, index) => index);
        const packet = capture({
            Physics: {
                speed_kmh: 120.5, gear: 0, pit_limiter_on: false,
                velocity: { x: -1, y: 2, z: 3 },
                wheel_pressure: { front_left: 26.5, front_right: 26.6, rear_left: 26.7, rear_right: 26.8 },
                tyre_contact_point: { front_left: { x: 1, y: 2, z: 3 } },
                _internal: 10,
            },
            Graphics: {
                status: 2, session_type: -1, rain_tyres: 0, last_sector_time_str: 12345,
                car_coordinates: coordinates, car_id: carIds,
                current_time_str: '1:23.456\0', last_time_str: '\0  ',
            },
            Static: { track: 'monza\0', player_name: 'André\0', player_nick: null, player_surname: '', wet_tyres_name: '  ' },
        }, { positions: { 0: 0, 1052: 1 }, sessionTime: 12125 });
        const original = JSON.parse(JSON.stringify(packet));

        expect(adapter.adapt(packet, 1250)).toEqual({
            Physics_speed_kmh: 120.5, Physics_gear: 0, Physics_pit_limiter_on: false,
            Physics_velocity_x: -1, Physics_velocity_y: 3, Physics_velocity_z: 2,
            Physics_wheel_pressure_front_left: 26.5, Physics_wheel_pressure_front_right: 26.6,
            Physics_wheel_pressure_rear_left: 26.7, Physics_wheel_pressure_rear_right: 26.8,
            Physics_tyre_contact_point_front_left_x: 1,
            Physics_tyre_contact_point_front_left_y: 3,
            Physics_tyre_contact_point_front_left_z: 2,
            Graphics_status: 2, Graphics_session_type: -1, Graphics_rain_tyres: 0,
            Graphics_last_sector_time_str: 12345,
            Graphics_car_coordinates: Array.from({ length: 60 }, (_, index) => ({ x: index, y: -index - 1, z: index + 1 })),
            Graphics_car_id: carIds,
            Graphics_current_time_str: '1:23.456', Static_track: 'monza', Static_player_name: 'Andr',
            Graphics_normalized_positions: { 0: 0, 1052: 1 }, Graphics_game_time: 12.125,
            Graphics_analyst_session_time: 1.25,
        });
        expect(packet).toEqual(original);
    });

    it.each(['velocity', 'local_velocity', 'g_force', 'local_angular_vel'])(
        'swaps Y/Z in %s without negating components or changing orientation angles', (field) => {
            const vector = Object.freeze({ x: -4, y: 2, z: -9 });
            const packet = capture({ Physics: { [field]: vector, heading: 0.5, pitch: -0.2, roll: 0.1 } });
            const expected = {
                [`Physics_${field}_x`]: -4, [`Physics_${field}_y`]: -9, [`Physics_${field}_z`]: 2,
                Physics_heading: 0.5, Physics_pitch: -0.2, Physics_roll: 0.1,
            };
            expect(adapter.adapt(packet)).toEqual(expected);
            expect(adapter.adapt(packet)).toEqual(expected);
            expect(vector).toEqual({ x: -4, y: 2, z: -9 });
        },
    );

    it.each(['tyre_contact_point', 'tyre_contact_normal', 'tyre_contact_heading'])(
        'swaps Y/Z in %s for every wheel', (field) => {
            const wheels = {
                front_left: { x: 1, y: 2, z: 3 }, front_right: { x: 4, y: 5, z: 6 },
                rear_left: { x: 7, y: 8, z: 9 }, rear_right: { x: 10, y: 11, z: 12 },
            };
            expect(adapter.adapt(capture({ Physics: { [field]: wheels } }))).toEqual({
                [`Physics_${field}_front_left_x`]: 1, [`Physics_${field}_front_left_y`]: 3, [`Physics_${field}_front_left_z`]: 2,
                [`Physics_${field}_front_right_x`]: 4, [`Physics_${field}_front_right_y`]: 6, [`Physics_${field}_front_right_z`]: 5,
                [`Physics_${field}_rear_left_x`]: 7, [`Physics_${field}_rear_left_y`]: 9, [`Physics_${field}_rear_left_z`]: 8,
                [`Physics_${field}_rear_right_x`]: 10, [`Physics_${field}_rear_right_y`]: 12, [`Physics_${field}_rear_right_z`]: 11,
            });
        },
    );

    it.each([
        [{ y: -5 }, { Physics_velocity_z: -5 }],
        [{ z: 0 }, { Physics_velocity_y: 0 }],
        [{ x: 3, y: null, z: -2 }, { Physics_velocity_x: 3, Physics_velocity_y: -2 }],
    ])('preserves missing components when swapping a partial vector: %o', (velocity, expected) => {
        expect(adapter.adapt(capture({ Physics: { velocity } }))).toEqual(expected);
    });

    it.each([0, 1, 3])('omits Broadcasting fields outside live status (%s)', (status) => {
        expect(adapter.adapt(capture({ Graphics: { status } }, { positions: { 7: 0.5 }, sessionTime: 12125 })))
            .toEqual({ Graphics_status: status });
    });

    it('omits unavailable Broadcasting data and preserves an empty connected position map', () => {
        expect(adapter.adapt(capture({ Graphics: { status: 2 } }, { positions: null, sessionTime: 12125 })))
            .toEqual({ Graphics_status: 2 });
        expect(adapter.adapt(capture({ Graphics: { status: 2 } }, { positions: {}, sessionTime: 0 }), 0))
            .toEqual({ Graphics_status: 2, Graphics_normalized_positions: {},
                Graphics_game_time: 0, Graphics_analyst_session_time: 0 });
    });

    it.each([undefined, null, -1, NaN, Infinity, '12125'])('omits unavailable native session time (%s)', (sessionTime) => {
        expect(adapter.adapt(capture({ Graphics: { status: 2 } }, { positions: {}, sessionTime })))
            .toEqual({ Graphics_status: 2, Graphics_normalized_positions: {} });
    });

    it.each([
        null, {}, { available: false }, { Physics_speed_kmh: 120 },
        { type: 'sample', sharedMemory: {}, broadcasting: null },
        capture({}), capture({ Static: { track: '\0 ' }, Graphics: { car_id: [] } }),
        capture({ Physics: { speedKph: 120 } }), capture({ Physics: { gear: 1.5 } }),
        capture({ Physics: { pit_limiter_on: 1 } }), capture({ Graphics: { car_id: [1, 2] } }),
        capture({ Graphics: { car_coordinates: Array(60).fill({ x: 1, y: 2, z: 3, extra: 4 }) } }),
        capture({ Graphics: { car_coordinates: Array(60).fill({ x: 1, y: 2 }) } }),
        capture({ Graphics: { status: 2 } }, { positions: { 7: 1.5 } }),
    ])('rejects malformed captures before they reach downstream consumers: %o', (packet) => {
        expect(() => adapter.adapt(packet, 1250)).toThrow(TypeError);
    });
});
