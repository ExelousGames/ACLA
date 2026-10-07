import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { findTriggeredTrackGuideCorners, getTrackGuideCorners } from '../track-guide-corners';

const map: CircuitMapDto = {
    id: 'map', game: 'acc', circuit_name: 'Custom circuit', resolution: 1000,
    samples: {
        middle_line: [0, 0.5, 0.99].map((position) => ({
            bin: position * 1000, normalized_position: position,
            x: position * 100, y: 0, z: 0, sample_count: 1, updated_at: 'now',
        })),
    },
    centerline_tags: [
        { id: 'fast', label: 'fast corner', start_position: 0.6, end_position: 0.7 },
        { id: 'straight', label: 'long straight', start_position: 0.3, end_position: 0.5 },
        { id: 'slow', label: 'slow corner', start_position: 0.1, end_position: 0.2 },
        { id: 'wrap', label: 'slow corner', start_position: 0.97, end_position: 0.04 },
    ],
};

describe('middleline track guide corners', () => {
    it('numbers a segment once when it contains several corner and speed tags', () => {
        expect(getTrackGuideCorners({ ...map, centerline_segments: [
            { id: 'turn', tags: ['slow', 'corner', 'slow corner'], start_position: 0.1, end_position: 0.2 },
            { id: 'speed', tags: ['fast'], start_position: 0.3, end_position: 0.4 },
            { id: 'wrap', tags: ['corner', 'fast'], start_position: 0.95, end_position: 0.05 },
        ] })).toEqual([
            { id: 'turn', number: 1, type: 'corner', from: 0.1, to: 0.2 },
            { id: 'wrap', number: 2, type: 'corner', from: 0.95, to: 0.05 },
        ]);
        expect(getTrackGuideCorners({ ...map, centerline_segments: [] })).toEqual([]);
    });

    it('numbers legacy slow and fast corner tags together without changing the saved map', () => {
        expect(getTrackGuideCorners(map)).toEqual([
            { id: 'slow', number: 1, type: 'slow corner', from: 0.1, to: 0.2 },
            { id: 'fast', number: 2, type: 'fast corner', from: 0.6, to: 0.7 },
            { id: 'wrap', number: 3, type: 'slow corner', from: 0.97, to: 0.04 },
        ]);
        expect(map.centerline_tags?.map((tag) => tag.id)).toEqual(['fast', 'straight', 'slow', 'wrap']);
    });

    it('numbers corner tags while ignoring separate speed tags, including overlapping ranges', () => {
        expect(getTrackGuideCorners({ ...map, centerline_tags: [
            { id: 'speed-only', label: 'fast', start_position: 0.05, end_position: 0.1 },
            { id: 'wrap', label: 'corner', start_position: 0.97, end_position: 0.04 },
            { id: 'slow', label: 'slow', start_position: 0.1, end_position: 0.2 },
            { id: 'corner', label: 'corner', start_position: 0.1, end_position: 0.2 },
            { id: 'fast', label: 'fast', start_position: 0.97, end_position: 0.04 },
            map.centerline_tags![0],
        ] })).toEqual([
            { id: 'corner', number: 1, type: 'corner', from: 0.1, to: 0.2 },
            { id: 'fast', number: 2, type: 'fast corner', from: 0.6, to: 0.7 },
            { id: 'wrap', number: 3, type: 'corner', from: 0.97, to: 0.04 },
        ]);
    });

    it('requires a middleline and valid corner tags instead of falling back to known track positions', () => {
        expect(getTrackGuideCorners(null)).toEqual([]);
        expect(getTrackGuideCorners({ ...map, source_track_key: 'monza', samples: {} })).toEqual([]);
        expect(getTrackGuideCorners({ ...map, centerline_tags: [] })).toEqual([]);
        expect(getTrackGuideCorners({ ...map, centerline_tags: [
            { id: 'invalid', label: 'slow corner', start_position: NaN, end_position: 0.2 },
            { id: 'outside', label: 'fast corner', start_position: 0.5, end_position: 1.1 },
            { id: 'empty', label: 'slow corner', start_position: 0.4, end_position: 0.4 },
            map.centerline_tags![0],
        ] })).toEqual([{ id: 'fast', number: 1, type: 'fast corner', from: 0.6, to: 0.7 }]);
    });

    it('triggers at saved tag entries, in driving order across start/finish', () => {
        const corners = getTrackGuideCorners(map);
        expect(findTriggeredTrackGuideCorners(corners, 0.05, 0.1).map((corner) => corner.number)).toEqual([1]);
        expect(findTriggeredTrackGuideCorners(corners, 0.1, 0.12)).toEqual([]);
        expect(findTriggeredTrackGuideCorners(corners, 0.96, 0.11).map((corner) => corner.number)).toEqual([3, 1]);
        expect(findTriggeredTrackGuideCorners(corners, 0.5, 0.48)).toEqual([]);
        expect(findTriggeredTrackGuideCorners(corners, 0.1, NaN)).toEqual([]);
        expect(findTriggeredTrackGuideCorners(corners, -0.1, 0.2)).toEqual([]);
    });
});
