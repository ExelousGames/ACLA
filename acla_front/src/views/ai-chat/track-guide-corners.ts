import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getLiveMapMiddleLine } from 'views/live-session/live-map-data';

export interface TrackGuideCorner {
    id: string;
    number: number;
    type: 'corner' | 'slow corner' | 'fast corner';
    from: number;
    to: number;
}

export const isTrackGuidePosition = (value: unknown): value is number => (
    typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
);

export const getTrackGuideCorners = (map: CircuitMapDto | null): TrackGuideCorner[] => {
    if (getLiveMapMiddleLine(map).length < 2) return [];

    // Corner tags, including legacy combined labels, share one sequence by entry.
    // A range spanning start/finish keeps its entry near the end of the lap.
    return (map?.centerline_tags ?? [])
        .filter((tag) => (tag.label === 'corner' || tag.label === 'slow corner' || tag.label === 'fast corner')
            && isTrackGuidePosition(tag.start_position)
            && isTrackGuidePosition(tag.end_position)
            && tag.start_position !== tag.end_position)
        .sort((left, right) => left.start_position - right.start_position
            || left.end_position - right.end_position || left.id.localeCompare(right.id))
        .map((tag, index) => ({
            id: tag.id,
            number: index + 1,
            type: tag.label as TrackGuideCorner['type'],
            from: tag.start_position,
            to: tag.end_position,
        }));
};

export const findTriggeredTrackGuideCorners = (
    corners: TrackGuideCorner[],
    lastPos: number,
    currentPos: number,
): TrackGuideCorner[] => {
    if (!isTrackGuidePosition(lastPos) || !isTrackGuidePosition(currentPos)) return [];
    // Small backwards movements are not a lap crossing.
    if (currentPos < lastPos && lastPos - currentPos < 0.5) return [];

    return corners.filter((corner) => currentPos >= lastPos
        ? lastPos < corner.from && currentPos >= corner.from
        : lastPos < corner.from || currentPos >= corner.from)
        .sort((left, right) => ((left.from - lastPos + 1) % 1) - ((right.from - lastPos + 1) % 1));
};
