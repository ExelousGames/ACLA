import type { CircuitMapCenterlineSegment, CircuitMapDto } from './circuit-map-types';

const isCorner = (label: string) => ['corner', 'slow corner', 'fast corner'].includes(label.trim().toLowerCase());

export const getCircuitMapCenterlineSegments = (
    map: Pick<CircuitMapDto, 'centerline_segments' | 'centerline_tags'> | null | undefined,
): CircuitMapCenterlineSegment[] => {
    // An explicitly empty segment list must not resurrect removed legacy tags.
    if (map && Array.isArray(map.centerline_segments)) return map.centerline_segments;
    const segments: CircuitMapCenterlineSegment[] = [];
    for (const tag of map?.centerline_tags ?? []) {
        const existing = segments.find((segment) => segment.start_position === tag.start_position
            && segment.end_position === tag.end_position);
        if (existing) {
            // Preserve the corner identity used by track guides and live coaching.
            if (isCorner(tag.label) && !existing.tags.some(isCorner)) existing.id = tag.id;
            if (!existing.tags.includes(tag.label)) existing.tags.push(tag.label);
        } else {
            segments.push({ id: tag.id, tags: [tag.label], start_position: tag.start_position, end_position: tag.end_position });
        }
    }
    return segments;
};
