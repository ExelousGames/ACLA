import type { CircuitMapCenterlineSegment, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getCircuitMapCenterlineSegments } from 'views/circuit-maps/centerline-segments';
import { getLiveMapMiddleLine } from '../live-map-data';
import { getPhraseCornerGeometry, lapDistance as distance, segmentSpan, PhraseCornerGeometry } from './phrase-corner-geometry';

export interface PhraseMapContext {
    sectionId?: string;
    phase?: 'straight' | 'entry' | 'middle' | 'exit';
    cornerSpeed?: 'slow' | 'fast';
    linkedOpposite?: 0 | 1;
    linkedSameDirection?: 0 | 1;
    cornerShape?: PhraseCornerGeometry['shape'];
    cornerGeometry?: PhraseCornerGeometry;
    sequenceId?: string;
    sequenceShape?: 'alternating' | 'same-direction' | 'mixed';
    sequenceCornerCount?: number;
    sequenceCornerIndex?: number;
    sequenceRemaining?: number;
}

const normalized = (value: unknown): value is number => typeof value === 'number'
    && Number.isFinite(value) && value >= 0 && value <= 1;
const contains = (tag: CircuitMapCenterlineSegment, position: number) =>
    distance(tag.start_position, position) <= segmentSpan(tag) + 1e-9;
const labels = (segment: CircuitMapCenterlineSegment) => segment.tags.map((tag) => tag.trim().toLowerCase());
const isCorner = (segment: CircuitMapCenterlineSegment) => labels(segment).some((label) => ['corner', 'slow corner', 'fast corner'].includes(label));

/** Prepared once per map; telemetry only resolves the current tagged section. */
export function createPhraseMapContext(map: CircuitMapDto | null) {
    const line = getLiveMapMiddleLine(map);
    const tags = getCircuitMapCenterlineSegments(map).filter((tag) => normalized(tag.start_position)
        && normalized(tag.end_position) && segmentSpan(tag) > 0);
    const corners = tags.filter((tag) => isCorner(tag) && !labels(tag).includes('consecutive corners'))
        .sort((a, b) => a.start_position - b.start_position)
        .filter((tag, index, all) => !all.slice(0, index).some((other) =>
            other.start_position === tag.start_position && other.end_position === tag.end_position));
    const trackLength = line.reduce((total, point, index) => {
        const next = line[(index + 1) % line.length];
        return total + Math.hypot(next.x - point.x, next.y - point.y, next.z - point.z);
    }, 0);
    // Bound lookahead to 150 m and at most 3% of a lap on sparse maps.
    const lookahead = Math.min(0.03, trackLength > 0 ? 150 / trackLength : 0);
    const geometry = new Map(corners.map((corner) => [corner, getPhraseCornerGeometry(line, corner)]));
    const sequences = tags.filter((tag) => labels(tag).includes('consecutive corners')).map((area) => {
        const members = corners.filter((corner) =>
            distance(area.start_position, corner.start_position) + segmentSpan(corner) <= segmentSpan(area) + 1e-9)
            .sort((a, b) => distance(area.start_position, a.start_position) - distance(area.start_position, b.start_position));
        const signs = members.map((corner) => geometry.get(corner)?.turnSign);
        let shape: PhraseMapContext['sequenceShape'];
        if (members.length >= 2 && signs.every((sign) => sign !== undefined)) {
            const changes = signs.slice(1).filter((sign, index) => sign !== signs[index]).length;
            shape = changes === 0 ? 'same-direction' : changes === members.length - 1 ? 'alternating' : 'mixed';
        }
        return { area, members, shape };
    }).sort((a, b) => segmentSpan(a.area) - segmentSpan(b.area) || a.area.id.localeCompare(b.area.id));
    return {
        ready: line.length >= 2 && tags.length > 0,
        at(position: unknown): PhraseMapContext {
            if (line.length < 2 || !normalized(position)) return {};
            // Resolve overlapping corners by the smallest range, then the latest entry.
            const current = corners.filter((tag) => contains(tag, position))
                .sort((a, b) => segmentSpan(a) - segmentSpan(b)
                    || distance(a.start_position, position) - distance(b.start_position, position))[0];
            const activeSequence = sequences.find(({ area }) => contains(area, position));
            const next = corners.slice().sort((a, b) => distance(position, a.start_position) - distance(position, b.start_position))[0];
            const sequenceNext = activeSequence?.members.find((member) =>
                distance(activeSequence.area.start_position, member.start_position) > distance(activeSequence.area.start_position, position));
            const approaching = activeSequence ? sequenceNext : next;
            const corner = current ?? (approaching && distance(position, approaching.start_position) <= lookahead ? approaching : undefined);
            if (!corner) {
                if (activeSequence) return {
                    sectionId: activeSequence.area.id,
                    sequenceId: activeSequence.area.id,
                    sequenceShape: activeSequence.shape,
                    sequenceCornerCount: activeSequence.members.length,
                    sequenceRemaining: activeSequence.members.filter((member) =>
                        distance(activeSequence.area.start_position, member.start_position) > distance(activeSequence.area.start_position, position)).length,
                };
                const straight = tags.find((tag) => labels(tag).some((label) => ['straight', 'long straight'].includes(label)) && contains(tag, position));
                return straight ? { sectionId: straight.id, phase: 'straight' } : {};
            }
            const span = segmentSpan(corner);
            const progress = current ? distance(corner.start_position, position) / span : 0;
            const speedTags = tags.filter((tag) => {
                // Separate slow/fast tags may cover all or part of this corner.
                const midpoint = (corner.start_position + span / 2) % 1;
                return contains(tag, midpoint);
            }).flatMap(labels);
            const slow = speedTags.some((label) => label === 'slow' || label === 'slow corner');
            const fast = speedTags.some((label) => label === 'fast' || label === 'fast corner');
            const sequence = sequences.find(({ members }) => members.includes(corner));
            const sequenceIndex = sequence?.members.indexOf(corner) ?? -1;
            const following = sequence ? sequence.members[sequenceIndex + 1]
                : corners[(corners.indexOf(corner) + 1) % corners.length];
            const turn = geometry.get(corner)?.turnSign;
            const followingTurn = following && geometry.get(following)?.turnSign;
            // Opposite turn signs are coordinate-system independent (ACC / iRacing).
            // An explicit sequence defines the link even when its internal gap exceeds lookahead.
            const linked = following && following !== corner && turn !== undefined && followingTurn !== undefined
                && (sequence || distance(corner.end_position, following.start_position) <= lookahead);
            return {
                sectionId: corner.id,
                phase: progress < 0.35 ? 'entry' : progress < 0.7 ? 'middle' : 'exit',
                cornerSpeed: slow !== fast ? slow ? 'slow' : 'fast' : undefined,
                cornerShape: geometry.get(corner)?.shape,
                cornerGeometry: geometry.get(corner),
                linkedOpposite: linked && turn !== followingTurn ? 1 : 0,
                linkedSameDirection: linked && turn === followingTurn ? 1 : 0,
                ...(sequence && {
                    sequenceId: sequence.area.id,
                    sequenceShape: sequence.shape,
                    sequenceCornerCount: sequence.members.length,
                    sequenceCornerIndex: sequenceIndex + 1,
                    sequenceRemaining: sequence.members.length - sequenceIndex - 1,
                }),
            };
        },
    };
}
