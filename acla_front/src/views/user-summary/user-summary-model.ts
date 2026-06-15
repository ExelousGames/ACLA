export type ChildSegmentView = {
    id: string;
    name: string;
    count: number;
    kind: 'strength' | 'needs_work' | 'recovery' | 'info';
    parentLabelName?: string | null;
};

export type ParentSegmentView = {
    id: string;
    name: string;
    expertLevelTurns: number;
    mistakes: number;
    practiceMistakes: number;
    racingMistakes: number;
    childSegments: ChildSegmentView[];
};

export type TrackHighlightView = {
    parentSegmentId: string;
    parentSegmentName: string;
    childSegmentId: string;
    childSegmentName: string;
    count: number;
    kind: ChildSegmentView['kind'];
};

export type TrackSummaryView = {
    id: string;
    name: string;
    sessionsAnalyzed: number;
    sessionsSkipped: number;
    sessionsFailed: number;
    totalTelemetryRows: number;
    cars: Record<string, number>;
    parentSegments: ParentSegmentView[];
    strengths: TrackHighlightView[];
    improvementAreas: TrackHighlightView[];
};

export const asRecord = (value: unknown): Record<string, any> => (
    value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, any> : {}
);

const asArray = (value: unknown): any[] => (
    Array.isArray(value) ? value : []
);

const toNumber = (value: unknown): number => {
    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue : 0;
};

const labelKind = (labelId: string): ChildSegmentView['kind'] => {
    if (labelId === 'EA' || labelId.startsWith('O') || labelId.startsWith('OD')) return 'strength';
    if (labelId.startsWith('MSP') || labelId.startsWith('MSR')) return 'needs_work';
    if (labelId.startsWith('RM')) return 'recovery';
    return 'info';
};

export const formatNumber = (value: number): string => (
    new Intl.NumberFormat().format(value)
);

export const formatCount = (value: number): string => (
    `${formatNumber(value)} ${value === 1 ? 'time' : 'times'}`
);

const normalizeChildSegment = (child: Record<string, any>, fallbackId: string): ChildSegmentView => {
    const id = String(child.childSegmentId || child.labelId || child.label_id || fallbackId);
    return {
        id,
        name: String(child.childSegmentName || child.labelName || child.label_name || id),
        count: toNumber(child.count || 1),
        kind: child.kind || labelKind(id),
        parentLabelName: child.parentLabelName || child.parent_label_name || null,
    };
};

const buildLegacyChildSegments = (section: Record<string, any>): ChildSegmentView[] => (
    Object.entries(asRecord(section.labelCounts))
        .map(([labelId, count]) => normalizeChildSegment({ labelId, count }, labelId))
        .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
);

const normalizeParentSegment = (parent: Record<string, any>, fallbackId: string): ParentSegmentView => {
    const id = String(parent.parentSegmentId || parent.parent_segment_id || parent.sectionId || fallbackId);
    const childSource = asArray(parent.childSegments || parent.child_segments || parent.sub_segments);
    const childSegments = childSource.length > 0
        ? childSource.map((child, index) => normalizeChildSegment(asRecord(child), `${id}-${index}`))
        : buildLegacyChildSegments(parent);

    return {
        id,
        name: String(parent.parentSegmentName || parent.parent_segment_name || parent.sectionName || id),
        expertLevelTurns: toNumber(parent.expertLevelTurns),
        mistakes: toNumber(parent.mistakes),
        practiceMistakes: toNumber(parent.practiceMistakes),
        racingMistakes: toNumber(parent.racingMistakes),
        childSegments,
    };
};

const buildParentSegments = (track: Record<string, any>): ParentSegmentView[] => {
    const parentSegments = asArray(track.parentSegments || track.parent_segments);
    if (parentSegments.length > 0) {
        return parentSegments.map((parent, index) => normalizeParentSegment(asRecord(parent), String(index)));
    }

    return Object.entries(asRecord(track.sections))
        .map(([sectionId, section]) => normalizeParentSegment(asRecord(section), sectionId))
        .filter((parent) => parent.childSegments.length > 0);
};

const buildHighlights = (
    parentSegments: ParentSegmentView[],
    kinds: ChildSegmentView['kind'][],
    limit = 5,
): TrackHighlightView[] => {
    const kindSet = new Set(kinds);
    return parentSegments
        .flatMap((parentSegment) => parentSegment.childSegments
            .filter((childSegment) => kindSet.has(childSegment.kind))
            .map((childSegment) => ({
                parentSegmentId: parentSegment.id,
                parentSegmentName: parentSegment.name,
                childSegmentId: childSegment.id,
                childSegmentName: childSegment.name,
                count: childSegment.count,
                kind: childSegment.kind,
            })))
        .sort((a, b) => b.count - a.count || a.parentSegmentName.localeCompare(b.parentSegmentName))
        .slice(0, limit);
};

const normalizeHighlights = (
    source: unknown,
    fallback: TrackHighlightView[],
): TrackHighlightView[] => {
    const highlights = asArray(source);
    if (highlights.length === 0) return fallback;

    return highlights.map((highlight, index) => {
        const item = asRecord(highlight);
        const childId = String(item.childSegmentId || item.labelId || index);
        return {
            parentSegmentId: String(item.parentSegmentId || item.parent_segment_id || ''),
            parentSegmentName: String(item.parentSegmentName || item.parent_segment_name || 'Track area'),
            childSegmentId: childId,
            childSegmentName: String(item.childSegmentName || item.labelName || childId),
            count: toNumber(item.count),
            kind: item.kind || labelKind(childId),
        };
    });
};

export const buildTrackSummaryViews = (summary: Record<string, any>): TrackSummaryView[] => {
    const root = asRecord(summary.sessionAnalysis || summary);
    const tracks = asRecord(root.tracks);

    return Object.entries(tracks).map(([trackId, rawTrack]) => {
        const track = asRecord(rawTrack);
        const parentSegments = buildParentSegments(track);
        return {
            id: trackId,
            name: String(track.trackName || track.name || trackId),
            sessionsAnalyzed: toNumber(track.sessionsAnalyzed),
            sessionsSkipped: toNumber(track.sessionsSkipped),
            sessionsFailed: toNumber(track.sessionsFailed),
            totalTelemetryRows: toNumber(track.totalTelemetryRows),
            cars: asRecord(track.cars),
            parentSegments,
            strengths: normalizeHighlights(track.strengths, buildHighlights(parentSegments, ['strength'])),
            improvementAreas: normalizeHighlights(
                track.improvementAreas || track.improvement_areas,
                buildHighlights(parentSegments, ['needs_work', 'recovery']),
            ),
        };
    });
};
