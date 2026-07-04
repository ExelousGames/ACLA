export type SegmentLabelResolver = (labelId: string) => string | undefined;

export type SegmentTimeGap = {
    start_ms: number;
    end_ms: number;
    delta_ms: number;
};

export type SegmentClassificationSegment = {
    id?: string;
    labels: string[];
    track_section?: string;
    start_index: number;
    end_index: number;
    time_gap?: SegmentTimeGap;
};

export const getSegmentLabelText = (labelId: string, resolveLabel?: SegmentLabelResolver): string => (
    resolveLabel?.(labelId) || labelId
);

const dedupeTexts = (texts: string[]): string[] => {
    const seen = new Set<string>();
    return texts.filter((text) => {
        if (seen.has(text)) return false;
        seen.add(text);
        return true;
    });
};

export const getSegmentTrackSectionText = (
    segment: SegmentClassificationSegment,
    resolveLabel?: SegmentLabelResolver,
): string => {
    if (segment.track_section) {
        return getSegmentLabelText(segment.track_section, resolveLabel);
    }

    return 'Unknown section';
};

export const getSegmentLabelIds = (segment: SegmentClassificationSegment): string[] => (
    Array.isArray(segment.labels) ? dedupeTexts(segment.labels) : []
);

export const resolveSegmentLabelTexts = (
    segment: SegmentClassificationSegment,
    resolveLabel?: SegmentLabelResolver,
): string[] => (
    getSegmentLabelIds(segment).map((labelId) => getSegmentLabelText(labelId, resolveLabel))
);

export const resolveActiveSegmentLabelTexts = (
    segment: SegmentClassificationSegment,
    _sourceIndex: number,
    resolveLabel?: SegmentLabelResolver,
): string[] => resolveSegmentLabelTexts(segment, resolveLabel);
