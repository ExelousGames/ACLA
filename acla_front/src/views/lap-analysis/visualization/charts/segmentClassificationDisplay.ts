export type SegmentLabelResolver = (labelId: string) => string | undefined;

export type SegmentTimeGap = {
    start_ms: number;
    end_ms: number;
    delta_ms: number;
};

/** A flat label interval in original telemetry indices; end_index is exclusive. */
export type SegmentClassificationLabel = {
    label_name: string;
    start_index: number;
    end_index: number;
};

export type SegmentClassificationSegment = {
    id?: string;
    labels: SegmentClassificationLabel[];
    track_section?: string;
    start_index: number;
    end_index: number;
    time_gap?: SegmentTimeGap;
};

export const normalizeSegmentLabels = (value: unknown): SegmentClassificationLabel[] => {
    if (!Array.isArray(value)) return [];
    return value.flatMap((label): SegmentClassificationLabel[] => {
        if (!label || typeof label !== 'object'
            || typeof label.label_name !== 'string' || !label.label_name.trim()
            || !Number.isInteger(label.start_index) || label.start_index < 0
            || !Number.isInteger(label.end_index) || label.end_index <= label.start_index) {
            return [];
        }
        return [{
            label_name: label.label_name.trim(),
            start_index: label.start_index,
            end_index: label.end_index,
        }];
    });
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
    Array.isArray(segment.labels) ? dedupeTexts(segment.labels.map((label) => label.label_name)) : []
);

export const resolveSegmentLabelTexts = (
    segment: SegmentClassificationSegment,
    resolveLabel?: SegmentLabelResolver,
): string[] => (
    getSegmentLabelIds(segment).map((labelId) => getSegmentLabelText(labelId, resolveLabel))
);

export const resolveActiveSegmentLabelTexts = (
    segment: SegmentClassificationSegment,
    sourceIndex: number,
    resolveLabel?: SegmentLabelResolver,
): string[] => resolveSegmentLabelTexts({
    ...segment,
    labels: segment.labels.filter((label) => (
        sourceIndex >= label.start_index && sourceIndex < label.end_index
    )),
}, resolveLabel);
