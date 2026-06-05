export type SegmentClassificationLabel = {
    label_id: string;
    label_name: string;
};

export type SegmentClassificationSubSegment = {
    start_index: number;
    end_index: number;
    labels: SegmentClassificationLabel[];
};

export type SegmentClassificationSegment = {
    id?: string;
    labels?: string[];
    main_label_id?: string;
    main_label_name?: string;
    start_index: number;
    end_index: number;
    sub_labels?: SegmentClassificationLabel[];
    sub_segments?: SegmentClassificationSubSegment[];
};

export const getSegmentLabelText = (label: SegmentClassificationLabel): string => (
    label.label_name || label.label_id
);

export const getSegmentMainLabelText = (segment: SegmentClassificationSegment): string => (
    segment.main_label_name
    || segment.main_label_id
    || segment.labels?.join(', ')
    || 'Unlabeled'
);

export const getSegmentSubLabelTexts = (segment: SegmentClassificationSegment): string[] => {
    if (Array.isArray(segment.sub_labels) && segment.sub_labels.length > 0) {
        return segment.sub_labels.map(getSegmentLabelText);
    }

    if (segment.main_label_id && Array.isArray(segment.labels)) {
        return segment.labels.filter((label) => label !== segment.main_label_id);
    }

    return [];
};

export const getActiveSubLabelTexts = (segment: SegmentClassificationSegment, sourceIndex: number): string[] => {
    const activeSubSegment = segment.sub_segments?.find((subSegment) => (
        sourceIndex >= subSegment.start_index && sourceIndex < subSegment.end_index
    ));

    if (activeSubSegment) {
        return activeSubSegment.labels.map(getSegmentLabelText);
    }

    return getSegmentSubLabelTexts(segment);
};
