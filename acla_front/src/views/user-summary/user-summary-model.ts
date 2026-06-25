export type PracticeSegmentType = 'mistake' | 'expert_adherence' | 'recovery_merge';

export type PracticeChildSegmentView = {
    id: string;
    name: string;
    count: number;
    startIndex?: number;
    endIndex?: number;
};

export type PracticeParentSegmentView = {
    id: string;
    name: string;
    type: PracticeSegmentType;
    count: number;
    childSegments: PracticeChildSegmentView[];
};

export type PracticeSectionSummaryView = {
    id: string;
    name: string;
    analyzedTimeCount: number;
    mistakeCount: number;
    expertAdherenceCount: number;
    mistakePercent: number;
    expertAdherencePercent: number;
    mistakeSegments: PracticeParentSegmentView[];
    expertAdherenceSegments: PracticeParentSegmentView[];
    recoveryMergeSegments: PracticeParentSegmentView[];
};

export type PracticeTrackSummaryView = {
    id: string;
    name: string;
    analyzedSessionCount: number;
    skippedSessionCount: number;
    failedSessionCount: number;
    totalAnalyzedTimeCount: number;
    sections: PracticeSectionSummaryView[];
};

type LabelResolver = (labelId: string) => string | undefined;
type CategoryResolver = (category: string) => string[];

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

const normalizeOptionalIndex = (value: unknown): number | undefined => {
    const numericValue = toNumber(value);
    return Number.isFinite(Number(value)) ? numericValue : undefined;
};

const resolveLabelName = (labelId: string, resolveLabel?: LabelResolver): string => (
    resolveLabel?.(labelId) || labelId
);

const normalizeChildSegment = (
    child: Record<string, any>,
    fallbackId: string,
    resolveLabel?: LabelResolver,
): PracticeChildSegmentView => {
    const id = String(child.id || fallbackId);
    const startIndex = normalizeOptionalIndex(child.startIndex);
    const endIndex = normalizeOptionalIndex(child.endIndex);

    return {
        id,
        name: resolveLabelName(id, resolveLabel),
        count: toNumber(child.count),
        ...(startIndex !== undefined ? { startIndex } : {}),
        ...(endIndex !== undefined ? { endIndex } : {}),
    };
};

const normalizeSegmentType = (type: unknown): PracticeSegmentType => (
    type === 'expert_adherence' ? 'expert_adherence' : 'mistake'
);

const PRACTICE_MISTAKE_PARENT_IDS = ['MSP'];
const RECOVERY_MERGE_PARENT_IDS = ['RM'];
const EXPERT_ADHERENCE_PARENT_IDS = ['EA'];

const normalizeParentSegment = (
    parent: Record<string, any>,
    fallbackId: string,
    resolveLabel?: LabelResolver,
): PracticeParentSegmentView => {
    const id = String(parent.id || fallbackId);

    return {
        id,
        name: resolveLabelName(id, resolveLabel),
        type: normalizeSegmentType(parent.type),
        count: toNumber(parent.count),
        childSegments: asArray(parent.childSegments)
            .map((child, index) => normalizeChildSegment(asRecord(child), `${id}-${index}`, resolveLabel))
            .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name)),
    };
};

const calculatePercent = (count: number, total: number): number => (
    total > 0 ? (count / total) * 100 : 0
);

const normalizeSection = (
    sectionId: string,
    rawSection: unknown,
    resolveLabel?: LabelResolver,
): PracticeSectionSummaryView => {
    const section = asRecord(rawSection);
    const analyzedTimeCount = toNumber(section.analyzedTimeCount);
    const mistakeCount = toNumber(section.mistakeCount);
    const expertAdherenceCount = toNumber(section.expertAdherenceCount);
    const parentSegments = asArray(section.parentSegments)
        .map((parent, index) => normalizeParentSegment(asRecord(parent), `${sectionId}-${index}`, resolveLabel))
        .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));

    return {
        id: sectionId,
        name: resolveLabelName(sectionId, resolveLabel),
        analyzedTimeCount,
        mistakeCount,
        expertAdherenceCount,
        mistakePercent: calculatePercent(mistakeCount, analyzedTimeCount),
        expertAdherencePercent: calculatePercent(expertAdherenceCount, analyzedTimeCount),
        mistakeSegments: parentSegments.filter((segment) => segment.type === 'mistake'),
        expertAdherenceSegments: parentSegments.filter((segment) => segment.type === 'expert_adherence'),
        recoveryMergeSegments: parentSegments.filter((segment) => segment.type === 'recovery_merge'),
    };
};

const getOrderedSectionIds = (
    trackId: string,
    rawSections: Record<string, any>,
    resolveCategory?: CategoryResolver,
): string[] => {
    const aiSectionIds = resolveCategory?.(trackId) ?? [];
    if (aiSectionIds.length > 0) {
        return aiSectionIds;
    }

    return Object.keys(rawSections);
};

const normalizeCurrentAnalyzerChildSegments = (
    parentId: string,
    labelCounts: Record<string, any>,
    childIds: string[],
    resolveLabel?: LabelResolver,
): PracticeChildSegmentView[] => (
    childIds
        .filter((labelId) => toNumber(labelCounts[labelId]) > 0)
        .map((labelId) => ({
            id: labelId,
            name: resolveLabelName(labelId, resolveLabel),
            count: toNumber(labelCounts[labelId]),
        }))
        .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
);

const parentCountFromLabels = (
    parentId: string,
    labelCounts: Record<string, any>,
    childIds: string[],
): number => {
    const parentCount = toNumber(labelCounts[parentId]);
    if (parentCount > 0) return parentCount;

    return childIds.reduce((sum, childId) => sum + toNumber(labelCounts[childId]), 0);
};

const normalizeCurrentAnalyzerParentSegments = (
    parentIds: string[],
    type: PracticeSegmentType,
    labelCounts: Record<string, any>,
    resolveLabel?: LabelResolver,
    resolveCategory?: CategoryResolver,
): PracticeParentSegmentView[] => parentIds
    .map((parentId) => {
        const childIds = resolveCategory?.(parentId) ?? [];
        const count = parentCountFromLabels(parentId, labelCounts, childIds);
        if (count <= 0) return null;

        return {
            id: parentId,
            name: resolveLabelName(parentId, resolveLabel),
            type,
            count,
            childSegments: normalizeCurrentAnalyzerChildSegments(
                parentId,
                labelCounts,
                childIds,
                resolveLabel,
            ),
        };
    })
    .filter((segment): segment is PracticeParentSegmentView => Boolean(segment))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));

const normalizeCurrentAnalyzerSection = (
    sectionId: string,
    rawSection: unknown,
    resolveLabel?: LabelResolver,
    resolveCategory?: CategoryResolver,
): PracticeSectionSummaryView => {
    const section = asRecord(rawSection);
    const labelCounts = asRecord(section.labelCounts);
    const parentSegments = [
        ...normalizeCurrentAnalyzerParentSegments(
            PRACTICE_MISTAKE_PARENT_IDS,
            'mistake',
            labelCounts,
            resolveLabel,
            resolveCategory,
        ),
        ...normalizeCurrentAnalyzerParentSegments(
            EXPERT_ADHERENCE_PARENT_IDS,
            'expert_adherence',
            labelCounts,
            resolveLabel,
            resolveCategory,
        ),
        ...normalizeCurrentAnalyzerParentSegments(
            RECOVERY_MERGE_PARENT_IDS,
            'recovery_merge',
            labelCounts,
            resolveLabel,
            resolveCategory,
        ),
    ];
    const mistakeCount = toNumber(section.mistakeCount)
        || parentSegments
            .filter((segment) => segment.type === 'mistake')
            .reduce((sum, segment) => sum + segment.count, 0);
    const expertAdherenceCount = toNumber(section.expertAdherenceCount)
        || parentSegments
            .filter((segment) => segment.type === 'expert_adherence')
            .reduce((sum, segment) => sum + segment.count, 0);
    const derivedAnalyzedTimeCount = parentSegments.reduce((sum, segment) => sum + segment.count, 0);
    const explicitAnalyzedTimeCount = toNumber(
        section.analyzedTimeCount
        || section.totalAnalyzedTimeCount
        || section.analyzedSegmentCount
    );
    const analyzedTimeCount = Math.max(explicitAnalyzedTimeCount, derivedAnalyzedTimeCount);

    return {
        id: sectionId,
        name: resolveLabelName(sectionId, resolveLabel),
        analyzedTimeCount,
        mistakeCount,
        expertAdherenceCount,
        mistakePercent: calculatePercent(mistakeCount, analyzedTimeCount),
        expertAdherencePercent: calculatePercent(expertAdherenceCount, analyzedTimeCount),
        mistakeSegments: parentSegments.filter((segment) => segment.type === 'mistake'),
        expertAdherenceSegments: parentSegments.filter((segment) => segment.type === 'expert_adherence'),
        recoveryMergeSegments: parentSegments.filter((segment) => segment.type === 'recovery_merge'),
    };
};

const getPracticeTracks = (summary: Record<string, any>): Record<string, any> => {
    const sessionAnalysis = asRecord(summary.sessionAnalysis);
    const practiceTracks = asRecord(asRecord(sessionAnalysis.practice).tracks);
    if (Object.keys(practiceTracks).length > 0) {
        return practiceTracks;
    }

    const currentAnalyzerTracks = asRecord(sessionAnalysis.tracks);
    if (Object.keys(currentAnalyzerTracks).length > 0) {
        return currentAnalyzerTracks;
    }

    return asRecord(summary.tracks);
};

const isNewPracticeTrack = (track: Record<string, any>): boolean => (
    track.analyzedSessionCount !== undefined
    || track.totalAnalyzedTimeCount !== undefined
    || Object.values(asRecord(track.sections)).some((section) => (
        asRecord(section).analyzedTimeCount !== undefined
        || asRecord(section).parentSegments !== undefined
    ))
);

export const formatNumber = (value: number): string => (
    new Intl.NumberFormat().format(value)
);

export const formatCount = (value: number): string => (
    `${formatNumber(value)} ${value === 1 ? 'time' : 'times'}`
);

export const formatPercent = (value: number): string => (
    `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value)}%`
);

export const buildPracticeTrackSummaryViews = (
    summary: Record<string, any>,
    resolveLabel?: LabelResolver,
    resolveCategory?: CategoryResolver,
): PracticeTrackSummaryView[] => {
    const tracks = getPracticeTracks(summary);

    return Object.entries(tracks).map(([trackId, rawTrack]) => {
        const track = asRecord(rawTrack);
        const rawSections = asRecord(track.sections);
        const sectionIds = getOrderedSectionIds(trackId, rawSections, resolveCategory);
        const isNewTrack = isNewPracticeTrack(track);
        const sections = isNewTrack
            ? sectionIds.map((sectionId) => normalizeSection(sectionId, rawSections[sectionId], resolveLabel))
            : sectionIds.map((sectionId) => normalizeCurrentAnalyzerSection(
                sectionId,
                rawSections[sectionId],
                resolveLabel,
                resolveCategory,
            ));
        const totalAnalyzedTimeCount = toNumber(track.totalAnalyzedTimeCount)
            || sections.reduce((sum, section) => sum + section.analyzedTimeCount, 0);

        return {
            id: trackId,
            name: String(track.trackName || resolveLabelName(trackId, resolveLabel)),
            analyzedSessionCount: toNumber(track.analyzedSessionCount || track.sessionsAnalyzed),
            skippedSessionCount: toNumber(track.skippedSessionCount),
            failedSessionCount: toNumber(track.failedSessionCount),
            totalAnalyzedTimeCount,
            sections,
        };
    });
};
