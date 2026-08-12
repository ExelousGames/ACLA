import {
    PracticeParentSegmentView,
    PracticeSectionSummaryView,
    asRecord,
    buildPracticeTrackSummaryViews,
} from './user-summary-model';
import { QueryRequiredError, UserSummaryUnavailableError } from 'contexts/AiToolComponentError';
import { AI_TOOL_COMPONENT_NAMES } from 'contexts/AiToolComponentRefContext';

export interface UserSummaryAiSource {
    userSummary?: Record<string, any>;
    loading?: boolean;
    error?: string;
    getLabelName?: (labelId: string) => string | undefined;
    getCategoryLabels?: (category: string) => string[];
}

const summarizeSegments = (segments: PracticeParentSegmentView[]) => segments
    .filter((segment) => segment.count > 0)
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, 5)
    .map((segment) => ({
        id: segment.id,
        name: segment.name,
        count: segment.count,
        child_segments: segment.childSegments
            .filter((child) => child.count > 0)
            .slice(0, 5)
            .map((child) => ({
                id: child.id,
                name: child.name,
                count: child.count,
                ...(child.startIndex !== undefined ? { start_index: child.startIndex } : {}),
                ...(child.endIndex !== undefined ? { end_index: child.endIndex } : {}),
            })),
    }));

const summarizeSection = (section: PracticeSectionSummaryView) => ({
    id: section.id,
    name: section.name,
    analyzed_time_count: section.analyzedTimeCount,
    mistake_count: section.mistakeCount,
    mistake_percent: section.mistakePercent,
    expert_adherence_count: section.expertAdherenceCount,
    expert_adherence_percent: section.expertAdherencePercent,
    mistake_segments: summarizeSegments(section.mistakeSegments),
    expert_adherence_segments: summarizeSegments(section.expertAdherenceSegments),
    recovery_merge_segments: summarizeSegments(section.recoveryMergeSegments),
});

const normalizeSearchText = (value: unknown): string => String(value ?? '')
    .toLowerCase()
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const getSearchLimit = (value: unknown): number => {
    const parsed = Math.floor(Number(value));
    return Number.isFinite(parsed) && parsed > 0 ? Math.min(parsed, 10) : 5;
};

export const getUserSummaryMapLevel = (
    source: UserSummaryAiSource,
    args: Record<string, any>,
) => {
    if (source.loading) return { status: 'loading', maps: [] };
    if (source.error) {
        throw new UserSummaryUnavailableError(
            AI_TOOL_COMPONENT_NAMES.USER_SUMMARY,
            source.error,
        );
    }
    const summary = asRecord(source.userSummary);
    if (Object.keys(summary).length === 0) return { status: 'empty', maps: [] };

    const tracks = buildPracticeTrackSummaryViews(
        summary,
        source.getLabelName,
        source.getCategoryLabels,
    );
    const requestedMapId = typeof args.map_id === 'string' && args.map_id.trim()
        ? args.map_id.trim()
        : undefined;
    const filteredTracks = requestedMapId
        ? tracks.filter((track) => track.id === requestedMapId || track.name.toLowerCase() === requestedMapId.toLowerCase())
        : tracks;

    return {
        status: 'ready',
        map_count: filteredTracks.length,
        maps: filteredTracks.map((track) => {
            const mistakeCount = track.sections.reduce((sum, section) => sum + section.mistakeCount, 0);
            const expertAdherenceCount = track.sections.reduce((sum, section) => sum + section.expertAdherenceCount, 0);
            const totalAnalyzedTimeCount = track.totalAnalyzedTimeCount
                || track.sections.reduce((sum, section) => sum + section.analyzedTimeCount, 0);
            return {
                id: track.id,
                name: track.name,
                analyzed_session_count: track.analyzedSessionCount,
                skipped_session_count: track.skippedSessionCount,
                failed_session_count: track.failedSessionCount,
                total_analyzed_time_count: totalAnalyzedTimeCount,
                section_count: track.sections.length,
                mistake_count: mistakeCount,
                expert_adherence_count: expertAdherenceCount,
                mistake_percent: totalAnalyzedTimeCount > 0 ? (mistakeCount / totalAnalyzedTimeCount) * 100 : 0,
                expert_adherence_percent: totalAnalyzedTimeCount > 0 ? (expertAdherenceCount / totalAnalyzedTimeCount) * 100 : 0,
                top_mistake_sections: track.sections
                    .filter((section) => section.mistakeCount > 0)
                    .sort((a, b) => b.mistakeCount - a.mistakeCount || a.name.localeCompare(b.name))
                    .slice(0, 3)
                    .map(summarizeSection),
                top_expert_adherence_sections: track.sections
                    .filter((section) => section.expertAdherenceCount > 0)
                    .sort((a, b) => b.expertAdherenceCount - a.expertAdherenceCount || a.name.localeCompare(b.name))
                    .slice(0, 3)
                    .map(summarizeSection),
                sections: requestedMapId ? track.sections.map(summarizeSection) : undefined,
            };
        }),
    };
};

export const getAvailableUserSummaryMaps = (source: UserSummaryAiSource) => {
    const mapLevel = getUserSummaryMapLevel(source, {});
    if (mapLevel.status !== 'ready' || !('map_count' in mapLevel)) return mapLevel;
    const maps = mapLevel.maps.map((map) => ({
        id: map.id,
        name: map.name,
        analyzed_session_count: map.analyzed_session_count,
        total_analyzed_time_count: map.total_analyzed_time_count,
        section_count: map.section_count,
    }));
    const mapOptions = maps.map((map) => (
        `${map.name} (${map.id}) - ${map.analyzed_session_count} analyzed session${map.analyzed_session_count === 1 ? '' : 's'}`
    ));
    return {
        status: 'ready',
        map_count: mapLevel.map_count,
        maps,
        map_options: mapOptions,
        response_text: mapOptions.length > 0
            ? `Available maps in your summary:\n${mapOptions.map((option) => `- ${option}`).join('\n')}\nWhich map should I inspect?`
            : 'I do not see any maps in your user summary yet.',
    };
};

export const searchUserSummaryMapLevel = (
    source: UserSummaryAiSource,
    args: Record<string, any>,
) => {
    const mapLevel = getUserSummaryMapLevel(source, {});
    if (mapLevel.status !== 'ready' || !('map_count' in mapLevel)) return mapLevel;
    const query = normalizeSearchText(args.query);
    const terms = Array.from(new Set(query.split(' ').filter(Boolean)));
    if (terms.length === 0) {
        throw new QueryRequiredError(
            AI_TOOL_COMPONENT_NAMES.USER_SUMMARY,
            'Provide a user-summary search query.',
        );
    }
    const matches = mapLevel.maps.map((map) => {
        const fields: Array<{ name: string; value: unknown; weight: number }> = [
            { name: 'map_name', value: map.name, weight: 8 },
            { name: 'map_id', value: map.id, weight: 6 },
        ];
        map.top_mistake_sections.forEach((section) => fields.push(
            { name: 'top_mistake_section', value: section.name, weight: 5 },
            { name: 'top_mistake_section_id', value: section.id, weight: 3 },
        ));
        map.top_expert_adherence_sections.forEach((section) => fields.push(
            { name: 'top_expert_adherence_section', value: section.name, weight: 4 },
            { name: 'top_expert_adherence_section_id', value: section.id, weight: 3 },
        ));
        const matchedTerms = new Set<string>();
        const matchedFields = new Set<string>();
        let score = 0;
        fields.forEach((field) => {
            const value = normalizeSearchText(field.value);
            if (value.includes(query)) {
                score += field.weight * 2;
                matchedFields.add(field.name);
            }
            terms.forEach((term) => {
                if (!value.includes(term)) return;
                matchedTerms.add(term);
                matchedFields.add(field.name);
                score += field.weight;
            });
        });
        return matchedTerms.size === terms.length
            ? { map, search_score: score, matched_fields: Array.from(matchedFields) }
            : null;
    }).filter((match): match is NonNullable<typeof match> => Boolean(match))
        .sort((a, b) => b.search_score - a.search_score || a.map.name.localeCompare(b.map.name));
    return {
        status: 'ready',
        query,
        match_count: matches.length,
        map_count: mapLevel.map_count,
        maps: matches.slice(0, getSearchLimit(args.limit)).map((match) => ({
            ...match.map,
            search_score: match.search_score,
            matched_fields: match.matched_fields,
        })),
    };
};
