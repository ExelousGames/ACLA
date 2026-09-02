import { useEffect, useMemo, useRef, useState } from 'react';
import {
    Box,
    Card,
    Flex,
    Heading,
    Separator,
    Text,
    TextArea
} from '@radix-ui/themes';
import { useAiLabels } from 'contexts/AiLabelsContext';
import { useUserSummary } from 'contexts/UserSummaryContext';
import {
    NamedAiToolComponentHandle,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import AnalyzeAllSessionsControl from './AnalyzeAllSessionsControl';
import {
    asRecord,
    buildPracticeTrackSummaryViews,
    formatCount,
    formatNumber,
    formatPercent,
    PracticeParentSegmentView
} from './user-summary-model';
import './user-summary.css';
import {
    getAvailableUserSummaryMaps,
    getUserSummaryMapLevel,
    searchUserSummaryMapLevel,
} from './user-summary-ai-tools';
import {
    createAiToolOperationFrom,
    type AiToolOperation,
} from 'components/ai-engineering-tools';

export type UserSummaryMapLevelResult = {
    status: unknown;
    message?: unknown;
    map_count: unknown;
    maps: Array<{
        id: unknown;
        name: unknown;
        section_count: unknown;
        mistake_percent: unknown;
        expert_adherence_percent: unknown;
    }>;
};

export type AvailableUserSummaryMapsResult = {
    status: unknown;
    message?: unknown;
    map_count: unknown;
    map_options: unknown[];
};

export type UserSummaryMapSearchResult = {
    status: unknown;
    message?: unknown;
    query: unknown;
    match_count: unknown;
    maps: Array<{ id: unknown; name: unknown; matched_fields: unknown }>;
};

export interface UserSummaryHandle extends NamedAiToolComponentHandle {
    getUserSummaryMapLevel(args: Record<string, any>): AiToolOperation<UserSummaryMapLevelResult>;
    getAvailableUserSummaryMaps(): AiToolOperation<AvailableUserSummaryMapsResult>;
    searchUserSummaryMapLevel(args: Record<string, any>): AiToolOperation<UserSummaryMapSearchResult>;
}

const compactMapLevelForAi = (result: Record<string, any>) => ({
    status: result.status,
    ...(result.message ? { message: result.message } : {}),
    map_count: result.map_count ?? 0,
    maps: Array.isArray(result.maps)
        ? result.maps.map((map: Record<string, any>) => ({
            id: map.id,
            name: map.name,
            section_count: map.section_count,
            mistake_percent: map.mistake_percent,
            expert_adherence_percent: map.expert_adherence_percent,
        }))
        : [],
});

const compactAvailableMapsForAi = (result: Record<string, any>) => ({
    status: result.status,
    ...(result.message ? { message: result.message } : {}),
    map_count: result.map_count ?? 0,
    map_options: Array.isArray(result.map_options) ? result.map_options : [],
});

const compactMapSearchForAi = (result: Record<string, any>) => ({
    status: result.status,
    ...(result.message ? { message: result.message } : {}),
    query: result.query ?? null,
    match_count: result.match_count ?? 0,
    maps: Array.isArray(result.maps)
        ? result.maps.map((map: Record<string, any>) => ({
            id: map.id,
            name: map.name,
            matched_fields: map.matched_fields,
        }))
        : [],
});

type SegmentGroupProps = {
    title: string;
    emptyText: string;
    segments: PracticeParentSegmentView[];
    variant: 'mistake' | 'expert' | 'recovery';
};

const SegmentGroup = ({ title, emptyText, segments, variant }: SegmentGroupProps) => (
    <section className="user-summary-segment-group">
        <Text className="user-summary-section-title">{title}</Text>
        {segments.length > 0 ? (
            segments.map((segment) => (
                <div className={`user-summary-parent-segment user-summary-parent-segment--${variant}`} key={segment.id}>
                    <div className="user-summary-parent-heading">
                        <span>{segment.name}</span>
                        <span>{formatCount(segment.count)}</span>
                    </div>
                    <div className="user-summary-child-list">
                        {segment.childSegments.length > 0 ? (
                            segment.childSegments.map((childSegment) => (
                                <span
                                    className={`user-summary-child-pill user-summary-child-pill--${variant}`}
                                    key={`${segment.id}-${childSegment.id}-${childSegment.startIndex ?? 'start'}-${childSegment.endIndex ?? 'end'}`}
                                >
                                    {childSegment.name}
                                    <strong>{formatNumber(childSegment.count)}</strong>
                                    {childSegment.startIndex !== undefined && childSegment.endIndex !== undefined && (
                                        <em>{childSegment.startIndex}-{childSegment.endIndex}</em>
                                    )}
                                </span>
                            ))
                        ) : (
                            <Text size="2" color="gray">No child segments recorded.</Text>
                        )}
                    </div>
                </div>
            ))
        ) : (
            <Text size="2" color="gray">{emptyText}</Text>
        )}
    </section>
);

const UserSummary = ({ name }: { name: string }) => {
    const {
        userSummary,
        userSummaryLoading,
        userSummaryError,
        loadUserSummary
    } = useUserSummary();
    const {
        getLabelName,
        getCategoryLabels,
        loading: labelsLoading,
        error: labelsError
    } = useAiLabels();
    const [summaryText, setSummaryText] = useState('{}');

    useEffect(() => {
        setSummaryText(JSON.stringify(userSummary || {}, null, 2));
    }, [userSummary]);

    const parsedSummary = useMemo(() => {
        try {
            return JSON.parse(summaryText);
        } catch (error) {
            return null;
        }
    }, [summaryText]);

    const trackSummaries = useMemo(
        () => buildPracticeTrackSummaryViews(asRecord(parsedSummary), getLabelName, getCategoryLabels),
        [getCategoryLabels, getLabelName, parsedSummary],
    );
    const screenStateRef = useRef({
        userSummary,
        userSummaryLoading,
        userSummaryError,
        labelsLoading,
        labelsError,
        getLabelName,
        getCategoryLabels,
    });
    screenStateRef.current = {
        userSummary,
        userSummaryLoading,
        userSummaryError,
        labelsLoading,
        labelsError,
        getLabelName,
        getCategoryLabels,
    };
    const componentRef = useRef<UserSummaryHandle | null>(null);

    if (componentRef.current === null) {
        componentRef.current = {
            getComponentName: () => name,
            getUserSummaryMapLevel: (args) => createAiToolOperationFrom(() => compactMapLevelForAi(getUserSummaryMapLevel({
                userSummary: screenStateRef.current.userSummary || undefined,
                loading: screenStateRef.current.userSummaryLoading || screenStateRef.current.labelsLoading,
                error: screenStateRef.current.userSummaryError || screenStateRef.current.labelsError || undefined,
                getLabelName: screenStateRef.current.getLabelName,
                getCategoryLabels: screenStateRef.current.getCategoryLabels,
            }, args)), 'complete'),
            getAvailableUserSummaryMaps: () => createAiToolOperationFrom(() => compactAvailableMapsForAi(getAvailableUserSummaryMaps({
                userSummary: screenStateRef.current.userSummary || undefined,
                loading: screenStateRef.current.userSummaryLoading || screenStateRef.current.labelsLoading,
                error: screenStateRef.current.userSummaryError || screenStateRef.current.labelsError || undefined,
                getLabelName: screenStateRef.current.getLabelName,
                getCategoryLabels: screenStateRef.current.getCategoryLabels,
            })), 'complete'),
            searchUserSummaryMapLevel: (args) => createAiToolOperationFrom(() => compactMapSearchForAi(searchUserSummaryMapLevel({
                userSummary: screenStateRef.current.userSummary || undefined,
                loading: screenStateRef.current.userSummaryLoading || screenStateRef.current.labelsLoading,
                error: screenStateRef.current.userSummaryError || screenStateRef.current.labelsError || undefined,
                getLabelName: screenStateRef.current.getLabelName,
                getCategoryLabels: screenStateRef.current.getCategoryLabels,
            }, args)), 'complete'),
        };
    }
    useRegisterAiToolComponentRef(componentRef);


    return (
        <Box className="user-summary-container">
            <Card className="user-summary-card">
                <Flex direction="column" gap="4">
                    <Flex justify="between" align="center" gap="3" wrap="wrap">
                        <Box>
                            <Heading size="6">User Summary</Heading>
                            <Text size="2" color="gray">Most recent 10 practice sessions by track section</Text>
                        </Box>
                        <Flex gap="3" align="start" wrap="wrap">
                            <AnalyzeAllSessionsControl onCompleted={loadUserSummary} />
                        </Flex>
                    </Flex>

                    <Separator />

                    <Box className="user-summary-display">
                        {userSummaryLoading ? (
                            <Text size="2" color="gray">Loading summary</Text>
                        ) : !parsedSummary ? (
                            <Text size="2" color="red">Preview unavailable until JSON is valid.</Text>
                        ) : labelsLoading ? (
                            <Text size="2" color="gray">Loading AI track sections</Text>
                        ) : labelsError ? (
                            <Text size="2" color="red">Unable to load AI track sections.</Text>
                        ) : trackSummaries.length === 0 ? (
                            <Text size="2" color="gray">No practice session summary available yet.</Text>
                        ) : (
                            trackSummaries.map((track) => (
                                <article className="user-summary-track" key={track.id}>
                                    <header className="user-summary-track-header">
                                        <Box>
                                            <Text className="user-summary-kicker">Practice Track</Text>
                                            <Heading size="5">{track.name}</Heading>
                                        </Box>
                                        <div className="user-summary-stats">
                                            <span>{formatNumber(track.analyzedSessionCount)} analyzed</span>
                                            <span>{formatCount(track.totalAnalyzedTimeCount)} analyzed time</span>
                                            <span>{formatNumber(track.sections.length)} sections</span>
                                            {track.skippedSessionCount > 0 && (
                                                <span>{formatNumber(track.skippedSessionCount)} skipped</span>
                                            )}
                                            {track.failedSessionCount > 0 && (
                                                <span>{formatNumber(track.failedSessionCount)} failed</span>
                                            )}
                                        </div>
                                    </header>

                                    <section className="user-summary-section-list">
                                        {track.sections.length > 0 ? (
                                            track.sections.map((section) => (
                                                <div className="user-summary-track-section" key={section.id}>
                                                    <div className="user-summary-section-heading">
                                                        <Box>
                                                            <Text className="user-summary-kicker">Section</Text>
                                                            <Heading size="4">{section.name}</Heading>
                                                        </Box>
                                                        <div className="user-summary-section-metrics">
                                                            <span className="user-summary-section-metric user-summary-section-metric--mistake">
                                                                <strong>{formatPercent(section.mistakePercent)}</strong>
                                                                mistakes
                                                                <em>{formatCount(section.mistakeCount)}</em>
                                                            </span>
                                                            <span className="user-summary-section-metric user-summary-section-metric--expert">
                                                                <strong>{formatPercent(section.expertAdherencePercent)}</strong>
                                                                expert adherence
                                                                <em>{formatCount(section.expertAdherenceCount)}</em>
                                                            </span>
                                                            <span className="user-summary-section-metric">
                                                                <strong>{formatNumber(section.analyzedTimeCount)}</strong>
                                                                analyzed
                                                            </span>
                                                        </div>
                                                    </div>
                                                    <div className="user-summary-section-body">
                                                        <SegmentGroup
                                                            title="Mistakes"
                                                            emptyText="No mistakes recorded for this section."
                                                            segments={section.mistakeSegments}
                                                            variant="mistake"
                                                        />
                                                        <SegmentGroup
                                                            title="Recovery & Merge"
                                                            emptyText="No recovery or merge events recorded for this section."
                                                            segments={section.recoveryMergeSegments}
                                                            variant="recovery"
                                                        />
                                                        <SegmentGroup
                                                            title="Expert Adherence"
                                                            emptyText="No expert adherence recorded for this section."
                                                            segments={section.expertAdherenceSegments}
                                                            variant="expert"
                                                        />
                                                    </div>
                                                </div>
                                            ))
                                        ) : (
                                            <Text size="2" color="gray">No AI track sections available for this track.</Text>
                                        )}
                                    </section>
                                </article>
                            ))
                        )}
                    </Box>

                    <details className="user-summary-json-panel">
                        <summary>Raw JSON</summary>
                        <TextArea
                            className="user-summary-editor"
                            value={summaryText}
                            readOnly
                            disabled={userSummaryLoading}
                            spellCheck={false}
                        />
                    </details>

                    {userSummaryError && (
                        <Text size="2" color="red">
                            {userSummaryError}
                        </Text>
                    )}
                </Flex>
            </Card>
        </Box>
    );
};

export default UserSummary;
