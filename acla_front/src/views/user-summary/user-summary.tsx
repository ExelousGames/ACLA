import { useEffect, useMemo, useState } from 'react';
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

const UserSummary = () => {
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
