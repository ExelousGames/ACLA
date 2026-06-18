import { useCallback, useEffect, useMemo, useState } from 'react';
import {
    Box,
    Button,
    Card,
    Flex,
    Heading,
    Separator,
    Text,
    TextArea
} from '@radix-ui/themes';
import { CheckIcon } from '@radix-ui/react-icons';
import { useAiLabels } from 'contexts/AiLabelsContext';
import apiService from 'services/api.service';
import AnalyzeAllSessionsControl from './AnalyzeAllSessionsControl';
import { asRecord, buildTrackSummaryViews, formatCount, formatNumber } from './user-summary-model';
import './user-summary.css';

type UserSummaryResponse = {
    summary: Record<string, any>;
};

const UserSummary = () => {
    const { getLabelName } = useAiLabels();
    const [summaryText, setSummaryText] = useState('{}');
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');

    const loadSummary = useCallback(async () => {
        try {
            const response = await apiService.get<UserSummaryResponse>('/userinfo/summary');
            setSummaryText(JSON.stringify(response.data.summary || {}, null, 2));
        } catch (error) {
            setError('Unable to load user summary');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        void loadSummary();
    }, [loadSummary]);

    const handleSave = async () => {
        setError('');
        setMessage('');

        let summary: Record<string, any>;
        try {
            summary = JSON.parse(summaryText);
        } catch (error) {
            setError('Summary must be valid JSON');
            return;
        }

        setIsSaving(true);
        try {
            const response = await apiService.put<UserSummaryResponse>('/userinfo/summary', { summary });
            setSummaryText(JSON.stringify(response.data.summary || {}, null, 2));
            setMessage('Saved');
        } catch (error) {
            setError('Unable to save user summary');
        } finally {
            setIsSaving(false);
        }
    };

    const parsedSummary = useMemo(() => {
        try {
            return JSON.parse(summaryText);
        } catch (error) {
            return null;
        }
    }, [summaryText]);

    const trackSummaries = useMemo(
        () => buildTrackSummaryViews(asRecord(parsedSummary), getLabelName),
        [getLabelName, parsedSummary],
    );

    return (
        <Box className="user-summary-container">
            <Card className="user-summary-card">
                <Flex direction="column" gap="4">
                    <Flex justify="between" align="center" gap="3" wrap="wrap">
                        <Box>
                            <Heading size="6">User Summary</Heading>
                            <Text size="2" color="gray">Track performance summary</Text>
                        </Box>
                        <Flex gap="3" align="start" wrap="wrap">
                            <AnalyzeAllSessionsControl onCompleted={loadSummary} />
                            <Button onClick={handleSave} disabled={isLoading || isSaving}>
                                <CheckIcon />
                                {isSaving ? 'Saving' : 'Save JSON'}
                            </Button>
                        </Flex>
                    </Flex>

                    <Separator />

                    <Box className="user-summary-display">
                        {isLoading ? (
                            <Text size="2" color="gray">Loading summary</Text>
                        ) : !parsedSummary ? (
                            <Text size="2" color="red">Preview unavailable until JSON is valid.</Text>
                        ) : trackSummaries.length === 0 ? (
                            <Text size="2" color="gray">No track summary available yet.</Text>
                        ) : (
                            trackSummaries.map((track) => (
                                <article className="user-summary-track" key={track.id}>
                                    <header className="user-summary-track-header">
                                        <Box>
                                            <Text className="user-summary-kicker">Track</Text>
                                            <Heading size="5">{track.name}</Heading>
                                        </Box>
                                        <div className="user-summary-stats">
                                            <span>{formatNumber(track.sessionsAnalyzed)} analyzed</span>
                                            <span>{formatNumber(track.totalTelemetryRows)} rows</span>
                                            <span>{formatNumber(track.parentSegments.length)} areas</span>
                                        </div>
                                    </header>

                                    <div className="user-summary-track-body">
                                        <section className="user-summary-highlight-group">
                                            <Text className="user-summary-section-title">Did Well</Text>
                                            {track.strengths.length > 0 ? (
                                                track.strengths.map((item) => (
                                                    <div className="user-summary-highlight" key={`${item.parentSegmentId}-${item.childSegmentId}`}>
                                                        <span className="user-summary-highlight-name">{item.childSegmentName}</span>
                                                        <span className="user-summary-highlight-meta">{item.parentSegmentName} - {formatCount(item.count)}</span>
                                                    </div>
                                                ))
                                            ) : (
                                                <Text size="2" color="gray">No strengths detected yet.</Text>
                                            )}
                                        </section>

                                        <section className="user-summary-highlight-group">
                                            <Text className="user-summary-section-title">Needs Work</Text>
                                            {track.improvementAreas.length > 0 ? (
                                                track.improvementAreas.map((item) => (
                                                    <div className="user-summary-highlight" key={`${item.parentSegmentId}-${item.childSegmentId}`}>
                                                        <span className="user-summary-highlight-name">{item.childSegmentName}</span>
                                                        <span className="user-summary-highlight-meta">{item.parentSegmentName} - {formatCount(item.count)}</span>
                                                    </div>
                                                ))
                                            ) : (
                                                <Text size="2" color="gray">No problem areas detected yet.</Text>
                                            )}
                                        </section>
                                    </div>

                                    <section className="user-summary-parent-list">
                                        <Text className="user-summary-section-title">Track Areas</Text>
                                        {track.parentSegments.map((parentSegment) => (
                                            <div className="user-summary-parent-segment" key={parentSegment.id}>
                                                <div className="user-summary-parent-heading">
                                                    <span>{parentSegment.name}</span>
                                                    <span>{formatNumber(parentSegment.expertLevelTurns)} good / {formatNumber(parentSegment.mistakes)} issues</span>
                                                </div>
                                                <div className="user-summary-child-list">
                                                    {parentSegment.childSegments.map((childSegment) => (
                                                        <span
                                                            className={`user-summary-child-pill user-summary-child-pill--${childSegment.kind}`}
                                                            key={childSegment.id}
                                                        >
                                                            {childSegment.name}
                                                            <strong>{formatNumber(childSegment.count)}</strong>
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        ))}
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
                            onChange={(event) => setSummaryText(event.target.value)}
                            disabled={isLoading}
                            spellCheck={false}
                        />
                    </details>

                    {(message || error) && (
                        <Text size="2" color={error ? 'red' : 'green'}>
                            {error || message}
                        </Text>
                    )}
                </Flex>
            </Card>
        </Box>
    );
};

export default UserSummary;
