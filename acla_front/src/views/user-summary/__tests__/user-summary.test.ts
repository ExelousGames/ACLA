import { buildTrackSummaryViews } from '../user-summary-model';

describe('buildTrackSummaryViews', () => {
    it('normalizes parent segment summary JSON for display', () => {
        const tracks = buildTrackSummaryViews({
            sessionAnalysis: {
                tracks: {
                    brands_hatch: {
                        trackName: 'Brands Hatch',
                        sessionsAnalyzed: 2,
                        totalTelemetryRows: 3000,
                        parentSegments: [
                            {
                                parentSegmentId: 'brands_hatch2',
                                parentSegmentName: 'Paddock Hill Bend',
                                expertLevelTurns: 4,
                                mistakes: 1,
                                childSegments: [
                                    {
                                        childSegmentId: 'EA',
                                        childSegmentName: 'Expert Adherence (Training)',
                                        count: 4,
                                        kind: 'strength',
                                    },
                                    {
                                        childSegmentId: 'MSP1',
                                        childSegmentName: 'Initiate brake too late',
                                        count: 1,
                                        kind: 'needs_work',
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        });

        expect(tracks).toHaveLength(1);
        expect(tracks[0].name).toBe('Brands Hatch');
        expect(tracks[0].parentSegments[0].name).toBe('Paddock Hill Bend');
        expect(tracks[0].strengths[0].childSegmentName).toBe('Expert Adherence (Training)');
        expect(tracks[0].improvementAreas[0].childSegmentName).toBe('Initiate brake too late');
    });

    it('builds parent segments from legacy section summaries', () => {
        const tracks = buildTrackSummaryViews({
            tracks: {
                brands_hatch: {
                    trackName: 'Brands Hatch',
                    sections: {
                        brands_hatch3: {
                            sectionName: 'Druids',
                            expertLevelTurns: 0,
                            mistakes: 2,
                            labelCounts: {
                                MSP2: 2,
                            },
                        },
                    },
                },
            },
        });

        expect(tracks[0].parentSegments[0].name).toBe('Druids');
        expect(tracks[0].parentSegments[0].childSegments[0]).toMatchObject({
            id: 'MSP2',
            name: 'MSP2',
            count: 2,
            kind: 'needs_work',
        });
        expect(tracks[0].improvementAreas[0].parentSegmentName).toBe('Druids');
    });

    it('uses AI labels when resolving segment names', () => {
        const labels: Record<string, string> = {
            brands_hatch3: 'Druids',
            MSP2: 'Missed apex',
        };
        const tracks = buildTrackSummaryViews({
            tracks: {
                brands_hatch: {
                    trackName: 'Brands Hatch',
                    sections: {
                        brands_hatch3: {
                            sectionName: 'Section 3',
                            mistakes: 2,
                            labelCounts: {
                                MSP2: 2,
                            },
                        },
                    },
                    improvementAreas: [
                        {
                            parentSegmentId: 'brands_hatch3',
                            childSegmentId: 'MSP2',
                            count: 2,
                        },
                    ],
                },
            },
        }, (labelId) => labels[labelId]);

        expect(tracks[0].parentSegments[0].name).toBe('Druids');
        expect(tracks[0].parentSegments[0].childSegments[0].name).toBe('Missed apex');
        expect(tracks[0].improvementAreas[0].parentSegmentName).toBe('Druids');
        expect(tracks[0].improvementAreas[0].childSegmentName).toBe('Missed apex');
    });
});
