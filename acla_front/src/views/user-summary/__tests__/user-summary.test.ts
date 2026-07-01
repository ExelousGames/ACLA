import { buildPracticeTrackSummaryViews, formatPercent } from '../user-summary-model';

describe('buildPracticeTrackSummaryViews', () => {
    const labelNames: Record<string, string> = {
        brands_hatch: 'Brands Hatch',
        brands_hatch1: 'Paddock Hill Bend',
        brands_hatch2: 'Druids',
        MSP: 'Mistake (Practice)',
        MSP1: 'Brake too late',
        MSP2: 'Missed apex',
        RM: 'Recovery & Merge',
        RM7: 'Merge back to expert line',
        EA: 'Expert Adherence',
        EA1: 'Strong throttle pickup',
        MSR: 'Mistake (Racing)',
        MSR2: 'Defense broken',
    };
    const resolveLabel = (labelId: string) => labelNames[labelId];
    const categories: Record<string, string[]> = {
        brands_hatch: ['brands_hatch1', 'brands_hatch2'],
        MSP: ['MSP1', 'MSP2'],
        RM: ['RM7'],
        EA: [],
        MSR: ['MSR2'],
    };
    const resolveCategory = (category: string) => categories[category] ?? [];

    it('normalizes the new practice summary JSON by AI track sections', () => {
        const tracks = buildPracticeTrackSummaryViews({
            sessionAnalysis: {
                practice: {
                    tracks: {
                        brands_hatch: {
                            trackName: 'Brands Hatch GP',
                            analyzedSessionCount: 3,
                            skippedSessionCount: 1,
                            failedSessionCount: 0,
                            totalAnalyzedTimeCount: 50,
                            sections: {
                                brands_hatch1: {
                                    analyzedTimeCount: 10,
                                    mistakeCount: 3,
                                    expertAdherenceCount: 7,
                                    parentSegments: [
                                        {
                                            id: 'MSP',
                                            type: 'mistake',
                                            count: 3,
                                            childSegments: [
                                                {
                                                    id: 'MSP1',
                                                    count: 2,
                                                    startIndex: 12,
                                                    endIndex: 18,
                                                },
                                            ],
                                        },
                                        {
                                            id: 'EA',
                                            type: 'expert_adherence',
                                            count: 7,
                                            childSegments: [
                                                {
                                                    id: 'EA1',
                                                    count: 7,
                                                },
                                            ],
                                        },
                                    ],
                                },
                            },
                        },
                    },
                },
            },
        }, resolveLabel, resolveCategory);

        expect(tracks).toHaveLength(1);
        expect(tracks[0]).toMatchObject({
            id: 'brands_hatch',
            name: 'Brands Hatch GP',
            analyzedSessionCount: 3,
            skippedSessionCount: 1,
            failedSessionCount: 0,
            totalAnalyzedTimeCount: 50,
        });
        expect(tracks[0].sections.map((section) => section.name)).toEqual([
            'Paddock Hill Bend',
            'Druids',
        ]);
        expect(tracks[0].sections[0].mistakeSegments[0]).toMatchObject({
            id: 'MSP',
            name: 'Mistake (Practice)',
            count: 3,
        });
        expect(tracks[0].sections[0].mistakeSegments[0].childSegments[0]).toMatchObject({
            id: 'MSP1',
            name: 'Brake too late',
            count: 2,
            startIndex: 12,
            endIndex: 18,
        });
        expect(tracks[0].sections[0].expertAdherenceSegments[0].childSegments[0]).toMatchObject({
            id: 'EA1',
            name: 'Strong throttle pickup',
            count: 7,
        });
    });

    it('calculates mistake and expert adherence percentages from analyzed time count', () => {
        const tracks = buildPracticeTrackSummaryViews({
            sessionAnalysis: {
                practice: {
                    tracks: {
                        brands_hatch: {
                            sections: {
                                brands_hatch1: {
                                    analyzedTimeCount: 10,
                                    mistakeCount: 3,
                                    expertAdherenceCount: 4,
                                    parentSegments: [],
                                },
                            },
                        },
                    },
                },
            },
        }, resolveLabel, () => ['brands_hatch1']);

        expect(tracks[0].sections[0].mistakePercent).toBe(30);
        expect(tracks[0].sections[0].expertAdherencePercent).toBe(40);
        expect(formatPercent(tracks[0].sections[0].mistakePercent)).toBe('30%');
    });

    it('returns zero percentages when no analyzed time exists', () => {
        const tracks = buildPracticeTrackSummaryViews({
            sessionAnalysis: {
                practice: {
                    tracks: {
                        brands_hatch: {
                            sections: {
                                brands_hatch1: {
                                    analyzedTimeCount: 0,
                                    mistakeCount: 3,
                                    expertAdherenceCount: 4,
                                    parentSegments: [],
                                },
                            },
                        },
                    },
                },
            },
        }, resolveLabel, () => ['brands_hatch1']);

        expect(tracks[0].sections[0].mistakePercent).toBe(0);
        expect(tracks[0].sections[0].expertAdherencePercent).toBe(0);
        expect(formatPercent(tracks[0].sections[0].mistakePercent)).toBe('0%');
    });

    it('uses only AI-provided sections for section ordering', () => {
        const tracks = buildPracticeTrackSummaryViews({
            sessionAnalysis: {
                practice: {
                    tracks: {
                        brands_hatch: {
                            sections: {
                                brands_hatch2: {
                                    analyzedTimeCount: 8,
                                    mistakeCount: 1,
                                    expertAdherenceCount: 7,
                                    parentSegments: [],
                                },
                                unknown_section: {
                                    analyzedTimeCount: 99,
                                    mistakeCount: 99,
                                    expertAdherenceCount: 0,
                                    parentSegments: [],
                                },
                            },
                        },
                    },
                },
            },
        }, resolveLabel, () => ['brands_hatch2']);

        expect(tracks[0].sections).toHaveLength(1);
        expect(tracks[0].sections[0].id).toBe('brands_hatch2');
        expect(tracks[0].sections[0].analyzedTimeCount).toBe(8);
    });

    it('normalizes current analyzer section label counts into practice sections', () => {
        const tracks = buildPracticeTrackSummaryViews({
            sessionAnalysis: {
                tracks: {
                    brands_hatch: {
                        trackName: 'Brands Hatch',
                        sessionsAnalyzed: 2,
                        sections: {
                            brands_hatch1: {
                                labelCounts: {
                                    brands_hatch1: 1,
                                    MSP: 3,
                                    MSP1: 2,
                                    RM: 7,
                                    RM7: 1,
                                    EA: 4,
                                    EA1: 1,
                                    MSR: 99,
                                    MSR2: 10,
                                },
                            },
                        },
                    },
                },
            },
        }, resolveLabel, resolveCategory);

        expect(tracks).toHaveLength(1);
        expect(tracks[0]).toMatchObject({
            name: 'Brands Hatch',
            analyzedSessionCount: 2,
            totalAnalyzedTimeCount: 14,
        });
        expect(tracks[0].sections[0]).toMatchObject({
            id: 'brands_hatch1',
            analyzedTimeCount: 14,
            mistakeCount: 3,
            expertAdherenceCount: 4,
        });
        expect(tracks[0].sections[0].mistakePercent).toBeCloseTo(21.428571);
        expect(tracks[0].sections[0].expertAdherencePercent).toBeCloseTo(28.571428);
        expect(tracks[0].sections[0].mistakeSegments[0]).toMatchObject({
            id: 'MSP',
            name: 'Mistake (Practice)',
            count: 3,
        });
        expect(tracks[0].sections[0].recoveryMergeSegments[0]).toMatchObject({
            id: 'RM',
            name: 'Recovery & Merge',
            count: 7,
        });
        expect(tracks[0].sections[0].recoveryMergeSegments[0].childSegments[0]).toMatchObject({
            id: 'RM7',
            name: 'Merge back to expert line',
            count: 1,
        });
        expect(tracks[0].sections[0].mistakeSegments[0].childSegments[0]).toMatchObject({
            id: 'MSP1',
            name: 'Brake too late',
            count: 2,
        });
        expect(tracks[0].sections[0].mistakeSegments.map((segment) => segment.id)).not.toContain('MSR');
        expect(tracks[0].sections[0].expertAdherenceSegments[0].childSegments).toEqual([]);
    });
});
