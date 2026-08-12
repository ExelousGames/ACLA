import {
    QueryRequiredError,
    UserSummaryComponentError,
    UserSummaryUnavailableError,
} from 'contexts/AiToolComponentError';
import { AI_TOOL_COMPONENT_NAMES } from 'contexts/AiToolComponentRefContext';
import {
    getAvailableUserSummaryMaps,
    getUserSummaryMapLevel,
    searchUserSummaryMapLevel,
} from '../user-summary-ai-tools';

describe('user summary AI component operations', () => {
    const populatedSource = {
        userSummary: {
            sessionAnalysis: {
                practice: {
                    tracks: {
                        monza: {
                            trackName: 'Monza',
                            analyzedSessionCount: 1,
                            sections: {},
                        },
                    },
                },
            },
        },
    };

    it('keeps loading and empty summaries as expected values', () => {
        expect(getUserSummaryMapLevel({ loading: true }, {})).toEqual({
            status: 'loading',
            maps: [],
        });
        expect(getAvailableUserSummaryMaps({ userSummary: {} })).toEqual({
            status: 'empty',
            maps: [],
        });
    });

    it('throws a user-summary component error when summary loading failed', () => {
        let thrown: unknown;
        try {
            getUserSummaryMapLevel({ error: 'Summary service unavailable.' }, {});
        } catch (error) {
            thrown = error;
        }

        expect(thrown).toBeInstanceOf(UserSummaryComponentError);
        expect(thrown).toBeInstanceOf(UserSummaryUnavailableError);
        expect(thrown).toMatchObject({
            name: 'UserSummaryUnavailableError',
            componentName: AI_TOOL_COMPONENT_NAMES.USER_SUMMARY,
            message: 'Summary service unavailable.',
        });
    });

    it('throws a user-summary component error when a search query is missing', () => {
        expect(() => searchUserSummaryMapLevel(populatedSource, { query: '   ' }))
            .toThrow(QueryRequiredError);
        expect(() => searchUserSummaryMapLevel(populatedSource, { query: '   ' }))
            .toThrow('Provide a user-summary search query.');
    });
});
