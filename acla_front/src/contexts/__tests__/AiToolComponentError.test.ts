import {
    AiChatComponentError,
    AiToolComponentError,
    AiToolComponentRefError,
    BaselineCollectionComponentError,
    BaselineCollectionNotStartedError,
    BaselineLapRecordRequiredError,
    ComponentRefUnavailableError,
    GoalComponentError,
    GoalDeterminationInputIncompatibleError,
    GoalStepFailedError,
    InvalidLiveRangeTodoListError,
    LiveRangeTodoListComponentError,
    NoProcedurePlanError,
    NoRecordedSessionError,
    ProcedurePlanComponentError,
    ProcedurePlanStepFailedError,
    QueryRequiredError,
    SessionAnalysisComponentError,
    UserSummaryComponentError,
    VisualizationComponentError,
    VisualizationUpdateFailedError,
} from '../AiToolComponentError';

describe('AI tool component errors', () => {
    const subclasses = [
        [ComponentRefUnavailableError, AiToolComponentRefError],
        [GoalStepFailedError, GoalComponentError],
        [GoalDeterminationInputIncompatibleError, GoalComponentError],
        [ProcedurePlanStepFailedError, ProcedurePlanComponentError],
        [InvalidLiveRangeTodoListError, LiveRangeTodoListComponentError],
        [BaselineCollectionNotStartedError, BaselineCollectionComponentError],
        [BaselineLapRecordRequiredError, BaselineCollectionComponentError],
        [VisualizationUpdateFailedError, VisualizationComponentError],
        [QueryRequiredError, UserSummaryComponentError],
        [NoProcedurePlanError, AiChatComponentError],
        [NoRecordedSessionError, SessionAnalysisComponentError],
    ] as const;

    it.each(subclasses)('%s preserves only component identity and cause metadata', (ErrorType, Category) => {
        const cause = new Error('root cause');
        const error = new ErrorType(
            'test-component',
            'Component operation failed.',
            { cause },
        );

        expect(error).toBeInstanceOf(ErrorType);
        expect(error).toBeInstanceOf(Category);
        expect(error).toBeInstanceOf(AiToolComponentError);
        expect(error).toBeInstanceOf(Error);
        expect(error.name).toBe(ErrorType.name);
        expect(error.componentName).toBe('test-component');
        expect(error.message).toBe('Component operation failed.');
        expect(error.cause).toBe(cause);
        expect(error).not.toHaveProperty('code');
        expect(error).not.toHaveProperty('details');
    });
});
