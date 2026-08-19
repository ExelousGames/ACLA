import { AiToolError, AiToolErrorOptions } from 'errors/AiToolError';

export { AiToolError } from 'errors/AiToolError';
export type { AiToolErrorOptions } from 'errors/AiToolError';

export type AiToolComponentErrorConstructor<TError extends AiToolComponentError = AiToolComponentError> = new (
    componentName: string,
    message: string,
    options?: AiToolErrorOptions,
) => TError;

export abstract class AiToolComponentError extends AiToolError {
    override name = 'AiToolComponentError';
    readonly componentName: string;

    constructor(componentName: string, message: string, options: AiToolErrorOptions = {}) {
        super(message, options);
        this.componentName = componentName;
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

export abstract class AiToolComponentRefError extends AiToolComponentError {
    override name = 'AiToolComponentRefError';
}

export class ComponentRefUnavailableError extends AiToolComponentRefError {
    override name = 'ComponentRefUnavailableError';
}

export class DuplicateComponentNameError extends AiToolComponentRefError {
    override name = 'DuplicateComponentNameError';
}

export class ComponentMountTimeoutError extends AiToolComponentRefError {
    override name = 'ComponentMountTimeoutError';
}

export abstract class GoalComponentError extends AiToolComponentError {
    override name = 'GoalComponentError';
}

export class InvalidGoalNameError extends GoalComponentError {
    override name = 'InvalidGoalNameError';
}

export class InvalidGoalStepsError extends GoalComponentError {
    override name = 'InvalidGoalStepsError';
}

export class DuplicateGoalStepIdError extends GoalComponentError {
    override name = 'DuplicateGoalStepIdError';
}

export class RecursiveGoalStepError extends GoalComponentError {
    override name = 'RecursiveGoalStepError';
}

export class InvalidGoalDeterminationError extends GoalComponentError {
    override name = 'InvalidGoalDeterminationError';
}

export class RecursiveGoalDeterminationError extends GoalComponentError {
    override name = 'RecursiveGoalDeterminationError';
}

export class GoalStepTaskUnavailableError extends GoalComponentError {
    override name = 'GoalStepTaskUnavailableError';
}

export class GoalDeterminationTaskUnavailableError extends GoalComponentError {
    override name = 'GoalDeterminationTaskUnavailableError';
}

export class GoalStepFailedError extends GoalComponentError {
    override name = 'GoalStepFailedError';
}

export class GoalStepOutputToolMismatchError extends GoalComponentError {
    override name = 'GoalStepOutputToolMismatchError';
}

export class GoalDeterminationFailedError extends GoalComponentError {
    override name = 'GoalDeterminationFailedError';
}

export class GoalDeterminationOutputToolMismatchError extends GoalComponentError {
    override name = 'GoalDeterminationOutputToolMismatchError';
}

export class GoalDeterminationInputIncompatibleError extends GoalComponentError {
    override name = 'GoalDeterminationInputIncompatibleError';
}

export class GoalReplacedError extends GoalComponentError {
    override name = 'GoalReplacedError';
}

export class GoalClearedError extends GoalComponentError {
    override name = 'GoalClearedError';
}

export class GoalDisposedError extends GoalComponentError {
    override name = 'GoalDisposedError';
}

export class GoalTaskRetryUnavailableError extends GoalComponentError {
    override name = 'GoalTaskRetryUnavailableError';
}

export abstract class ProcedurePlanComponentError extends AiToolComponentError {
    override name = 'ProcedurePlanComponentError';
}

export class ProcedurePlanStepFailedError extends ProcedurePlanComponentError {
    override name = 'ProcedurePlanStepFailedError';
}

export class ProcedurePlanReplacedError extends ProcedurePlanComponentError {
    override name = 'ProcedurePlanReplacedError';
}

export abstract class LiveRangeTodoListComponentError extends AiToolComponentError {
    override name = 'LiveRangeTodoListComponentError';
}

export class InvalidLiveRangeTodoListError extends LiveRangeTodoListComponentError {
    override name = 'InvalidLiveRangeTodoListError';
}

export class LiveRangeTodoListUnavailableError extends LiveRangeTodoListComponentError {
    override name = 'LiveRangeTodoListUnavailableError';
}

export abstract class BaselineCollectionComponentError extends AiToolComponentError {
    override name = 'BaselineCollectionComponentError';
}

export class BaselineCollectionAlreadyStartedError extends BaselineCollectionComponentError {
    override name = 'BaselineCollectionAlreadyStartedError';
}

export class BaselineCollectionNotStartedError extends BaselineCollectionComponentError {
    override name = 'BaselineCollectionNotStartedError';
}

export class BaselineCollectionVisualizationRequiredError extends BaselineCollectionComponentError {
    override name = 'BaselineCollectionVisualizationRequiredError';
}

export class BaselineLapRecordRequiredError extends BaselineCollectionComponentError {
    override name = 'BaselineLapRecordRequiredError';
}

export class BaselineAnalysisCancelledError extends BaselineCollectionComponentError {
    override name = 'BaselineAnalysisCancelledError';
}

export class AnalysisResultsVisualizationUnavailableError extends BaselineCollectionComponentError {
    override name = 'AnalysisResultsVisualizationUnavailableError';
}

export class BaselineCollectionIncompleteError extends BaselineCollectionComponentError {
    override name = 'BaselineCollectionIncompleteError';
}

export abstract class VisualizationComponentError extends AiToolComponentError {
    override name = 'VisualizationComponentError';
}

export class AnalysisResultsVisualizationNotReadyError extends VisualizationComponentError {
    override name = 'AnalysisResultsVisualizationNotReadyError';
}

export class VisualizationControlFailedError extends VisualizationComponentError {
    override name = 'VisualizationControlFailedError';
}

export class VisualizationManagerUnavailableError extends VisualizationComponentError {
    override name = 'VisualizationManagerUnavailableError';
}

export class VisualizationRequestFailedError extends VisualizationComponentError {
    override name = 'VisualizationRequestFailedError';
}

export class VisualizationUpdateFailedError extends VisualizationComponentError {
    override name = 'VisualizationUpdateFailedError';
}

export class VisualizationCloseFailedError extends VisualizationComponentError {
    override name = 'VisualizationCloseFailedError';
}

export class ComponentDisableFailedError extends VisualizationComponentError {
    override name = 'ComponentDisableFailedError';
}

export abstract class UserSummaryComponentError extends AiToolComponentError {
    override name = 'UserSummaryComponentError';
}

export class UserSummaryUnavailableError extends UserSummaryComponentError {
    override name = 'UserSummaryUnavailableError';
}

export class QueryRequiredError extends UserSummaryComponentError {
    override name = 'QueryRequiredError';
}

export abstract class AiChatComponentError extends AiToolComponentError {
    override name = 'AiChatComponentError';
}

export class NoProcedurePlanError extends AiChatComponentError {
    override name = 'NoProcedurePlanError';
}

export class ProcedurePlanAdvanceFailedError extends AiChatComponentError {
    override name = 'ProcedurePlanAdvanceFailedError';
}

export class RecordedSessionLiveToolsUnavailableError extends AiChatComponentError {
    override name = 'RecordedSessionLiveToolsUnavailableError';
}

export class NonLiveContextLiveToolsUnavailableError extends AiChatComponentError {
    override name = 'NonLiveContextLiveToolsUnavailableError';
}

export abstract class SessionAnalysisComponentError extends AiToolComponentError {
    override name = 'SessionAnalysisComponentError';
}

export class NoRecordedSessionError extends SessionAnalysisComponentError {
    override name = 'NoRecordedSessionError';
}

export class RecordedAnalysisFailedError extends SessionAnalysisComponentError {
    override name = 'RecordedAnalysisFailedError';
}

export class SessionAnalysisFailedError extends SessionAnalysisComponentError {
    override name = 'SessionAnalysisFailedError';
}

export class PerformanceInsightsFailedError extends SessionAnalysisComponentError {
    override name = 'PerformanceInsightsFailedError';
}

export class LapComparisonFailedError extends SessionAnalysisComponentError {
    override name = 'LapComparisonFailedError';
}

export class ExpertLineGuidanceFailedError extends SessionAnalysisComponentError {
    override name = 'ExpertLineGuidanceFailedError';
}

export class TelemetryDataFailedError extends SessionAnalysisComponentError {
    override name = 'TelemetryDataFailedError';
}
