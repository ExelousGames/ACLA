import { OperationError, OperationErrorOptions } from 'errors/OperationError';

export { OperationError } from 'errors/OperationError';
export type { OperationErrorOptions } from 'errors/OperationError';

export type OperationComponentErrorConstructor<TError extends OperationComponentError = OperationComponentError> = new (
    componentName: string,
    message: string,
    options?: OperationErrorOptions,
) => TError;

export abstract class OperationComponentError extends OperationError {
    override name = 'OperationComponentError';
    readonly componentName: string;

    constructor(componentName: string, message: string, options: OperationErrorOptions = {}) {
        super(message, options);
        this.componentName = componentName;
        Object.setPrototypeOf(this, new.target.prototype);
    }
}

export abstract class OperationComponentRefError extends OperationComponentError {
    override name = 'OperationComponentRefError';
}

export class ComponentRefUnavailableError extends OperationComponentRefError {
    override name = 'ComponentRefUnavailableError';
}

export class DuplicateComponentNameError extends OperationComponentRefError {
    override name = 'DuplicateComponentNameError';
}

export class ComponentMountTimeoutError extends OperationComponentRefError {
    override name = 'ComponentMountTimeoutError';
}

export abstract class GoalComponentError extends OperationComponentError {
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

export class InvalidGoalStopWhenError extends GoalComponentError {
    override name = 'InvalidGoalStopWhenError';
}

export class RecursiveGoalStopWhenError extends GoalComponentError {
    override name = 'RecursiveGoalStopWhenError';
}

export class GoalStepTaskUnavailableError extends GoalComponentError {
    override name = 'GoalStepTaskUnavailableError';
}

export class GoalStopWhenTaskUnavailableError extends GoalComponentError {
    override name = 'GoalStopWhenTaskUnavailableError';
}

export class GoalStepFailedError extends GoalComponentError {
    override name = 'GoalStepFailedError';
}

export class GoalStepOutputToolMismatchError extends GoalComponentError {
    override name = 'GoalStepOutputToolMismatchError';
}

export class GoalStopWhenFailedError extends GoalComponentError {
    override name = 'GoalStopWhenFailedError';
}

export class GoalStopWhenOutputToolMismatchError extends GoalComponentError {
    override name = 'GoalStopWhenOutputToolMismatchError';
}

export class GoalStopWhenInputIncompatibleError extends GoalComponentError {
    override name = 'GoalStopWhenInputIncompatibleError';
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

export abstract class ProcedurePlanComponentError extends OperationComponentError {
    override name = 'ProcedurePlanComponentError';
}

export class ProcedurePlanStepFailedError extends ProcedurePlanComponentError {
    override name = 'ProcedurePlanStepFailedError';
}

export class ProcedurePlanReplacedError extends ProcedurePlanComponentError {
    override name = 'ProcedurePlanReplacedError';
}

export abstract class LiveRangeTodoListComponentError extends OperationComponentError {
    override name = 'LiveRangeTodoListComponentError';
}

export class InvalidLiveRangeTodoListError extends LiveRangeTodoListComponentError {
    override name = 'InvalidLiveRangeTodoListError';
}

export class LiveRangeTodoListUnavailableError extends LiveRangeTodoListComponentError {
    override name = 'LiveRangeTodoListUnavailableError';
}

export abstract class BaselineCollectionComponentError extends OperationComponentError {
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

export abstract class VisualizationComponentError extends OperationComponentError {
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

export abstract class UserSummaryComponentError extends OperationComponentError {
    override name = 'UserSummaryComponentError';
}

export class UserSummaryUnavailableError extends UserSummaryComponentError {
    override name = 'UserSummaryUnavailableError';
}

export class QueryRequiredError extends UserSummaryComponentError {
    override name = 'QueryRequiredError';
}

export abstract class AiChatComponentError extends OperationComponentError {
    override name = 'AiChatComponentError';
}

export class NoProcedurePlanError extends AiChatComponentError {
    override name = 'NoProcedurePlanError';
}

export class ProcedurePlanAdvanceFailedError extends AiChatComponentError {
    override name = 'ProcedurePlanAdvanceFailedError';
}

export class RecordedSessionLiveOperationsUnavailableError extends AiChatComponentError {
    override name = 'RecordedSessionLiveOperationsUnavailableError';
}

export class NonLiveContextLiveOperationsUnavailableError extends AiChatComponentError {
    override name = 'NonLiveContextLiveOperationsUnavailableError';
}

export abstract class SessionAnalysisComponentError extends OperationComponentError {
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
