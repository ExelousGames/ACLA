import { OperationComponentBase } from './OperationComponentBase';

/** Shared component lifecycle for procedure, repeatable, and live range workflows. */
export abstract class WorkflowComponentBase<TSnapshot> extends OperationComponentBase<TSnapshot> {
    readonly kind = 'workflow' as const;
}
