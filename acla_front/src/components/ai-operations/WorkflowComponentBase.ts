import { OperationComponentBase } from './OperationComponentBase';

/** Shared component lifecycle for procedure, repeatable, and live range workflows. */
export abstract class WorkflowComponentBase<TSnapshot> extends OperationComponentBase<TSnapshot> {
    readonly kind = 'workflow' as const;
}

export interface MountedWorkflow {
    runner: WorkflowComponentBase<any>;
    dispose: () => void;
    retainOnHide?: boolean;
}

export type MountWorkflow = (workflow: MountedWorkflow) => void;
