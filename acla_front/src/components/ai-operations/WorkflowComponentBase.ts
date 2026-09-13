import { OperationComponentBase } from './OperationComponentBase';
import type { Operation } from './operation';

/** Shared component lifecycle for procedure, repeatable, and live range workflows. */
export abstract class WorkflowComponentBase<TSnapshot> extends OperationComponentBase<TSnapshot> {
    readonly kind = 'workflow' as const;
    readonly runnerId = Symbol('workflow-runner');
    private parent: WorkflowComponentBase<any> | undefined;
    private executionToken: symbol | undefined;

    assertAvailable(): void {
        if (this.isDisposed()) throw new Error('The workflow runner has been disposed.');
    }

    private isAncestorOf(runner: WorkflowComponentBase<any> | undefined): boolean {
        for (let parent = runner?.parent; parent; parent = parent.parent) {
            if (parent === this) return true;
        }
        return false;
    }

    assertCanReplace(caller?: WorkflowComponentBase<any>): void {
        caller?.assertAvailable();
        if (caller === this || this.isAncestorOf(caller)) {
            throw new Error('Cannot replace the executing workflow or an active ancestor.');
        }
    }

    assertCanAppend(caller?: WorkflowComponentBase<any>): void {
        this.assertAvailable();
        caller?.assertAvailable();
        if (this.isAncestorOf(caller) || caller?.isAncestorOf(this)) {
            throw new Error('Cannot append to an ancestor or descendant workflow.');
        }
    }

    protected beginExecution(parent?: WorkflowComponentBase<any>): symbol {
        this.assertAvailable();
        this.parent = parent;
        return this.executionToken = Symbol('workflow-execution');
    }

    protected trackExecution<T extends Operation<any, any>>(operation: T, token: symbol): T {
        operation.notifyTerminated(() => {
            if (this.executionToken !== token) return;
            this.parent = undefined;
            this.executionToken = undefined;
        });
        return operation;
    }
}

export interface MountedWorkflow {
    runner: WorkflowComponentBase<any>;
    dispose: () => void;
}

export type MountWorkflow = (workflow: MountedWorkflow) => void;
