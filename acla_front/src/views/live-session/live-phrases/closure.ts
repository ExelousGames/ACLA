export interface ConditionSnapshot {
    description: string;
    conditionFit: boolean | null;
    inputMissing: boolean;
    connector?: 'and' | 'or';
    conditions?: readonly ConditionSnapshot[];
}

export type Condition<Context> = ((context: Context, state: State<Context>) => boolean) & {
    describe?: (context: Context, state: State<Context>) => readonly ConditionSnapshot[];
};

/** Descriptions must be read-only. A string description requires a pure predicate. */
export function describeCondition<Context>(
    predicate: Condition<Context>,
    description: string | ((context: Context, state: State<Context>) => readonly ConditionSnapshot[]),
): Condition<Context> {
    return Object.assign((context: Context, state: State<Context>) => predicate(context, state), {
        describe: typeof description === 'string'
            ? (context: Context, state: State<Context>) => [{ description, conditionFit: predicate(context, state), inputMissing: false }]
            : description,
    });
}

export interface NodeInspection {
    status?: string;
    fields?: Readonly<Record<string, string | number | boolean | null>>;
}

export interface NodeMetadata<Context> {
    id?: string;
    inspect?: (context: Context, state: State<Context>) => NodeInspection;
}

export interface ActionExecution {
    status: 'idle' | 'running' | 'completed' | 'failed';
    error?: string;
}

/** Plain display data: no callbacks, class instances or action-name conventions. */
export interface NodeSnapshot {
    id: string;
    kind: 'closure' | 'action';
    name: string;
    description: string;
    current: boolean;
    onPath: boolean;
    status: string;
    conditions: readonly ConditionSnapshot[];
    fields: NonNullable<NodeInspection['fields']>;
    execution?: ActionExecution;
    children: NodeSnapshot[];
}

/** An entered scope never returns to its parent automatically. */
export class Closure<Context> {
    constructor(
        readonly name: string,
        readonly description: string,
        readonly condition: Condition<Context>,
        readonly children: readonly (Closure<Context> | Action<Context>)[] = [],
        readonly metadata: NodeMetadata<Context> = {},
    ) {}
}

/** The callback may run any operation, including an asynchronous one. */
export class Action<Context> {
    constructor(
        readonly name: string,
        readonly description: string,
        readonly run: (context: Context, state: State<Context>) => unknown,
        readonly condition: Condition<Context> = describeCondition(() => true, 'Always eligible'),
        readonly metadata: NodeMetadata<Context> = {},
    ) {}

    static exitToRoot<Context>(condition?: Condition<Context>): Action<Context> {
        return new Action('exit to root', 'Return to the root closure so another guide can be selected.',
            (_context, state) => state.exitToRoot(), condition);
    }
}

/** Each step descends through eligible closures and runs at most one action. */
export class State<Context> {
    private scopes: Closure<Context>[];
    private location: Closure<Context> | Action<Context>;
    private executions = new Map<Action<Context>, ActionExecution>();
    private checked = new Map<Closure<Context> | Action<Context>, boolean>();
    private pending = false;

    constructor(readonly root: Closure<Context>) {
        this.scopes = [root];
        this.location = root;
    }

    get current(): Closure<Context> | Action<Context> { return this.location; }
    get closure(): Closure<Context> { return this.scopes[this.scopes.length - 1]; }
    get path(): readonly (Closure<Context> | Action<Context>)[] {
        return this.location === this.closure ? [...this.scopes] : [...this.scopes, this.location];
    }

    getExecution(action: Action<Context>): Readonly<ActionExecution> {
        return { ...(this.executions.get(action) ?? { status: 'idle' }) };
    }

    exitToRoot(): void {
        this.scopes = [this.root];
        this.location = this.root;
        this.executions.clear();
        this.checked.clear();
    }

    /** Inspect the entire tree without stepping it or invoking opaque predicates. */
    snapshot(context: Context): NodeSnapshot {
        const path = this.path;
        const copyCondition = (condition: ConditionSnapshot): ConditionSnapshot => ({
            description: condition.description, conditionFit: condition.conditionFit, inputMissing: condition.inputMissing,
            ...(condition.connector ? { connector: condition.connector } : {}),
            ...(condition.conditions ? { conditions: condition.conditions.map(copyCondition) } : {}),
        });
        const visit = (node: Closure<Context> | Action<Context>, id: string, parent?: Closure<Context>): NodeSnapshot => {
            const inspection = node.metadata.inspect?.(context, this);
            const conditions = node.condition.describe?.(context, this) ?? [{
                description: 'Custom condition (last checked)',
                conditionFit: this.checked.get(node) ?? null,
                inputMissing: false,
            }];
            const execution = node instanceof Action ? this.getExecution(node) : undefined;
            const onPath = path.includes(node);
            const status = execution && execution.status !== 'idle'
                ? { running: 'Running', completed: 'Completed', failed: 'Failed' }[execution.status]
                : node instanceof Action ? parent === this.closure ? 'Waiting' : 'Inactive'
                    : onPath ? 'Active' : 'Inactive';
            const nodeId = node.metadata.id ?? id;
            return {
                id: nodeId, kind: node instanceof Closure ? 'closure' : 'action',
                name: node.name, description: node.description, current: node === this.current, onPath,
                status: inspection?.status ?? status, conditions: conditions.map(copyCondition), fields: { ...inspection?.fields },
                ...(execution ? { execution } : {}),
                children: node instanceof Closure ? node.children.map((child, index) => visit(child, `${nodeId}/${index}`, node)) : [],
            };
        };
        return visit(this.root, 'root');
    }

    step(context: Context, onActionError?: (error: unknown) => void): unknown {
        if (this.pending) return;
        while (true) {
            const next = this.closure.children.find((child) => {
                if (child instanceof Action && this.executions.has(child)) return false;
                const matches = child.condition(context, this);
                this.checked.set(child, matches);
                return matches;
            });
            if (!next) return;
            this.location = next;
            if (next instanceof Closure) {
                this.scopes.push(next);
                continue;
            }
            // Claim the action before invoking user code, so reentrant steps cannot repeat it.
            const execution: ActionExecution = { status: 'running' };
            this.executions.set(next, execution);
            this.pending = true;
            try {
                const result = next.run(context, this);
                if (result && typeof (result as PromiseLike<unknown>).then === 'function') {
                    return Promise.resolve(result).then((value) => {
                        execution.status = 'completed';
                        return value;
                    }, (error) => {
                        execution.status = 'failed';
                        execution.error = error instanceof Error ? error.message : String(error);
                        if (onActionError) return onActionError(error);
                        throw error;
                    }).finally(() => { this.pending = false; });
                }
                execution.status = 'completed';
                this.pending = false;
                return result;
            } catch (error) {
                execution.status = 'failed';
                execution.error = error instanceof Error ? error.message : String(error);
                this.pending = false;
                if (onActionError) return onActionError(error);
                throw error;
            }
        }
    }
}
