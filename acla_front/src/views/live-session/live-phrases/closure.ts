export interface ConditionSnapshot {
    description: string;
    conditionFit: boolean | null;
    inputMissing: boolean;
    connector?: 'and' | 'or';
    conditions?: readonly ConditionSnapshot[];
}

function copyCondition(condition: ConditionSnapshot): ConditionSnapshot {
    return {
        description: condition.description, conditionFit: condition.conditionFit, inputMissing: condition.inputMissing,
        ...(condition.connector ? { connector: condition.connector } : {}),
        ...(condition.conditions ? { conditions: condition.conditions.map(copyCondition) } : {}),
    };
}

export type Condition<Context> = ((context: Context, state: State<Context>) => boolean) & {
    describe?: (context: Context, state: State<Context>, conditionFit?: boolean) => readonly ConditionSnapshot[];
    unevaluated?: readonly ConditionSnapshot[];
    reset?: (state: State<Context>, parentExited: boolean) => void;
};

/** Descriptions are read only and run with the condition, never during inspection. */
export function describeCondition<Context>(
    predicate: Condition<Context>,
    description: string | ((context: Context, state: State<Context>) => readonly ConditionSnapshot[]),
    unevaluated?: readonly ConditionSnapshot[],
): Condition<Context> {
    return Object.assign((context: Context, state: State<Context>) => predicate(context, state), {
        unevaluated: unevaluated ?? (typeof description === 'string'
            ? [{ description, conditionFit: null, inputMissing: false }] : undefined),
        describe: typeof description === 'string'
            ? (_context: Context, _state: State<Context>, conditionFit?: boolean) => [{ description, conditionFit: conditionFit ?? null, inputMissing: false }]
            : description,
    });
}

export interface NodeInspection {
    status?: string;
    fields?: Readonly<Record<string, string | number | boolean | null>>;
}

export interface NodeMetadata<Context> {
    preparationTimeMs?: number;
    id?: string;
    inspect?: (context: Context, state: State<Context>) => NodeInspection;
    fields?: NodeInspection['fields'];
    /** Higher-priority siblings are checked first when no child closure is active. */
    priority?: number;
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

export interface Evaluable<Context> {
    evaluate(context: Context, state: State<Context>, onActionError?: (error: unknown) => void): unknown;
}

/** An entered scope evaluates its own children and leaves through an explicit action. */
export class Closure<Context> implements Evaluable<Context> {
    get preparationTimeMs(): number { return this.metadata.preparationTimeMs ?? 1000; }
    constructor(
        readonly name: string,
        readonly description: string,
        readonly condition: Condition<Context>,
        readonly children: readonly (Closure<Context> | Action<Context>)[] = [],
        readonly metadata: NodeMetadata<Context> = {},
    ) {}

    evaluate(context: Context, state: State<Context>, onActionError?: (error: unknown) => void): unknown {
        if (!state.path.includes(this)) return;
        context = state.contextFor(this, context);
        const active = this.children.find((child) => child instanceof Closure && state.path.includes(child));
        if (active) return active.evaluate(context, state, onActionError);
        const children = [...this.children].sort((a, b) => (b.metadata.priority ?? 0) - (a.metadata.priority ?? 0));
        for (const child of children) {
            if (!state.check(child, context)) continue;
            state.enter(child);
            return child.evaluate(context, state, onActionError);
        }
    }
}

/** The callback may run any operation, including an asynchronous one. */
export class Action<Context> implements Evaluable<Context> {
    constructor(
        readonly name: string,
        readonly description: string,
        readonly run: (context: Context, state: State<Context>) => unknown,
        readonly condition: Condition<Context> = describeCondition(() => true, 'Always eligible'),
        readonly metadata: NodeMetadata<Context> = {},
    ) {}

    evaluate(context: Context, state: State<Context>, onActionError?: (error: unknown) => void): unknown {
        return state.execute(this, context, onActionError);
    }

    static exitToRoot<Context>(condition?: Condition<Context>, metadata: NodeMetadata<Context> = {}): Action<Context> {
        return new Action('exit to root', 'Return to the root closure so another guide can be selected.',
            (_context, state) => state.exitToRoot(), condition, metadata);
    }

    static exitToParent<Context>(condition?: Condition<Context>, metadata: NodeMetadata<Context> = {}): Action<Context> {
        return new Action('exit to parent', 'Return to the enclosing closure.',
            (_context, state) => state.exitToParent(), condition, metadata);
    }
}

/** Each step descends through eligible closures and runs at most one action. */
export class State<Context> {
    private scopes: Closure<Context>[];
    private location: Closure<Context> | Action<Context>;
    private executions = new Map<Action<Context>, ActionExecution>();
    private checked = new Map<Closure<Context> | Action<Context>, { matches: boolean; conditions: readonly ConditionSnapshot[] }>();
    private inspections = new Map<Closure<Context> | Action<Context>, NodeInspection>();
    private pending = false;
    private parents = new Map<Closure<Context> | Action<Context>, Closure<Context>>();
    private results = new Map<Closure<Context> | Action<Context>, boolean>();
    private prepared = new Map<Closure<Context>, Context>();

    constructor(readonly root: Closure<Context>, private readonly prepareContext?: (
        node: Closure<Context>, context: Context, matches?: (candidate: Context) => boolean,
    ) => Context) {
        this.scopes = [root];
        this.location = root;
        const register = (parent: Closure<Context>) => parent.children.forEach((child) => {
            this.parents.set(child, parent);
            if (child instanceof Closure) register(child);
        });
        register(root);
    }

    get current(): Closure<Context> | Action<Context> { return this.location; }
    contextFor(node: Closure<Context>, context: Context): Context {
        return this.prepared.get(node) ?? this.prepareContext?.(node, context) ?? context;
    }
    /** Release an action's position while retaining its execution for this scope visit. */
    leaveAction(): void {
        if (this.location instanceof Action) this.location = this.closure;
    }
    get closure(): Closure<Context> { return this.scopes[this.scopes.length - 1]; }
    get path(): readonly (Closure<Context> | Action<Context>)[] {
        return this.location === this.closure ? [...this.scopes] : [...this.scopes, this.location];
    }

    getExecution(action: Action<Context>): Readonly<ActionExecution> {
        return { ...(this.executions.get(action) ?? { status: 'idle' }) };
    }

    exitToRoot(): void {
        this.root.children.forEach((child) => this.clear(child, false));
        this.scopes = [this.root];
        this.location = this.root;
        this.executions.clear();
        this.checked.clear();
        this.inspections.clear();
    }

    exitToParent(): void {
        if (this.scopes.length === 1) return;
        const exited = this.scopes.pop()!;
        this.location = this.closure;
        this.clear(exited, false);
    }

    private clear(node: Closure<Context> | Action<Context>, parentExited: boolean): void {
        this.checked.delete(node);
        this.results.delete(node);
        this.inspections.delete(node);
        if (node instanceof Closure) this.prepared.delete(node);
        node.condition.reset?.(this, parentExited);
        if (node instanceof Action) this.executions.delete(node);
        else node.children.forEach((child) => this.clear(child, true));
    }

    check(node: Closure<Context> | Action<Context>, context: Context): boolean {
        const parent = this.parents.get(node);
        if (parent && !this.scopes.includes(parent)) return false;
        if (node instanceof Action && this.executions.has(node)) return false;
        if (this.results.has(node)) return this.results.get(node)!;
        let matches: boolean | undefined;
        if (node instanceof Closure && this.prepareContext) {
            context = this.prepareContext(node, context, (candidate) => (matches = node.condition(candidate, this)));
            if (matches !== undefined) this.prepared.set(node, context);
        }
        matches ??= node.condition(context, this);
        this.results.set(node, matches);
        this.checked.set(node, { matches, conditions: (node.condition.describe?.(context, this, matches) ?? [{
            description: 'Custom condition (last checked)', conditionFit: matches, inputMissing: false,
        }]).map(copyCondition) });
        const inspection = node.metadata.inspect?.(context, this);
        if (inspection) this.inspections.set(node, { ...inspection, fields: { ...inspection.fields } });
        return matches;
    }

    enter(node: Closure<Context> | Action<Context>): void {
        this.location = node;
        if (node instanceof Closure) this.scopes.push(node);
    }

    /** Copy recorded results and static descriptions; inspection never measures conditions. */
    snapshot(_context?: Context): NodeSnapshot {
        const path = this.path;
        const visit = (node: Closure<Context> | Action<Context>, id: string): NodeSnapshot => {
            const inspection = this.inspections.get(node);
            const checked = this.checked.get(node);
            const conditions = checked?.conditions ?? node.condition.unevaluated ?? [{
                description: 'Custom condition (last checked)',
                conditionFit: null,
                inputMissing: false,
            }];
            const execution = node instanceof Action ? this.getExecution(node) : undefined;
            const onPath = path.includes(node);
            const status = execution && execution.status !== 'idle'
                ? { running: 'Running', completed: 'Completed', failed: 'Failed' }[execution.status]
                : onPath ? 'Active' : !checked ? 'Not evaluated'
                    : inspection?.status ?? (checked.matches ? 'Waiting' : 'Not matched');
            const nodeId = node.metadata.id ?? id;
            return {
                id: nodeId, kind: node instanceof Closure ? 'closure' : 'action',
                name: node.name, description: node.description, current: node === this.current, onPath,
                status,
                conditions: conditions.map(copyCondition), fields: {
                    ...(node instanceof Closure ? { 'Preparation time': `${node.preparationTimeMs / 1000} s` } : {}),
                    ...node.metadata.fields, ...inspection?.fields,
                },
                ...(execution ? { execution } : {}),
                children: node instanceof Closure ? node.children.map((child, index) => visit(child, `${nodeId}/${index}`)) : [],
            };
        };
        return visit(this.root, 'root');
    }

    step(context: Context, onActionError?: (error: unknown) => void): unknown {
        if (this.pending) return;
        this.results.clear();
        this.prepared.clear();
        return this.root.evaluate(context, this, onActionError);
    }

    execute(next: Action<Context>, context: Context, onActionError?: (error: unknown) => void): unknown {
        if (this.pending || this.current !== next || this.executions.has(next)) return;
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
