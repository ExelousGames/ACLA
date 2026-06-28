export type ProcedurePlanRequest = {
    type: string;
    title: string;
    detail?: string;
    name?: string;
    method?: string;
    url?: string;
    payload?: unknown;
};

export type ProcedurePlan = {
    goal: string;
    requests: ProcedurePlanRequest[];
    currentStep: number;
    focusName?: string;
    sourceEvent?: string;
};

export type ProcedurePlanAdvanceBlock = {
    status: 'blocked' | 'unavailable';
    error: string;
    message: string;
};

const LIVE_PROCEDURE_PLAN_EVENTS = new Set([
    'live_analysis_plan_started',
    'live_baseline_ready_for_classification',
    'recorded_analysis_plan_ready',
    'live_analysis_window',
]);

export const isProcedurePlanStartEvent = (sourceEvent?: string): boolean => (
    typeof sourceEvent === 'string'
    && (
        sourceEvent === 'procedure_plan_started'
        || sourceEvent.endsWith('_plan_started')
    )
);

const toNonEmptyString = (value: unknown): string | null => {
    if (typeof value !== 'string') return null;
    const trimmed = value.trim();
    return trimmed || null;
};

const toRecord = (value: unknown): Record<string, unknown> | null => (
    value && typeof value === 'object' && !Array.isArray(value)
        ? value as Record<string, unknown>
        : null
);

const requestTitleFromRecord = (request: Record<string, unknown>): string | null => (
    toNonEmptyString(request.title)
    || toNonEmptyString(request.summary)
    || toNonEmptyString(request.description)
    || toNonEmptyString(request.name)
    || toNonEmptyString(request.tool)
    || toNonEmptyString(request.tool_name)
    || toNonEmptyString(request.action)
    || toNonEmptyString(request.endpoint)
    || toNonEmptyString(request.url)
);

const buildProcedurePlanRequest = (value: unknown): ProcedurePlanRequest | null => {
    const text = toNonEmptyString(value);
    if (text) {
        return {
            type: 'request',
            title: text,
        };
    }

    const request = toRecord(value);
    if (!request) return null;

    const title = requestTitleFromRecord(request);
    if (!title) return null;

    return {
        type: (
            toNonEmptyString(request.type)
            || toNonEmptyString(request.kind)
            || toNonEmptyString(request.request_type)
            || 'request'
        ),
        title,
        detail: (
            toNonEmptyString(request.detail)
            || toNonEmptyString(request.message)
            || toNonEmptyString(request.reason)
        ) || undefined,
        name: (
            toNonEmptyString(request.name)
            || toNonEmptyString(request.tool)
            || toNonEmptyString(request.tool_name)
        ) || undefined,
        method: toNonEmptyString(request.method)?.toUpperCase(),
        url: (
            toNonEmptyString(request.url)
            || toNonEmptyString(request.endpoint)
        ) || undefined,
        payload: request.payload ?? request.body ?? request.arguments ?? request.args ?? request.params,
    };
};

const toProcedurePlanRequests = (value: unknown): ProcedurePlanRequest[] => (
    Array.isArray(value)
        ? value.map(buildProcedurePlanRequest).filter((item): item is ProcedurePlanRequest => Boolean(item))
        : []
);

const isLiveProcedurePlan = (plan: ProcedurePlan): boolean => (
    typeof plan.sourceEvent === 'string'
    && LIVE_PROCEDURE_PLAN_EVENTS.has(plan.sourceEvent)
);

export const getProcedurePlanAdvanceBlock = (
    plan: ProcedurePlan | null,
    snapshot?: Record<string, any> | null,
    hasFocus = false,
): ProcedurePlanAdvanceBlock | null => {
    if (!plan) {
        return {
            status: 'unavailable',
            error: 'no_procedure_plan',
            message: 'No active procedure plan is available.',
        };
    }

    if (!isLiveProcedurePlan(plan)) return null;

    const targetStep = Math.min(plan.currentStep + 1, plan.requests.length - 1);
    if (targetStep === plan.currentStep) return null;

    if (targetStep >= 1 && snapshot?.baseline_ready !== true) {
        return {
            status: 'blocked',
            error: 'baseline_collection_incomplete',
            message: 'Complete one clean baseline lap before advancing the plan.',
        };
    }

    if (targetStep >= 2 && !hasFocus) {
        return {
            status: 'blocked',
            error: 'focus_section_not_ready',
            message: 'Analyze the completed baseline and select a focus section before advancing the plan.',
        };
    }

    return null;
};

export const isProcedurePlanOptOutRequest = (text: unknown): boolean => {
    if (typeof text !== 'string') return false;
    const normalized = text
        .toLowerCase()
        .replace(/['\u2019]/g, '')
        .replace(/[^a-z0-9\s]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    if (!normalized) return false;

    const planTarget = '(?:the\\s+)?(?:procedure\\s+)?plan';
    const optOutVerb = '(?:cancel|clear|stop|end|exit|dismiss|hide|skip|drop|forget)';
    return [
        new RegExp(`\\b${optOutVerb}\\b.*\\b${planTarget}\\b`),
        new RegExp(`\\b${planTarget}\\b.*\\b${optOutVerb}\\b`),
        new RegExp(`\\bopt\\s*out\\b.*\\b${planTarget}\\b`),
        new RegExp(`\\b(?:dont|do\\s+not|no\\s+longer)\\s+(?:follow|use|respect)\\s+${planTarget}\\b`),
        new RegExp(`\\bignore\\s+${planTarget}\\b`),
    ].some((pattern) => pattern.test(normalized));
};

export const buildLiveProcedurePlan = (data: Record<string, unknown>): ProcedurePlan | null => {
    const sourceEvent = toNonEmptyString(data.event);
    if (
        !isProcedurePlanStartEvent(sourceEvent || undefined)
        && sourceEvent !== 'live_analysis_plan_started'
        && sourceEvent !== 'live_baseline_ready_for_classification'
        && sourceEvent !== 'recorded_analysis_plan_ready'
        && sourceEvent !== 'live_analysis_window'
    ) return null;

    const focus = data.focus && typeof data.focus === 'object'
        ? data.focus as Record<string, any>
        : null;
    const focusName = toNonEmptyString(data.focus_name) || toNonEmptyString(focus?.section?.name);

    const explicitRequests = toProcedurePlanRequests(data.requests);
    const planRequests = toProcedurePlanRequests(data.plan);
    const stepRequests = toProcedurePlanRequests(data.steps);
    const requests = explicitRequests.length > 0
        ? explicitRequests
        : planRequests.length > 0
            ? planRequests
            : stepRequests;
    if (requests.length === 0) return null;

    const requestedStep = Math.floor(Number(data.current_request ?? data.current_step ?? 0));
    const currentStep = Number.isFinite(requestedStep)
        ? Math.max(0, Math.min(requests.length - 1, requestedStep))
        : 0;

    return {
        goal: toNonEmptyString(data.goal) || toNonEmptyString(data.title) || requests[0].title,
        requests,
        currentStep,
        focusName: focusName || undefined,
        sourceEvent: sourceEvent || undefined,
    };
};
