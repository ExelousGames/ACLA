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
    sourceEvent?: string;
};

export type ProcedurePlanAdvanceBlock = {
    status: 'blocked' | 'unavailable';
    error: string;
    message: string;
};

const LIVE_PROCEDURE_PLAN_EVENTS = new Set([
    'live_analysis_plan_started',
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

const buildProcedurePlanRequest = (value: unknown): ProcedurePlanRequest | null => {
    const request = toRecord(value);
    if (!request) return null;

    const type = toNonEmptyString(request.type);
    const title = toNonEmptyString(request.title);
    if (!type || !title) return null;

    return {
        type,
        title,
        detail: toNonEmptyString(request.detail) || undefined,
        name: toNonEmptyString(request.name) || undefined,
        method: toNonEmptyString(request.method)?.toUpperCase(),
        url: toNonEmptyString(request.url) || undefined,
        payload: request.payload,
    };
};

const toProcedurePlanRequests = (value: unknown): ProcedurePlanRequest[] | null => {
    if (!Array.isArray(value) || value.length === 0) return null;
    const requests = value.map(buildProcedurePlanRequest);
    if (requests.some((item) => !item)) return null;
    return requests as ProcedurePlanRequest[];
};

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
        && sourceEvent !== 'recorded_analysis_plan_ready'
        && sourceEvent !== 'live_analysis_window'
    ) return null;

    const requests = toProcedurePlanRequests(data.requests);
    if (!requests) return null;

    const requestedStep = Math.floor(Number(data.current_request ?? 0));
    const currentStep = Number.isFinite(requestedStep)
        ? Math.max(0, Math.min(requests.length - 1, requestedStep))
        : 0;

    return {
        goal: toNonEmptyString(data.goal) || requests[0].title,
        requests,
        currentStep,
        sourceEvent: sourceEvent || undefined,
    };
};
