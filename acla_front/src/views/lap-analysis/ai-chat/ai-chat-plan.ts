export const PROCEDURE_PLAN_STEP_STATUSES = [
    'pending',
    'running',
    'complete',
    'blocked',
    'failed',
    'skipped',
] as const;

export type ProcedurePlanStepStatus = typeof PROCEDURE_PLAN_STEP_STATUSES[number];

export type ProcedurePlanRequest = {
    type: string;
    title: string;
    subscriber: string;
    status: ProcedurePlanStepStatus;
    detail?: string;
    payload?: unknown;
};

export type ProcedurePlan = {
    goal: string;
    requests: ProcedurePlanRequest[];
    currentStep: number;
    sourceEvent?: string;
};

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

const isProcedurePlanStepStatus = (value: unknown): value is ProcedurePlanStepStatus => (
    typeof value === 'string'
    && (PROCEDURE_PLAN_STEP_STATUSES as readonly string[]).includes(value)
);

const buildProcedurePlanRequest = (value: unknown): ProcedurePlanRequest | null => {
    const request = toRecord(value);
    if (!request) return null;

    const type = toNonEmptyString(request.type);
    const title = toNonEmptyString(request.title);
    const subscriber = toNonEmptyString(request.subscriber);
    if (!type || !title || !subscriber) return null;

    return {
        type,
        title,
        subscriber,
        status: isProcedurePlanStepStatus(request.status) ? request.status : 'pending',
        detail: toNonEmptyString(request.detail) || undefined,
        payload: request.payload,
    };
};

const toProcedurePlanRequests = (value: unknown): ProcedurePlanRequest[] | null => {
    if (!Array.isArray(value) || value.length === 0) return null;
    const requests = value.map(buildProcedurePlanRequest);
    if (requests.some((item) => !item)) return null;
    return requests as ProcedurePlanRequest[];
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

export const buildProcedurePlan = (data: Record<string, unknown>): ProcedurePlan | null => {
    const sourceEvent = toNonEmptyString(data.event);
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
