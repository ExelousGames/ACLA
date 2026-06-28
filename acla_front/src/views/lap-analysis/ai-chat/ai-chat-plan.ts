export type ProcedurePlan = {
    goal: string;
    steps: string[];
    currentStep: number;
    focusName?: string;
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

const toStringArray = (value: unknown): string[] => (
    Array.isArray(value)
        ? value.map(toNonEmptyString).filter((item): item is string => Boolean(item))
        : []
);

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

const buildInferredLiveProcedureSteps = (sectionName: string, primaryIssue: string | null): string[] => {
    if (primaryIssue) {
        return [
            `Use the next approach to ${sectionName} as the focus run.`,
            `Change one thing first: clean up ${primaryIssue}.`,
            'After the next pass, compare the focused section classification against this baseline.',
        ];
    }

    return [
        `Use the next approach to ${sectionName} as the focus run.`,
        'Make one clean, repeatable adjustment through the focused section.',
        'After the next pass, compare the focused section classification against this baseline.',
    ];
};

const buildLiveAnalysisStartupPlan = (snapshot: Record<string, any> | null): ProcedurePlan => ({
    goal: 'Run live analysis from a clean baseline.',
    steps: [
        'Collect a complete baseline lap.',
        'Analyze the baseline with recorded-session data or live section classification.',
        'Select the focus section and compare the next pass against the baseline.',
    ],
    currentStep: snapshot?.baseline_ready ? 1 : 0,
    sourceEvent: 'live_analysis_plan_started',
});

const buildBaselineReadyPlan = (): ProcedurePlan => ({
    goal: 'Analyze the completed baseline and choose the focus section.',
    steps: [
        'Collect a complete baseline lap.',
        'Classify baseline sections from the completed lap.',
        'Select the focus section and compare the next pass against the baseline.',
    ],
    currentStep: 1,
    sourceEvent: 'live_baseline_ready_for_classification',
});

const getFocusPrimaryIssue = (focus: Record<string, any> | null): string | null => {
    const baseline = focus?.baseline && typeof focus.baseline === 'object'
        ? focus.baseline as Record<string, unknown>
        : null;
    const childLabels = toStringArray(baseline?.childLabels);

    return childLabels[0]
        || toNonEmptyString(baseline?.parentLabel)
        || null;
};

export const buildLiveProcedurePlan = (data: Record<string, unknown>): ProcedurePlan | null => {
    const sourceEvent = toNonEmptyString(data.event);
    const snapshot = data.snapshot && typeof data.snapshot === 'object'
        ? data.snapshot as Record<string, any>
        : null;
    if (sourceEvent === 'live_analysis_plan_started') {
        return buildLiveAnalysisStartupPlan(snapshot);
    }
    if (sourceEvent === 'live_baseline_ready_for_classification') {
        return buildBaselineReadyPlan();
    }
    if (
        sourceEvent !== 'recorded_analysis_plan_ready'
        && sourceEvent !== 'live_analysis_window'
    ) return null;

    const focus = data.focus && typeof data.focus === 'object'
        ? data.focus as Record<string, any>
        : null;
    const focusName = toNonEmptyString(focus?.section?.name);
    const primaryIssue = getFocusPrimaryIssue(focus);

    const steps = toStringArray(data.plan);
    const inferredSteps = steps.length > 0
        ? steps
        : focusName
            ? buildInferredLiveProcedureSteps(focusName, primaryIssue)
            : [];
    if (inferredSteps.length === 0) return null;

    const requestedStep = Math.floor(Number(data.current_step ?? 0));
    const currentStep = Number.isFinite(requestedStep)
        ? Math.max(0, Math.min(inferredSteps.length - 1, requestedStep))
        : 0;
    const inferredGoal = focusName
        ? primaryIssue
            ? `Improve ${focusName} by reducing ${primaryIssue}.`
            : `Improve ${focusName} with a focused live analysis pass.`
        : 'Live procedure plan';

    return {
        goal: toNonEmptyString(data.goal) || inferredGoal,
        steps: inferredSteps,
        currentStep,
        focusName: focusName || undefined,
        sourceEvent,
    };
};
