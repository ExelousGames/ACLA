export type LiveCoachingPlan = {
    goal: string;
    steps: string[];
    currentStep: number;
    focusName?: string;
    sourceEvent?: string;
};

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

const buildInferredLivePlanSteps = (sectionName: string, primaryIssue: string | null): string[] => {
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

const getFocusPrimaryIssue = (focus: Record<string, any> | null): string | null => {
    const baseline = focus?.baseline && typeof focus.baseline === 'object'
        ? focus.baseline as Record<string, unknown>
        : null;
    const childLabels = toStringArray(baseline?.childLabels);

    return childLabels[0]
        || toNonEmptyString(baseline?.parentLabel)
        || null;
};

export const buildLiveCoachingPlan = (data: Record<string, unknown>): LiveCoachingPlan | null => {
    const sourceEvent = toNonEmptyString(data.event);
    if (
        sourceEvent !== 'recorded_analysis_plan_ready'
        && sourceEvent !== 'coaching_window'
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
            ? buildInferredLivePlanSteps(focusName, primaryIssue)
            : [];
    if (inferredSteps.length === 0) return null;

    const requestedStep = Math.floor(Number(data.current_step ?? 0));
    const currentStep = Number.isFinite(requestedStep)
        ? Math.max(0, Math.min(inferredSteps.length - 1, requestedStep))
        : 0;
    const inferredGoal = focusName
        ? primaryIssue
            ? `Improve ${focusName} by reducing ${primaryIssue}.`
            : `Improve ${focusName} with a focused live coaching pass.`
        : 'Live coaching plan';

    return {
        goal: toNonEmptyString(data.goal) || inferredGoal,
        steps: inferredSteps,
        currentStep,
        focusName: focusName || undefined,
        sourceEvent,
    };
};
