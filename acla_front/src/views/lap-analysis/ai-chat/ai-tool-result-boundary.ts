const UI_OUTPUT_KEYS = new Set([
    'ui_output',
    'uiOutput',
    'ui_ouptut',
    'uiOuptut',
]);

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

export const isToolOutputEnvelopeLike = (value: unknown): value is Record<string, unknown> => {
    if (!isRecord(value)) return false;
    return (
        typeof value.tool_name === 'string'
        && typeof value.run_id === 'string'
        && typeof value.status === 'string'
        && typeof value.final === 'boolean'
        && 'ui_output' in value
        && 'ai_output' in value
    );
};

export const assertNoUiOutputForAi = (value: unknown): void => {
    if (Array.isArray(value)) {
        value.forEach(assertNoUiOutputForAi);
        return;
    }
    if (!isRecord(value)) {
        return;
    }

    Object.entries(value).forEach(([key, item]) => {
        if (UI_OUTPUT_KEYS.has(key)) {
            throw new Error(`AI tool_result frames must not contain ${key}.`);
        }
        assertNoUiOutputForAi(item);
    });
};

export const getAiToolResult = (result: unknown): unknown => {
    if (isToolOutputEnvelopeLike(result)) {
        assertNoUiOutputForAi(result.ai_output);
        return result.ai_output;
    }
    assertNoUiOutputForAi(result);
    return result;
};
