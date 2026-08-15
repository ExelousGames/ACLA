const asRecord = (value: unknown): Record<string, unknown> => (
    value && typeof value === 'object' && !Array.isArray(value)
        ? value as Record<string, unknown>
        : {}
);

const omitRawTelemetryRows = (value: Record<string, unknown>): Record<string, unknown> => {
    if (!Array.isArray(value.telemetry_rows)) return value;
    const { telemetry_rows, ...rest } = value;
    return {
        ...rest,
        telemetry_row_count: typeof rest.telemetry_row_count === 'number'
            ? rest.telemetry_row_count
            : telemetry_rows.length,
    };
};

const getToolResultName = (data: Record<string, unknown>): string => {
    if (typeof data.name === 'string' && data.name.trim()) {
        return data.name.trim();
    }
    if (typeof data.tool_name === 'string' && data.tool_name.trim()) {
        return data.tool_name.trim();
    }
    if (typeof data.source === 'string' && data.source.trim()) {
        return data.source.trim();
    }
    if (typeof data.event === 'string' && data.event.trim()) {
        return data.event.trim();
    }
    return 'frontend_status_update';
};

const omitLiveAnalystAnalysis = (value: Record<string, unknown>): Record<string, unknown> => {
    if (
        value.source !== 'live_performance_analyst'
        && value.agent_mode !== 'live_performance_analyst'
    ) {
        return value;
    }
    if (!('analysis' in value)) return value;

    const { analysis: _analysis, ...rest } = value;
    return rest;
};

export const buildFormattedToolResultFrame = (
    data: Record<string, unknown>,
    fallbackId = `workflow-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
) => {
    const rawAiData = asRecord(data);
    const aiData = omitLiveAnalystAnalysis(omitRawTelemetryRows(rawAiData));
    const sourceStatus = typeof aiData.status === 'string' ? aiData.status : undefined;
    const id = typeof aiData.tool_run_id === 'string'
        ? aiData.tool_run_id
        : typeof data.run_id === 'string'
            ? data.run_id
            : fallbackId;
    const name = getToolResultName(aiData);
    const payload = {
        type: 'tool_result' as const,
        id,
        name,
        final: false,
        result: {
            ...aiData,
            ...(sourceStatus ? { source_status: sourceStatus } : {}),
            status: 'complete',
        },
    };

    return payload;
};
