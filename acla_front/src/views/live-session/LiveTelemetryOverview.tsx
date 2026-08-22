import React, { forwardRef, useImperativeHandle, useMemo, useState } from 'react';
import { Box, Grid, Text, TextField } from '@radix-ui/themes';
import { MagnifyingGlassIcon } from '@radix-ui/react-icons';
import { LiveTelemetry } from './live-session-types';
import { NamedAiToolComponentHandle, useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import { runVisualizationBooleanCallback } from 'views/lap-analysis/visualization/visualization-component-callbacks';
import { ComponentDisableFailedError, VisualizationUpdateFailedError } from 'contexts/AiToolComponentError';
import { useCurrentTelemetry } from './live-telemetry-store';

export interface LiveTelemetryOverviewHandle extends NamedAiToolComponentHandle {
    updateLiveTelemetry(data: LiveTelemetry): true;
    disableLiveTelemetry(): true;
}

interface LiveTelemetryOverviewProps {
    name: string;
    telemetry?: LiveTelemetry;
    onUpdate?: (data: LiveTelemetry) => boolean;
    onDisable?: () => boolean;
}

const LiveTelemetryOverview = forwardRef<LiveTelemetryOverviewHandle, LiveTelemetryOverviewProps>(({
    name,
    telemetry,
    onUpdate,
    onDisable,
}, forwardedRef) => {
    const currentTelemetry = useCurrentTelemetry();
    const displayedTelemetry = telemetry ?? currentTelemetry;
    const [search, setSearch] = useState('');
    const handle = useMemo<LiveTelemetryOverviewHandle>(() => ({
        getComponentName: () => name,
        updateLiveTelemetry: (data) => runVisualizationBooleanCallback(
            name,
            VisualizationUpdateFailedError,
            `Failed to update chart '${name}'.`,
            onUpdate ? () => onUpdate(data) : undefined,
        ),
        disableLiveTelemetry: () => runVisualizationBooleanCallback(
            name,
            ComponentDisableFailedError,
            `Component '${name}' could not be disabled.`,
            onDisable,
        ),
    }), [name, onDisable, onUpdate]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandleRef = React.useRef(handle);
    registeredHandleRef.current = handle;
    useRegisterAiToolComponentRef(registeredHandleRef);
    const entries = useMemo(() => {
        const term = search.trim().toLowerCase();
        return Object.entries(displayedTelemetry)
            .filter(([key]) => !term || key.toLowerCase().includes(term))
            .sort(([left], [right]) => left.localeCompare(right));
    }, [displayedTelemetry, search]);

    return (
        <Box className="live-optional-panel">
            <TextField.Root placeholder="Search live features..." value={search} onChange={(event) => setSearch(event.target.value)}>
                <TextField.Slot><MagnifyingGlassIcon /></TextField.Slot>
            </TextField.Root>
            <Box className="live-optional-panel__scroll">
                {entries.length === 0 ? <Text color="gray">No current telemetry available</Text> : (
                    <Grid columns="2" gap="3">
                        {entries.map(([key, value]) => (
                            <Box key={key}>
                                <Text size="1" color="gray">{key}</Text>
                                <Text size="2" weight="bold" as="div" className="live-optional-panel__value">
                                    {typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)}
                                </Text>
                            </Box>
                        ))}
                    </Grid>
                )}
            </Box>
        </Box>
    );
});

LiveTelemetryOverview.displayName = 'LiveTelemetryOverview';

export default LiveTelemetryOverview;
