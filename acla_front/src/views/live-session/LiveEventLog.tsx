import React, { useMemo, useState } from 'react';
import { Badge, Box, Flex, Table, Text, TextField } from '@radix-ui/themes';
import { MagnifyingGlassIcon } from '@radix-ui/react-icons';
import { EventType, SessionEvent } from 'views/lap-analysis/session-intelligence/types';

const EVENT_COLORS: Record<EventType, 'blue' | 'green' | 'red' | 'amber'> = {
    CORNER: 'blue',
    STRAIGHT: 'green',
    CRASHED: 'red',
    OVERTAKE: 'amber',
};

const LiveEventLog = ({ events }: { events: SessionEvent[] }) => {
    const [search, setSearch] = useState('');
    const filtered = useMemo(() => {
        const term = search.trim().toLowerCase();
        return events.filter((event) => !term || JSON.stringify(event).toLowerCase().includes(term)).slice().reverse();
    }, [events, search]);

    return (
        <Box className="live-optional-panel">
            <Flex justify="between" align="center" gap="2">
                <Text size="1" color="gray">{events.length} detected events</Text>
                <TextField.Root placeholder="Search events..." value={search} onChange={(event) => setSearch(event.target.value)}>
                    <TextField.Slot><MagnifyingGlassIcon /></TextField.Slot>
                </TextField.Root>
            </Flex>
            <Box className="live-optional-panel__scroll">
                {filtered.length === 0 ? <Text color="gray">No live events detected yet</Text> : (
                    <Table.Root size="1">
                        <Table.Header><Table.Row><Table.ColumnHeaderCell>Time</Table.ColumnHeaderCell><Table.ColumnHeaderCell>Type</Table.ColumnHeaderCell><Table.ColumnHeaderCell>Lap</Table.ColumnHeaderCell></Table.Row></Table.Header>
                        <Table.Body>
                            {filtered.map((event) => (
                                <Table.Row key={event.id}>
                                    <Table.Cell>{new Date(event.timestamp).toLocaleTimeString()}</Table.Cell>
                                    <Table.Cell><Badge color={EVENT_COLORS[event.type]}>{event.type}</Badge></Table.Cell>
                                    <Table.Cell>{event.lap}</Table.Cell>
                                </Table.Row>
                            ))}
                        </Table.Body>
                    </Table.Root>
                )}
            </Box>
        </Box>
    );
};

export default LiveEventLog;
