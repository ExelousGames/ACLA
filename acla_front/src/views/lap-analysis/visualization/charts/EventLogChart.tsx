import React, { useContext, useMemo, useState } from 'react';
import { Badge, Box, Card, Flex, ScrollArea, Table, Text, TextField } from '@radix-ui/themes';
import { MagnifyingGlassIcon } from '@radix-ui/react-icons';
import { AnalysisContext } from '../../analysis-context';
import { VisualizationProps } from '../VisualizationRegistry';
import { EventType, SessionEvent } from '../../session-intelligence/types';

const TYPE_COLOR: Record<EventType, 'blue' | 'green' | 'red' | 'amber'> = {
    CORNER: 'blue',
    STRAIGHT: 'green',
    CRASHED: 'red',
    OVERTAKE: 'amber',
};

const formatPosition = (position: number): string => {
    if (typeof position !== 'number' || Number.isNaN(position)) return '-';
    return `${(position * 100).toFixed(1)}%`;
};

const formatTimestamp = (timestamp: number): string => {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
};

const formatMetadata = (metadata?: Record<string, any>): string => {
    if (!metadata || Object.keys(metadata).length === 0) return '-';
    return Object.entries(metadata)
        .map(([key, value]) => {
            if (typeof value === 'number') {
                return `${key}: ${Number.isInteger(value) ? value : value.toFixed(2)}`;
            }
            return `${key}: ${value}`;
        })
        .join(', ');
};

const EventLogChart: React.FC<VisualizationProps> = ({ width = '100%', height = 320 }) => {
    const analysisContext = useContext(AnalysisContext);
    const [filter, setFilter] = useState<EventType | 'ALL'>('ALL');
    const [search, setSearch] = useState('');

    // Re-read on every liveData tick so newly emitted events appear in the table.
    const events: SessionEvent[] = useMemo(() => {
        return analysisContext.sessionIntelligence?.getAllEvents() ?? [];
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [analysisContext.sessionIntelligence, analysisContext.liveData]);

    const counts = useMemo(() => {
        const c: Record<EventType, number> = { CORNER: 0, STRAIGHT: 0, CRASHED: 0, OVERTAKE: 0 };
        events.forEach(e => { c[e.type] = (c[e.type] ?? 0) + 1; });
        return c;
    }, [events]);

    const filtered = useMemo(() => {
        const term = search.trim().toLowerCase();
        return events
            .filter(e => filter === 'ALL' || e.type === filter)
            .filter(e => {
                if (!term) return true;
                const haystack = `${e.type} ${e.lap} ${formatMetadata(e.metadata)}`.toLowerCase();
                return haystack.includes(term);
            })
            .slice()
            .reverse();
    }, [events, filter, search]);

    const filterOptions: Array<EventType | 'ALL'> = ['ALL', 'CORNER', 'STRAIGHT', 'CRASHED', 'OVERTAKE'];

    return (
        <Card style={{ width, height, padding: '16px', display: 'flex', flexDirection: 'column' }}>
            <Flex justify="between" align="center" mb="3" gap="3" wrap="wrap">
                <Box>
                    <Text size="3" weight="bold">Event Log</Text>
                    <Text size="1" color="gray" as="div">
                        {events.length} total • Corners {counts.CORNER} • Straights {counts.STRAIGHT} • Crashes {counts.CRASHED} • Overtakes {counts.OVERTAKE}
                    </Text>
                </Box>
                <TextField.Root
                    placeholder="Search events..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                >
                    <TextField.Slot>
                        <MagnifyingGlassIcon height="16" width="16" />
                    </TextField.Slot>
                </TextField.Root>
            </Flex>

            <Flex gap="2" mb="3" wrap="wrap">
                {filterOptions.map(option => {
                    const isActive = filter === option;
                    return (
                        <Badge
                            key={option}
                            color={option === 'ALL' ? 'gray' : TYPE_COLOR[option]}
                            variant={isActive ? 'solid' : 'soft'}
                            onClick={() => setFilter(option)}
                            style={{ cursor: 'pointer' }}
                        >
                            {option}
                        </Badge>
                    );
                })}
            </Flex>

            {filtered.length === 0 ? (
                <Box style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Text color="gray">No events logged yet</Text>
                </Box>
            ) : (
                <ScrollArea type="hover" style={{ flex: 1 }}>
                    <Table.Root size="1">
                        <Table.Header>
                            <Table.Row>
                                <Table.ColumnHeaderCell>Time</Table.ColumnHeaderCell>
                                <Table.ColumnHeaderCell>Type</Table.ColumnHeaderCell>
                                <Table.ColumnHeaderCell>Lap</Table.ColumnHeaderCell>
                                <Table.ColumnHeaderCell>Track Pos</Table.ColumnHeaderCell>
                                <Table.ColumnHeaderCell>Details</Table.ColumnHeaderCell>
                            </Table.Row>
                        </Table.Header>
                        <Table.Body>
                            {filtered.map(event => (
                                <Table.Row key={event.id}>
                                    <Table.Cell>{formatTimestamp(event.timestamp)}</Table.Cell>
                                    <Table.Cell>
                                        <Badge color={TYPE_COLOR[event.type]} variant="soft">
                                            {event.type}
                                        </Badge>
                                    </Table.Cell>
                                    <Table.Cell>{event.lap}</Table.Cell>
                                    <Table.Cell>{formatPosition(event.trackPosition)}</Table.Cell>
                                    <Table.Cell style={{ wordBreak: 'break-word' }}>
                                        {formatMetadata(event.metadata)}
                                    </Table.Cell>
                                </Table.Row>
                            ))}
                        </Table.Body>
                    </Table.Root>
                </ScrollArea>
            )}
        </Card>
    );
};

export default EventLogChart;
