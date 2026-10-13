import { useMemo, useState } from 'react';
import { Text, TextField } from '@radix-ui/themes';
import { MagnifyingGlassIcon } from '@radix-ui/react-icons';

interface TelemetryOverviewProps {
    data?: Readonly<Record<string, unknown>>;
    sampleIndex?: number;
    sampleCount: number;
}

const TelemetryOverview = ({ data, sampleIndex, sampleCount }: TelemetryOverviewProps) => {
    const [searchTerm, setSearchTerm] = useState('');
    const filteredData = useMemo(() => {
        const entries = Object.entries(data || {});
        const lowerSearch = searchTerm.trim().toLowerCase();
        if (!lowerSearch) return entries;

        return entries
            .filter(([key]) => key.toLowerCase().includes(lowerSearch))
            .sort(([keyA], [keyB]) => {
                const aStartsWith = keyA.toLowerCase().startsWith(lowerSearch);
                const bStartsWith = keyB.toLowerCase().startsWith(lowerSearch);
                if (aStartsWith && !bStartsWith) return -1;
                if (!aStartsWith && bStartsWith) return 1;
                return keyA.localeCompare(keyB);
            });
    }, [data, searchTerm]);

    return (
        <section className="recorded-telemetry-overview" aria-label="Telemetry Overview">
            <div className="recorded-telemetry-overview__header">
                <Text size="3" weight="bold">Telemetry Overview</Text>
                <Text size="1" color="gray" as="div">
                    {data && sampleIndex !== undefined
                        ? `Sample ${(sampleIndex + 1).toLocaleString()} of ${sampleCount.toLocaleString()} · ${Object.keys(data).length} fields`
                        : 'Values follow trajectory playback'}
                </Text>
                <TextField.Root
                    aria-label="Search telemetry features"
                    placeholder="Search features..."
                    value={searchTerm}
                    onChange={(event) => setSearchTerm(event.target.value)}
                >
                    <TextField.Slot>
                        <MagnifyingGlassIcon height="16" width="16" />
                    </TextField.Slot>
                </TextField.Root>
            </div>
            <div className="recorded-telemetry-overview__values">
                {filteredData.length > 0 ? (
                    <dl className="recorded-telemetry-overview__fields">
                        {filteredData.map(([key, value]) => {
                            const displayValue = typeof value === 'boolean'
                                ? value ? 'Yes' : 'No'
                                : typeof value === 'object' && value !== null
                                    ? JSON.stringify(value)
                                    : String(value);

                            return (
                                <div key={key} className="recorded-telemetry-overview__field">
                                    <dt>{key}</dt>
                                    <dd>{displayValue}</dd>
                                </div>
                            );
                        })}
                    </dl>
                ) : (
                    <Text size="2" color="gray">
                        {data ? 'No matching telemetry features found' : 'No telemetry data available at this playback position'}
                    </Text>
                )}
            </div>
        </section>
    );
};

export default TelemetryOverview;
