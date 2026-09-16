import { useEffect, useState } from 'react';
import { createDataStructure, DataStructure, includeDataStructure } from './data-structure';
import './data-structure.css';

type DataStructurePreviewProps = {
    readRecords: (onChunk: (records: unknown[]) => void) => Promise<unknown>;
};

export function DataStructureSection({ title, description, readRecords }: DataStructurePreviewProps & { title: string; description: string }) {
    const [open, setOpen] = useState(false);
    return (
        <details className="data-structure" aria-label={title} open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
            <summary>{title}</summary>
            {open && (
                <>
                    <p className="data-structure__message">{description}</p>
                    <DataStructurePreview readRecords={readRecords} />
                </>
            )}
        </details>
    );
}

function StructureNode({ name, structure }: { name: string; structure: DataStructure }) {
    const [open, setOpen] = useState(false);
    const fields = Array.from(structure.fields.entries());
    const expandable = fields.length > 0 || Boolean(structure.items);
    const label = (
        <>
            <code className="data-structure__name">{name}</code>
            <span className="data-structure__type">{Array.from(structure.types).sort().join(' | ')}</span>
            {structure.types.has('object') && (
                <span className="data-structure__count">{fields.length} {fields.length === 1 ? 'field' : 'fields'}</span>
            )}
            {structure.types.has('array') && !structure.items && (
                <span className="data-structure__count">empty</span>
            )}
        </>
    );

    if (!expandable) return <div className="data-structure__leaf">{label}</div>;

    return (
        <details className="data-structure__node" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
            <summary>{label}</summary>
            {open && (
                <div className="data-structure__children">
                    {fields.map(([field, child]) => <StructureNode key={field} name={field} structure={child} />)}
                    {structure.items && <StructureNode name="[items]" structure={structure.items} />}
                </div>
            )}
        </details>
    );
}

export default function DataStructurePreview({ readRecords }: DataStructurePreviewProps) {
    const [result, setResult] = useState<{ structure: DataStructure; count: number } | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [attempt, setAttempt] = useState(0);

    useEffect(() => {
        let cancelled = false;
        setResult(null);
        setError(null);
        const structure = createDataStructure();
        let count = 0;
        const read = async () => {
            try {
                await readRecords((records) => {
                    // Throwing stops the existing streaming reader after this panel closes.
                    if (cancelled) throw new Error('Data structure read cancelled.');
                    records.forEach((record) => includeDataStructure(structure, record));
                    count += records.length;
                });
                if (!cancelled) setResult({ structure, count });
            } catch (readError) {
                if (!cancelled) setError(readError instanceof Error ? readError.message : String(readError));
            }
        };
        void read();
        return () => { cancelled = true; };
    }, [readRecords, attempt]);

    if (error) {
        return (
            <div className="data-structure__message">
                <p role="alert">Unable to read the data structure: {error}</p>
                <button type="button" className="data-structure__retry" onClick={() => setAttempt(value => value + 1)}>Retry reading fields</button>
            </div>
        );
    }
    if (!result) return <p className="data-structure__message" role="status">Reading recorded fields…</p>;
    if (result.count === 0) return <p className="data-structure__message">No recorded fields available.</p>;

    return (
        <>
            <p className="data-structure__message">Structure from {result.count.toLocaleString()} {result.count === 1 ? 'record' : 'records'}. Array items combine all observed fields and types.</p>
            <div className="data-structure__tree" role="region" aria-label="Recorded data structure" tabIndex={0}>
                <StructureNode key={attempt} name="Record" structure={result.structure} />
            </div>
        </>
    );
}
