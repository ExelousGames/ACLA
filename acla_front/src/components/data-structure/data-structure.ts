export interface DataStructure {
    types: Set<string>;
    fields: Map<string, DataStructure>;
    items?: DataStructure;
}

export const createDataStructure = (): DataStructure => ({
    types: new Set(),
    fields: new Map(),
});

// Merge shapes across all records without retaining the records themselves.
export function includeDataStructure(structure: DataStructure, value: unknown): void {
    const type = value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;
    structure.types.add(type);

    if (Array.isArray(value)) {
        for (const item of value) {
            structure.items ??= createDataStructure();
            includeDataStructure(structure.items, item);
        }
    } else if (value !== null && typeof value === 'object') {
        for (const [name, child] of Object.entries(value)) {
            let field = structure.fields.get(name);
            if (!field) {
                field = createDataStructure();
                structure.fields.set(name, field);
            }
            includeDataStructure(field, child);
        }
    }
}
