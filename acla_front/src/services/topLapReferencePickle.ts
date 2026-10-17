// The backend stores TopLapEntry.to_components() as a protocol 4/5 pickle.
// Interpret only data and NumPy float64 arrays; never execute pickle globals.
export const decodeBase64Bytes = (value: string): Uint8Array => (
    Uint8Array.from(atob(value), (character) => character.charCodeAt(0))
);

const invalid = (): never => { throw new Error('Unsupported or invalid reference lap model data'); };
const MARK = Symbol('mark');
const GLOBAL = Symbol('global');
const DTYPE = Symbol('dtype');

const readArray = (bytes: unknown, dtype: any, shape: unknown, order: unknown): number[] | number[][] => {
    if (!(bytes instanceof Uint8Array) || dtype?.kind !== DTYPE
        || !Array.isArray(shape) || shape.length < 1 || shape.length > 2
        || !shape.every((size) => Number.isSafeInteger(size) && size >= 0)
        || (order !== 'C' && order !== 'F')) return invalid();
    const count = shape.reduce((product, size) => product * size, 1);
    if (!Number.isSafeInteger(count) || count * 8 !== bytes.length) return invalid();
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const at = (index: number) => view.getFloat64(index * 8, dtype.endian === '<');
    if (shape.length === 1) return Array.from({ length: shape[0] }, (_, index) => at(index));
    return Array.from({ length: shape[0] }, (_, row) => (
        Array.from({ length: shape[1] }, (_, column) => (
            at(order === 'F' ? column * shape[0] + row : row * shape[1] + column)
        ))
    ));
};

export const decodeTopLapPickle = (encoded: string): unknown => {
    const bytes = decodeBase64Bytes(encoded);
    const stack: any[] = [];
    const memo = new Map<number, any>();
    const decoder = new TextDecoder('utf-8', { fatal: true });
    let offset = 0;
    const take = (length: number): Uint8Array => {
        if (!Number.isSafeInteger(length) || length < 0 || offset + length > bytes.length) return invalid();
        const part = bytes.subarray(offset, offset + length);
        offset += length;
        return part;
    };
    const integer = (length: 1 | 2 | 4 | 8): number => {
        const part = take(length);
        const view = new DataView(part.buffer, part.byteOffset, length);
        const value = length === 1 ? part[0] : length === 2 ? view.getUint16(0, true)
            : length === 4 ? view.getUint32(0, true)
                : view.getUint32(0, true) + view.getUint32(4, true) * 4294967296;
        return Number.isSafeInteger(value) ? value : invalid();
    };
    const pop = (): any => stack.length ? stack.pop() : invalid();
    const top = (): any => stack.length ? stack[stack.length - 1] : invalid();
    const marked = (): any[] => {
        const index = stack.lastIndexOf(MARK);
        if (index < 0) return invalid();
        const values = stack.splice(index);
        return values.slice(1);
    };
    const setItems = (items: any[]) => {
        const target = top();
        if (!target || Object.getPrototypeOf(target) !== null || items.length % 2) return invalid();
        for (let index = 0; index < items.length; index += 2) {
            if (typeof items[index] !== 'string') return invalid();
            target[items[index]] = items[index + 1];
        }
    };

    while (offset < bytes.length) {
        const opcode = integer(1);
        switch (opcode) {
            case 0x80: { // PROTO
                const protocol = integer(1);
                if (protocol !== 4 && protocol !== 5) return invalid();
                break;
            }
            case 0x95: // FRAME (framing does not change the data stack)
                if (integer(8) > bytes.length - offset) return invalid();
                break;
            case 0x2e: // STOP
                if (stack.length !== 1 || offset !== bytes.length) return invalid();
                return pop();
            case 0x28: stack.push(MARK); break;
            case 0x4e: stack.push(null); break;
            case 0x88: stack.push(true); break;
            case 0x89: stack.push(false); break;
            case 0x4b: stack.push(integer(1)); break;
            case 0x4d: stack.push(integer(2)); break;
            case 0x4a: stack.push(integer(4) | 0); break;
            case 0x47: {
                const part = take(8);
                stack.push(new DataView(part.buffer, part.byteOffset, 8).getFloat64(0));
                break;
            }
            case 0x8c: stack.push(decoder.decode(take(integer(1)))); break;
            case 0x58: stack.push(decoder.decode(take(integer(4)))); break;
            case 0x8d: stack.push(decoder.decode(take(integer(8)))); break;
            case 0x43: stack.push(take(integer(1))); break;
            case 0x42: stack.push(take(integer(4))); break;
            case 0x8e:
            case 0x96: stack.push(take(integer(8))); break;
            case 0x7d: stack.push(Object.create(null)); break;
            case 0x5d:
            case 0x29: stack.push([]); break;
            case 0x74: stack.push(marked()); break;
            case 0x85:
            case 0x86:
            case 0x87: {
                const count = opcode - 0x84;
                if (stack.length < count) return invalid();
                stack.push(stack.splice(stack.length - count));
                break;
            }
            case 0x61: {
                const value = pop();
                if (!Array.isArray(top())) return invalid();
                top().push(value);
                break;
            }
            case 0x65: {
                const values = marked();
                const target = top();
                if (!Array.isArray(target)) return invalid();
                values.forEach((value) => target.push(value));
                break;
            }
            case 0x73: {
                const value = pop(), key = pop();
                setItems([key, value]);
                break;
            }
            case 0x75: setItems(marked()); break;
            case 0x94: memo.set(memo.size, top()); break;
            case 0x71: memo.set(integer(1), top()); break;
            case 0x72: memo.set(integer(4), top()); break;
            case 0x68:
            case 0x6a: {
                const key = integer(opcode === 0x68 ? 1 : 4);
                if (!memo.has(key)) return invalid();
                stack.push(memo.get(key));
                break;
            }
            case 0x93: { // STACK_GLOBAL: a fixed set of data constructors only
                const name = pop(), module = pop();
                const allowed = (module === 'numpy' && (name === 'dtype' || name === 'ndarray'))
                    || (/^numpy\.(?:_?core)\.numeric$/.test(module) && name === '_frombuffer')
                    || (/^numpy\.(?:_?core)\.multiarray$/.test(module) && name === '_reconstruct');
                if (!allowed) return invalid();
                stack.push({ kind: GLOBAL, name });
                break;
            }
            case 0x52: { // REDUCE: interpret the allowed constructors as data
                const args = pop(), constructor = pop();
                if (constructor?.kind !== GLOBAL || !Array.isArray(args)) return invalid();
                if (constructor.name === 'dtype') {
                    if (args[0] !== 'f8') return invalid();
                    stack.push({ kind: DTYPE, endian: '<' });
                } else if (constructor.name === '_frombuffer') {
                    stack.push(readArray(args[0], args[1], args[2], args[3]));
                } else if (constructor.name === '_reconstruct' && args[0]?.kind === GLOBAL
                    && args[0].name === 'ndarray') {
                    stack.push([]);
                } else return invalid();
                break;
            }
            case 0x62: { // BUILD: dtype state or a protocol 4 ndarray
                const state = pop(), target = top();
                if (!Array.isArray(state)) return invalid();
                if (target?.kind === DTYPE) {
                    if (state[1] !== '<' && state[1] !== '>') return invalid();
                    target.endian = state[1];
                } else if (Array.isArray(target) && target.length === 0 && state.length === 5) {
                    const values = readArray(state[4], state[2], state[1], state[3] ? 'F' : 'C');
                    values.forEach((value: number | number[]) => target.push(value));
                } else return invalid();
                break;
            }
            default: return invalid();
        }
    }
    return invalid();
};
