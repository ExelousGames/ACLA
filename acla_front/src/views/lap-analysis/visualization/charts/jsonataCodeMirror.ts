import {
    autocompletion,
    type Completion,
    type CompletionSource,
} from '@codemirror/autocomplete';
import {
    bracketMatching,
    StreamLanguage,
    type StreamParser,
    type StringStream,
} from '@codemirror/language';
import {
    linter,
    lintGutter,
    type Diagnostic,
} from '@codemirror/lint';
import type { Extension } from '@codemirror/state';
import { lineNumbers } from '@codemirror/view';
import {
    compileAnalysisResultsQuery,
    toAnalysisResultsQueryError,
    type AnalysisResultsQueryErrorDetail,
} from './analysisResultsQuery';

export const JSONATA_DIAGNOSTIC_DELAY_MS = 300;

export interface JsonataTokenizerState {
    blockComment: boolean;
}

const consumeBlockComment = (stream: StringStream, state: JsonataTokenizerState): void => {
    while (!stream.eol()) {
        if (stream.match('*/')) {
            state.blockComment = false;
            return;
        }
        stream.next();
    }
};

const consumeQuoted = (stream: StringStream, quote: string): void => {
    stream.next();
    let escaped = false;
    while (!stream.eol()) {
        const character = stream.next();
        if (escaped) {
            escaped = false;
        } else if (character === '\\') {
            escaped = true;
        } else if (character === quote) {
            return;
        }
    }
};

export const jsonataStreamParser: StreamParser<JsonataTokenizerState> = {
    name: 'jsonata',
    startState: () => ({ blockComment: false }),
    copyState: (state) => ({ ...state }),
    languageData: {
        closeBrackets: { brackets: ['(', '[', '{', "'", '"'] },
        commentTokens: { block: { open: '/*', close: '*/' }, line: '//' },
    },
    token(stream, state) {
        if (state.blockComment) {
            consumeBlockComment(stream, state);
            return 'comment';
        }
        if (stream.eatSpace()) return null;
        if (stream.match('/*')) {
            state.blockComment = true;
            consumeBlockComment(stream, state);
            return 'comment';
        }
        if (stream.match('//')) {
            stream.skipToEnd();
            return 'comment';
        }

        const next = stream.peek();
        if (next === '"' || next === "'") {
            consumeQuoted(stream, next);
            return 'string';
        }
        if (next === '`') {
            consumeQuoted(stream, next);
            return 'propertyName';
        }

        if (stream.match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/)) return 'number';
        if (stream.match(/^\${1,2}[A-Za-z_][A-Za-z0-9_]*/)) {
            return 'variableName';
        }
        if (stream.match(/^\${1,2}/)) return 'variableName';
        if (stream.match(/^(?:true|false)\b/)) return 'bool';
        if (stream.match(/^null\b/)) return 'null';
        if (stream.match(/^(?:and|or|in)\b/)) return 'operatorKeyword';
        if (stream.match(/^[A-Za-z_][A-Za-z0-9_]*/)) return 'propertyName';
        if (stream.match(/^\/(?:\\.|[^/\\\r\n])+\/[gimsy]*/)) return 'regexp';
        if (stream.match(/^(?:~>|:=|!=|<=|>=|\*\*|\.\.|\?|=|<|>|&|\+|-|\*|\/|%|\^)/)) {
            return 'operator';
        }
        if (stream.match(/^[()[\]{},;:.]/)) return 'punctuation';

        stream.next();
        return null;
    },
};

export const jsonataLanguage = StreamLanguage.define(jsonataStreamParser);

const NORMALIZED_QUERY_COMPLETIONS: readonly Completion[] = [
    { label: 'elements', type: 'variable', detail: 'Active-page query root' },
    { label: 'id', type: 'property', detail: 'Normalized element field' },
    { label: 'labels', type: 'property', detail: 'Normalized element field' },
    { label: 'title', type: 'property', detail: 'Normalized element field' },
    { label: 'section', type: 'property', detail: 'Normalized element field' },
    { label: 'normalizedPositionRange', type: 'property', detail: 'Normalized element field' },
    { label: 'start', type: 'property', detail: 'Normalized position field' },
    { label: 'end', type: 'property', detail: 'Normalized position field' },
    { label: 'timeGap', type: 'property', detail: 'Normalized element field' },
    { label: 'comparison', type: 'property', detail: 'Normalized element field' },
    { label: 'metadata', type: 'property', detail: 'Normalized element field' },
];

export const COMMON_JSONATA_FUNCTIONS = [
    '$abs',
    '$append',
    '$assert',
    '$average',
    '$boolean',
    '$ceil',
    '$contains',
    '$count',
    '$distinct',
    '$error',
    '$exists',
    '$filter',
    '$floor',
    '$join',
    '$keys',
    '$length',
    '$lookup',
    '$lowercase',
    '$map',
    '$match',
    '$max',
    '$merge',
    '$min',
    '$not',
    '$number',
    '$reduce',
    '$replace',
    '$reverse',
    '$round',
    '$shuffle',
    '$sort',
    '$split',
    '$spread',
    '$string',
    '$substring',
    '$sum',
    '$trim',
    '$type',
    '$uppercase',
    '$zip',
] as const;

export const getJsonataCompletionOptions = (labels: readonly string[]): Completion[] => {
    const exactLabels = Array.from(new Set(labels)).sort((left, right) => left.localeCompare(right));
    return [
        ...NORMALIZED_QUERY_COMPLETIONS,
        ...exactLabels.map((label): Completion => ({
            label,
            apply: JSON.stringify(label),
            type: 'constant',
            detail: 'Exact active-page label',
        })),
        ...COMMON_JSONATA_FUNCTIONS.map((label): Completion => ({
            label,
            apply: `${label}()`,
            type: 'function',
            detail: 'JSONata function',
        })),
    ];
};

export const createJsonataCompletionSource = (labels: readonly string[]): CompletionSource => {
    const options = getJsonataCompletionOptions(labels);
    return (context) => {
        const word = context.matchBefore(/[A-Za-z0-9_$-]*/);
        if (!context.explicit && (!word || word.from === word.to)) return null;
        return {
            from: word?.from ?? context.pos,
            options,
            validFor: /^[A-Za-z0-9_$-]*$/,
        };
    };
};

const clamp = (value: number, minimum: number, maximum: number): number => (
    Math.min(maximum, Math.max(minimum, value))
);

const closestTokenRange = (
    source: string,
    token: string | undefined,
    anchor: number,
): { from: number; to: number } | null => {
    if (!token || token.length === 0 || token.length > source.length) return null;
    let closestIndex = -1;
    let closestDistance = Number.POSITIVE_INFINITY;
    let index = source.indexOf(token);
    while (index >= 0) {
        const distance = Math.abs(index - anchor);
        if (distance < closestDistance) {
            closestIndex = index;
            closestDistance = distance;
        }
        index = source.indexOf(token, index + 1);
    }
    return closestIndex >= 0
        ? { from: closestIndex, to: closestIndex + token.length }
        : null;
};

export const analysisResultsQueryErrorToDiagnostic = (
    error: AnalysisResultsQueryErrorDetail,
    source: string,
): Diagnostic => {
    const documentLength = source.length;
    const hasPosition = Number.isInteger(error.position);
    const anchor = hasPosition
        ? clamp((error.position as number) - 1, 0, documentLength)
        : 0;
    const tokenRange = closestTokenRange(source, error.token, anchor);
    const fallbackRange = hasPosition
        ? { from: anchor, to: Math.min(documentLength, anchor + 1) }
        : { from: 0, to: documentLength };
    const range = tokenRange ?? fallbackRange;

    return {
        ...range,
        severity: 'error',
        source: 'JSONata',
        message: `${error.code}: ${error.message}`,
    };
};

export const getJsonataSyntaxDiagnostics = (source: string): Diagnostic[] => {
    try {
        compileAnalysisResultsQuery(source);
        return [];
    } catch (error) {
        return [analysisResultsQueryErrorToDiagnostic(
            toAnalysisResultsQueryError(error).detail,
            source,
        )];
    }
};

interface JsonataCodeMirrorExtensionOptions {
    labels: readonly string[];
    diagnosticDelayMs?: number;
    getAdditionalDiagnostics?: (source: string) => readonly Diagnostic[];
}

export const createJsonataCodeMirrorExtensions = ({
    labels,
    diagnosticDelayMs = JSONATA_DIAGNOSTIC_DELAY_MS,
    getAdditionalDiagnostics,
}: JsonataCodeMirrorExtensionOptions): Extension[] => [
    lineNumbers(),
    bracketMatching(),
    jsonataLanguage,
    jsonataLanguage.data.of({ autocomplete: createJsonataCompletionSource(labels) }),
    autocompletion(),
    linter(
        (view) => {
            const source = view.state.doc.toString();
            return [
                ...getJsonataSyntaxDiagnostics(source),
                ...(getAdditionalDiagnostics?.(source) ?? []),
            ];
        },
        { delay: diagnosticDelayMs },
    ),
    lintGutter(),
];
