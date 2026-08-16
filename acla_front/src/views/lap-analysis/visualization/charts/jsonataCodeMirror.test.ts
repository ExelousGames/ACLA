import { EditorState } from '@codemirror/state';
import { EditorView } from '@codemirror/view';
import { diagnosticCount } from '@codemirror/lint';
import { StringStream } from '@codemirror/language';
import * as analysisResultsQuery from './analysisResultsQuery';
import {
    analysisResultsQueryErrorToDiagnostic,
    COMMON_JSONATA_FUNCTIONS,
    createJsonataCodeMirrorExtensions,
    getJsonataCompletionOptions,
    getJsonataSyntaxDiagnostics,
    jsonataStreamParser,
    type JsonataTokenizerState,
} from './jsonataCodeMirror';

const tokenizeLine = (
    line: string,
    state: JsonataTokenizerState = { blockComment: false },
): Array<{ text: string; style: string | null }> => {
    const stream = new StringStream(line, 4, 2);
    const tokens: Array<{ text: string; style: string | null }> = [];
    while (!stream.eol()) {
        stream.start = stream.pos;
        const style = jsonataStreamParser.token(stream, state);
        if (stream.pos === stream.start) throw new Error('Tokenizer did not advance.');
        if (stream.current().trim()) tokens.push({ text: stream.current(), style });
    }
    return tokens;
};

describe('JSONata CodeMirror language support', () => {
    afterEach(() => {
        jest.useRealTimers();
        jest.restoreAllMocks();
        document.body.replaceChildren();
    });

    it('tokenizes identifiers, variables, operators, strings, and C-style comments', () => {
        const state = jsonataStreamParser.startState?.(2) ?? { blockComment: false };
        const tokens = tokenizeLine(
            'elements[id = "one" and $count(labels) > 0] /* note */',
            state,
        );

        expect(tokens).toEqual(expect.arrayContaining([
            { text: 'elements', style: 'propertyName' },
            { text: 'id', style: 'propertyName' },
            { text: '=', style: 'operator' },
            { text: '"one"', style: 'string' },
            { text: 'and', style: 'operatorKeyword' },
            { text: '$count', style: 'variableName' },
            { text: '>', style: 'operator' },
            { text: '/* note */', style: 'comment' },
        ]));

        const openCommentState = { blockComment: false };
        expect(tokenizeLine('/* first line', openCommentState)).toEqual([
            { text: '/* first line', style: 'comment' },
        ]);
        expect(openCommentState.blockComment).toBe(true);
        expect(tokenizeLine('second */ elements', openCommentState)).toEqual([
            { text: 'second */', style: 'comment' },
            { text: 'elements', style: 'propertyName' },
        ]);
        expect(openCommentState.blockComment).toBe(false);
    });

    it('offers the query root, normalized fields, exact labels, and common functions', () => {
        const options = getJsonataCompletionOptions([
            'Wheel lock',
            'Quote "label"',
            'Wheel lock',
        ]);
        const labels = options.map((option) => option.label);

        expect(labels).toEqual(expect.arrayContaining([
            'elements',
            'id',
            'labels',
            'title',
            'section',
            'normalizedPositionRange',
            'start',
            'end',
            'timeGap',
            'comparison',
            'metadata',
            'Wheel lock',
            'Quote "label"',
            '$count',
            '$filter',
            '$sort',
        ]));
        expect(labels.filter((label) => label === 'Wheel lock')).toHaveLength(1);
        expect(options.find(({ label }) => label === 'Quote "label"')?.apply)
            .toBe('"Quote \\"label\\""');
        expect(COMMON_JSONATA_FUNCTIONS).toContain('$map');
    });

    it('normalizes compile-only errors and maps Apply failures to clamped token ranges', () => {
        const syntaxDiagnostics = getJsonataSyntaxDiagnostics('elements[');
        expect(syntaxDiagnostics).toHaveLength(1);
        expect(syntaxDiagnostics[0].message).toContain('S0203');
        expect(syntaxDiagnostics[0].from).toBeGreaterThanOrEqual(0);
        expect(syntaxDiagnostics[0].to).toBeLessThanOrEqual('elements['.length);

        const source = 'elements.$unknown()';
        expect(analysisResultsQueryErrorToDiagnostic({
            code: 'T1006',
            message: 'Attempted to invoke a non-function.',
            position: 500,
            token: '$unknown',
        }, source)).toMatchObject({
            from: source.indexOf('$unknown'),
            to: source.indexOf('$unknown') + '$unknown'.length,
            severity: 'error',
        });
        expect(analysisResultsQueryErrorToDiagnostic({
            code: 'D1012',
            message: 'Evaluation timeout after 100 milliseconds.',
        }, source)).toMatchObject({ from: 0, to: source.length });
        expect(analysisResultsQueryErrorToDiagnostic({
            code: 'INVALID_ACTIVE_PAGE_QUERY_RESULT',
            message: 'The result cannot be rendered.',
        }, source)).toMatchObject({ from: 0, to: source.length });
    });

    it('debounces syntax checks to the latest draft and cancels pending work on destroy', async () => {
        const parent = document.createElement('div');
        document.body.appendChild(parent);
        const compile = jest.spyOn(analysisResultsQuery, 'compileAnalysisResultsQuery');
        const view = new EditorView({
            parent,
            state: EditorState.create({
                doc: 'elements[',
                extensions: createJsonataCodeMirrorExtensions({
                    labels: [],
                    diagnosticDelayMs: 20,
                }),
            }),
        });

        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: 'elements' } });
        await new Promise((resolve) => setTimeout(resolve, 35));
        expect(diagnosticCount(view.state)).toBe(0);
        expect(compile).toHaveBeenCalledTimes(1);
        expect(compile).toHaveBeenLastCalledWith('elements');

        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: 'elements[' } });
        await new Promise((resolve) => setTimeout(resolve, 35));
        expect(diagnosticCount(view.state)).toBe(1);
        expect(compile).toHaveBeenLastCalledWith('elements[');

        const callsBeforeDestroy = compile.mock.calls.length;
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: 'elements[' } });
        view.destroy();
        await new Promise((resolve) => setTimeout(resolve, 35));
        expect(compile).toHaveBeenCalledTimes(callsBeforeDestroy);
    });
});
