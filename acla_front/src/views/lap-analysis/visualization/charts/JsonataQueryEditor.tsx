import React from 'react';
import CodeMirror, {
    EditorView,
    ExternalChange,
} from '@uiw/react-codemirror';
import { setDiagnostics } from '@codemirror/lint';
import type { AnalysisResultsQueryErrorDetail } from './analysisResultsQuery';
import {
    analysisResultsQueryErrorToDiagnostic,
    createJsonataCodeMirrorExtensions,
} from './jsonataCodeMirror';
import styles from './JsonataQueryEditor.module.css';

export interface JsonataQueryEditorProps {
    id: string;
    value: string;
    labels: readonly string[];
    isDirty: boolean;
    isEvaluating: boolean;
    diagnostic: AnalysisResultsQueryErrorDetail | null;
    diagnosticFocusRequest: number;
    onChange: (value: string) => void;
    onApply: () => void | Promise<void>;
    onReset: () => void;
}

const diagnosticText = (diagnostic: AnalysisResultsQueryErrorDetail): string => (
    `${diagnostic.code}: ${diagnostic.message}${diagnostic.position !== undefined
        ? ` (position ${diagnostic.position})`
        : ''}`
);

export const JsonataQueryEditor: React.FC<JsonataQueryEditorProps> = ({
    id,
    value,
    labels,
    isDirty,
    isEvaluating,
    diagnostic,
    diagnosticFocusRequest,
    onChange,
    onApply,
    onReset,
}) => {
    const editorRef = React.useRef<EditorView | null>(null);
    const diagnosticRef = React.useRef<HTMLDivElement | null>(null);
    const latestDiagnosticRef = React.useRef(diagnostic);
    const handledFocusRequestRef = React.useRef(0);
    const applyPendingRef = React.useRef(false);
    const [isEditorMounted, setIsEditorMounted] = React.useState(false);
    latestDiagnosticRef.current = diagnostic;
    const labelId = `${id}-label`;
    const inputId = `${id}-input`;
    const diagnosticId = `${id}-diagnostic`;
    const editorExtensions = React.useMemo(
        () => createJsonataCodeMirrorExtensions({
            labels,
            getAdditionalDiagnostics: (source) => {
                const current = latestDiagnosticRef.current;
                return current
                    ? [analysisResultsQueryErrorToDiagnostic(current, source)]
                    : [];
            },
        }),
        [labels],
    );
    const submit = React.useCallback(() => {
        if (isEvaluating || applyPendingRef.current) return;
        applyPendingRef.current = true;
        void Promise.resolve(onApply()).finally(() => {
            applyPendingRef.current = false;
        });
    }, [isEvaluating, onApply]);
    const interactionExtensions = React.useMemo(() => [
        EditorView.contentAttributes.of({
            id: inputId,
            'aria-labelledby': labelId,
            'aria-describedby': diagnosticId,
            'aria-busy': isEvaluating ? 'true' : 'false',
            autocapitalize: 'off',
            spellcheck: 'false',
        }),
    ], [inputId, isEvaluating, labelId, diagnosticId]);

    React.useEffect(() => {
        const timeoutId = setTimeout(() => setIsEditorMounted(true), 0);
        return () => clearTimeout(timeoutId);
    }, []);

    React.useEffect(() => {
        const editor = editorRef.current;
        if (!editor) return;
        const diagnostics = diagnostic
            ? [analysisResultsQueryErrorToDiagnostic(diagnostic, editor.state.doc.toString())]
            : [];
        editor.dispatch(setDiagnostics(editor.state, diagnostics));
    }, [diagnostic]);

    React.useLayoutEffect(() => {
        const editor = editorRef.current;
        if (!editor || editor.state.doc.toString() === value) return;
        editor.dispatch({
            changes: { from: 0, to: editor.state.doc.length, insert: value },
            annotations: ExternalChange.of(true),
        });
    }, [value]);

    React.useEffect(() => {
        if (
            !diagnostic
            || diagnosticFocusRequest <= handledFocusRequestRef.current
        ) return;
        handledFocusRequestRef.current = diagnosticFocusRequest;
        const editor = editorRef.current;
        if (editor) {
            const range = analysisResultsQueryErrorToDiagnostic(
                diagnostic,
                editor.state.doc.toString(),
            );
            editor.dispatch({
                selection: { anchor: range.from },
                scrollIntoView: true,
            });
        }
        diagnosticRef.current?.focus();
    }, [diagnostic, diagnosticFocusRequest]);

    React.useEffect(() => () => {
        editorRef.current = null;
    }, []);

    return (
        <section className={styles.panel} aria-labelledby={`${id}-heading`}>
            <div className={styles.header}>
                <span className={styles.heading} id={`${id}-heading`}>Edit query</span>
                {isEvaluating ? (
                    <span className={styles.status} role="status" aria-live="polite">
                        Evaluating query…
                    </span>
                ) : isDirty ? (
                    <span className={styles.status} role="status">Unsaved changes</span>
                ) : null}
            </div>
            <span className={styles.label} id={labelId}>Query expression</span>
            {isEditorMounted ? (
                <CodeMirror
                    className={styles.editor}
                    value={value}
                    minHeight="84px"
                    maxHeight="260px"
                    theme="dark"
                    basicSetup={{
                        lineNumbers: false,
                        bracketMatching: false,
                        autocompletion: false,
                        foldGutter: false,
                    }}
                    extensions={[...editorExtensions, ...interactionExtensions]}
                    onChange={onChange}
                    onKeyDownCapture={(event) => {
                        if (event.ctrlKey && event.key === 'Enter') {
                            event.preventDefault();
                            event.stopPropagation();
                            submit();
                        }
                    }}
                    onCreateEditor={(editor) => {
                        editorRef.current = editor;
                        if (diagnostic) {
                            editor.dispatch(setDiagnostics(editor.state, [
                                analysisResultsQueryErrorToDiagnostic(
                                    diagnostic,
                                    editor.state.doc.toString(),
                                ),
                            ]));
                        }
                    }}
                />
            ) : (
                <div className={styles.editorPlaceholder} aria-hidden="true" />
            )}
            <div className={styles.actions}>
                <button
                    type="button"
                    className={styles.button}
                    disabled={isEvaluating}
                    onClick={onReset}
                >
                    Reset
                </button>
                <button
                    type="button"
                    className={styles.button}
                    disabled={isEvaluating}
                    onClick={submit}
                >
                    {isEvaluating ? 'Applying…' : 'Apply'}
                </button>
            </div>
            {diagnostic && (
                <div
                    className={styles.error}
                    data-testid="active-page-query-error"
                    id={diagnosticId}
                    ref={diagnosticRef}
                    role="alert"
                    tabIndex={-1}
                >
                    {diagnosticText(diagnostic)}
                </div>
            )}
        </section>
    );
};

export default JsonataQueryEditor;
