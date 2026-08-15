# JSONata Analysis-Result Queries

## Outcome

Replace the fixed `result_count`/`mistake_count` analysis-result query enum and the active-page Practice/Racing filter with one JSONata query system. The same evaluator powers:

- `query_analysis_result` AI tool calls;
- built-in active-page presets;
- page-label quick-filter presets derived from the active page's normalized `elements[].labels`; and
- custom expressions entered in the Analysis Results UI.

The evaluator receives only the normalized active page:

```json
{
  "elements": [
    {
      "id": "...",
      "labels": ["..."],
      "title": "...",
      "section": "...",
      "normalizedPositionRange": { "start": 0, "end": 1 },
      "timeGap": {},
      "comparison": {},
      "metadata": {}
    }
  ]
}
```

This is a breaking replacement. Do not translate old query names, accept a legacy enum alongside expressions, or keep the old active-page filter as a fallback.

## Success Criteria

- Any valid JSONata expression can query the full normalized active page and return its actual JSON scalar, object, array, or `null` result.
- The active-page UI derives its count, cards, sorting, and frequency graph from the last successfully applied element query.
- Training, Racing, All, and page-label quick-filter choices are presets that contain ordinary JSONata; they do not use a parallel filtering implementation.
- Syntax, runtime, resource-limit, and UI-result-shape failures are actionable and never replace a valid active-page result set with partial data.
- Overall Trends remains cross-page and taxonomy-specific, with its own Training/Racing state.
- No production code special-cases `result_count` or `mistake_count` after migration.

## 1. Shared Evaluator and Contracts

Add exact frontend dependencies consistent with the repository's pinned dependency style:

- [`jsonata@2.2.2`](https://www.npmjs.com/package/jsonata), which exposes the `timeout`, `stack`, and `sequence` evaluator options;
- `@uiw/react-codemirror@4.25.11`;
- `@codemirror/language@6.12.4`;
- `@codemirror/autocomplete@6.20.3`; and
- `@codemirror/lint@6.9.7`.

Create a pure `analysisResultsQuery` module rather than putting evaluator logic in the React chart or AI command registry. It owns:

The `analysisResultsQuery` contract is explicit: its input is the query, `{ query: string }`, and its output is `{ status: "ready", data: JsonValue }`.

- recursive `JsonValue` types;
- `{ query: string }` and `{ status: "ready", data: JsonValue }` query types;
- expression compilation and evaluation;
- structured JSONata error normalization;
- JSON-safe output validation/cloning;
- preset construction and string escaping; and
- conversion of a UI query result into canonical active-page elements.

Both the chart handle and command registry import the shared types. Remove the current chart-to-`ai-command-registry` type dependency.

### Evaluator behavior

- Reject a missing, non-string, or whitespace-only expression. Use the original untrimmed string for compilation so diagnostic positions still align with the editor.
- Compile with JSONata 2.2.2 guardrails `{ timeout: 100, stack: 64, sequence: 10000 }` and evaluate asynchronously against `{ elements }`.
- Do not bind application functions, browser globals, alternate roots, or hidden page data.
- Recursively validate and detach the normalized `{ elements }` input with the same JSON-safety rules before evaluation; `metadata` is typed as `unknown` today, so circular or executable values must not enter JSONata.
- Normalize a top-level `undefined`/empty-sequence result to `null`.
- Recursively validate the result as JSON-safe. Reject functions, symbols, bigint values, non-finite numbers, nested `undefined`, cycles, and other non-serializable values instead of silently dropping or coercing them. Return a detached JSON-safe value so JSONata sequence markers and object identities do not leak into tool output.
- Preserve JSONata `code`, one-based `position`, `token`, and message in a normalized error detail object. Convert `position` to CodeMirror offsets only at the UI boundary.
- Use a monotonically increasing evaluation generation in the chart so an older async Apply or page-change evaluation cannot overwrite a newer result. JSONata evaluation is not cancellable; stale completions are ignored.

JSONata's official guardrails are cooperative evaluator checks. Confirm the 100 ms failure with a real non-terminating/tail-recursive expression in tests. If that test can still block past the limit in the browser build, move evaluation behind a terminable Web Worker before shipping rather than advertising an unenforced deadline.

### UI element-result contract

AI tool calls may return any `JsonValue`. Active-page filtering is intentionally narrower:

- Accept `null` as zero matches.
- Accept one string ID, one object with a string `id`, or an array/JSONata sequence of those values. This handles JSONata singleton collapsing without requiring special query syntax.
- Resolve every ID against the current canonical `elements` map. A projected object is only an identifier carrier; render the canonical element, not arbitrary projected fields.
- Reject the whole UI result if any item has no usable ID, references an ID outside the active page, is nested unexpectedly, or has another scalar type. Do not partially apply valid members of a mixed result.
- Remove duplicate IDs while preserving query-result order.

## 2. AI Tool Breaking Change

Change `query_analysis_result` to accept exactly:

```ts
{ query: string }
```

and return:

```ts
{ status: 'ready', data: JsonValue }
```

Frontend validation still rejects missing/empty queries, non-string queries, and extra arguments even if a caller bypasses the backend schema. The chart handle executes the shared async evaluator through the normal `AiToolOperation` path.

Update the backend session-tool registry to:

- remove the enum;
- describe the active normalized `{ elements }` input;
- require one string `query` and disallow extra properties where the registry schema supports it;
- explain that results are actual JSONata values, not counts by default; and
- include representative expressions:
  - `$count(elements)`;
  - `$count(elements[labels[$ = "Mistake (Practice)"]])`; and
  - `elements[labels[$ = "Lockup"]].{ "id": id, "section": section }`.

`result_count` and `mistake_count` remain syntactically valid JSONata field lookups, but the root has no such fields, so they naturally return `null`. Do not recognize or translate them. Add a negative regression test that proves there is no alias behavior.

Goal determinations already require `{ status: "ready", data: finiteNumber }`. Keep that clean envelope and update tool guidance/tests so a determination using `query_analysis_result` supplies an expression returning a number, normally `$count(...)`.

For AI-visible query failures, add a structured error detail field to the existing tool-error serialization path; otherwise JSONata `code` and `position` are lost when `use-voice-conversation` emits the failed tool frame. Keep the human-readable message as well.

## 3. Presets and Query State

Use stable preset keys, separate from display labels and expressions:

- All: `elements`
- Training: a generated predicate matching every recognized Practice/Training parent-label value (canonical ID, canonical fallback name, and current resolved name)
- Racing: the equivalent generated predicate for Racing
- Page-label quick filters: one per distinct exact string found in the normalized active page's `elements[].labels`

### Page-label quick filters

These are convenience filters for labels that actually occur on the page the user is viewing. They are generated locally from the already-normalized active-page elements; they are not generated by AI, loaded from the complete taxonomy, inferred from label relationships, persisted as user presets, or collected from the other paginated analysis pages.

For example, given active-page elements with labels `['MSP', 'Lockup']` and `['MSR', 'Wide exit']`, show a separate **Page labels** group containing `Lockup`, `MSP`, `MSR`, and `Wide exit`. Selecting `Lockup` immediately applies:

```jsonata
elements[labels[$ = "Lockup"]]
```

The precise construction rules are:

- Treat the strings in normalized `elements[].labels` as opaque values. An observed taxonomy ID such as `MSP1` and an observed display name such as `Late turn-in` are separate choices, even if the taxonomy currently maps them to the same concept.
- Match one label value by exact, case-sensitive string equality. Do not expand it to aliases, descendants, fuzzy matches, or case-folded matches. Training and Racing remain the only semantic multi-value presets.
- Deduplicate repeated occurrences of the same exact value within an element and across elements. Values that differ by case remain distinct.
- Do not create a quick filter for an empty `labels` array. Users can express that case as a custom query, for example `elements[$count(labels) = 0]`.
- Display the exact observed value as the option text and prefix its internal key (for example, `page-label:<value>`) so a page label named `All`, `Training`, or `Racing` cannot collide with a built-in preset.
- Put these dynamic choices in a visibly separate **Page labels** option group after All, Training, and Racing. When the active page has no labels, omit the group or show it disabled; do not invent taxonomy-wide choices.
- Escape the value with `JSON.stringify` when constructing the JSONata string literal. Never interpolate unescaped label text.
- Sort the choices by case-insensitive label text, then by exact value for deterministic case-sensitive ties.

This distinction is intentional: selecting the Training preset can match `MSP`, `Mistake (Practice)`, and the current resolved Training parent name together, while selecting the literal `MSP` page-label quick filter matches only elements whose `labels` array actually contains `MSP`.

For Training and Racing, generate JSONata string literals with `JSON.stringify` and deduplicate predicate values before generating the expression.

State transitions must be explicit:

- Training is the initial active-page preset.
- Selecting a preset immediately applies that preset and copies its exact expression into the editor.
- Typing changes only the draft and marks it dirty; it does not change cards or graphs.
- Apply or Ctrl+Enter compiles/evaluates the draft. Only a successful element-shaped UI result commits the expression and matched IDs; a custom success changes the selection to Custom.
- Reset restores the last selected preset expression. When Custom is selected, Reset restores the last successfully applied custom expression.
- A failed Apply leaves the previously applied expression and current matched elements intact and focuses the diagnostic.
- When active-page data changes, re-evaluate the committed expression against the new canonical page. Never render element objects retained from a prior page. If this automatic re-evaluation fails, show the error and fail closed to zero matches for the new page.
- Rebuild page-label quick filters after active-page mutations and page switches. If the selected page-label value is absent from the new active page, retain its exact expression and matched-ID result semantics as Custom rather than silently switching presets. The normal page-change re-evaluation still applies, so this can validly produce zero matches.
- If taxonomy label resolution changes while Training or Racing is selected, regenerate and reapply that preset so its semantic meaning stays current.

## 4. Query Editor

Add an “Edit query” panel to the active-page screen only. Overall Trends keeps its current compact Training/Racing control and does not show the query editor.

Implement a local CodeMirror extension with `StreamLanguage.define` and the documented [custom-language APIs](https://codemirror.net/examples/lang-package/):

- JSONata token highlighting, including C-style comments;
- bracket matching and line numbers;
- completion for `elements`, normalized element fields, the exact label strings used by the active page's page-label quick filters, and common JSONata functions;
- debounced compile-only syntax diagnostics while editing; and
- runtime/resource/UI-shape diagnostics on Apply.

Keep editor concerns in a small `JsonataQueryEditor` component and a separately tested `jsonataCodeMirror` extension. Cancel pending diagnostic debounce work on draft change/unmount. Clamp JSONata error positions to the document length and highlight the token when available.

The UI should expose loading/evaluating state, disable duplicate Apply submissions, and retain the draft text after an error so it can be corrected.

## 5. Active-Page Derivations

After a successful UI query, use canonical `matchedElements` as the only input to:

- the “N of M total” count;
- cards;
- empty state;
- sorting; and
- the active-page frequency graph.

Preserve taxonomy-aware behavior for the Training and Racing presets: their graph and “most frequent” sort count recognized child labels and omit parent-category labels. All, page-label quick filters, and Custom count all exact labels in the matched set.

Rename active-page concepts that are no longer mistake-specific:

- `filteredElements` -> `matchedElements`;
- `most-frequent-sub-label` -> `most-frequent-label`;
- active-page `buildMistakeFrequencyData` -> a generic label-frequency helper;
- active-page `mistakeFrequency*` variables/test IDs/text -> `labelFrequency*`; and
- active-page “No Training/Racing Mistake results” copy -> query-aware empty-state copy.

Keep “Most time lost” unchanged. Preserve JSONata result order for Original order; other sorts operate on a copy and use the query order as their stable tie-breaker.

## 6. Separate Overall-Trend State

The current chart shares `mainLabelFilter` between active-page filtering and Overall Trends. Split it during this change:

- Rename the remaining taxonomy type/options/state to trend-specific concepts such as `MistakeTrendParent`, `TREND_PARENT_OPTIONS`, and `trendParent`.
- Use that state only in `OverallMistakeTrend`, `buildMistakeTrendData`, recognized trend sub-labels, and trend graph copy.
- Active-page presets/query state must not read or mutate the trend selection, and trend selection must not overwrite the active query.

This keeps the cross-page Overall Trends screen on its existing Training/Racing behavior while fully deleting the old active-page filter implementation.

## 7. Legacy Cleanup Checklist

Delete rather than deprecate or wrap:

- `AnalysisResultQueryData` and its `result_count`/`mistake_count` keyed generic types;
- enum-specific `validateAnalysisResultQueryArguments` branches and messages;
- the chart's `analysisResultQueryData` count memo and indexed lookup;
- `countAnalysisResultMistakes` once its query-only caller is removed;
- backend enum/schema descriptions and enum-based registry expectations;
- old active-page `mainLabelFilter` filtering, select, and Practice/Racing-only empty-state logic;
- old active-page mistake-frequency naming and the local sort-mode value it made obsolete; and
- tests/mocks that expect the aliases to return numbers or expect only Training/Racing active-page options.

Keep only where still semantically owned:

- `getAnalysisResultMistakeParentLabels`, because presets and Overall Trends still need taxonomy aliases;
- mistake terminology inside the cross-page Overall Trends implementation;
- `mistake_count` fields in `user-summary-ai-tools.ts`, which are a separate user-summary data contract; and
- `analysisResultsModel` input normalization for existing upstream array/`items`, snake_case, and camelCase analysis payloads. Removing ingestion compatibility is a separate migration and is not required by the query API clean break.

After implementation, search production frontend/backend code for `result_count` and `mistake_count`. The analysis-result aliases should have no production matches; unrelated user-summary fields and an explicit negative no-alias regression test are acceptable.

## 8. Implementation Order

1. Add dependencies and the pure evaluator/types/error normalizer.
2. Add evaluator, JSON-safety, UI-result resolution, preset-generation, and no-alias unit tests.
3. Replace frontend AI query types/validation/dispatch with the expression contract.
4. Update backend registry schema, descriptions, goal guidance, and tests.
5. Split trend state from active-page state.
6. Add query presets/state and drive all active-page derivations from `matchedElements`.
7. Add the CodeMirror editor and diagnostics.
8. Remove orphaned legacy helpers/names/tests and run the repository-wide alias search.

## 9. Test Plan

### Pure evaluator

- scalar, object, array, projection, aggregation, label filtering, singleton collapse, empty sequence -> `null`, and comments;
- rejection of non-JSON-safe/cyclic normalized input as well as unsafe output;
- syntax/runtime errors with code/position/token;
- timeout, stack, sequence, and JSON-safe result failures;
- function-valued, cyclic/non-finite/nested-undefined outputs;
- IDs, projected objects, unknown IDs, mixed invalid results, duplicates, ordering, and `null` UI results;
- special-character/Unicode label escaping, exact-value deduplication, case-distinct values, and collisions with built-in display names; and
- `result_count`/`mistake_count` produce no legacy count behavior.

### AI/backend

- missing, blank, non-string, and extra arguments;
- actual JSON-safe scalar/object/array/null output;
- structured query error details survive the frontend failed-tool frame;
- backend schema has no enum and advertises expression examples in every existing eligible session context; and
- goal determination accepts finite numeric `data` and rejects nonnumeric query results.

### UI

- All, Training, Racing, every distinct exact active-page label, and Custom, including the separate **Page labels** grouping;
- preset selection, dirty draft, Apply, Ctrl+Enter, Reset, and last-valid preservation;
- compile debounce, runtime/resource/UI-shape diagnostics, focus, and stale async result suppression;
- active-page switching, element append/update/remove, page-label quick-filter rebuilding, disappearing selected page labels becoming Custom, and taxonomy refresh;
- count, cards, empty state, Original/label-frequency/time-lost sorting, and frequency graphs all use `matchedElements`; and
- Overall Trends still opens/navigates as before and its Training/Racing state is independent.

## 10. Verification

Run from `acla_front`:

```powershell
npx tsc --noEmit
npm test -- --watchAll=false --runInBand AnalysisResultsChart analysisResultsQuery ai-command-registry Goal use-voice-conversation
npm run build
npm ls jsonata @codemirror/state @codemirror/view @codemirror/language @codemirror/autocomplete @codemirror/lint
```

Run the targeted backend registry/controller tests from `acla_backend`, then run the normal backend test suite if the targeted set passes.

Finally verify the lockfile has one compatible CodeMirror 6 dependency tree and search both projects for obsolete alias/filter production code.

## Assumptions

- Query scope is always the currently active normalized Analysis Results page, even while the pagination UI is displaying Overall Trends.
- Custom expressions apply explicitly, not live while typing.
- Page-label quick filters are ephemeral, active-page-only shortcuts over exact normalized label strings; they are not saved user presets or taxonomy aliases.
- The active-page UI accepts only results resolvable to active element IDs; the AI tool remains unrestricted to any JSON-safe result.
- This migration intentionally breaks the old analysis-result query aliases and active-page filter implementation. It does not rename unrelated domain fields or remove upstream analysis-payload normalization.
- Existing overlapping worktree changes are preserved and extended without resetting or cleaning unrelated edits.
