<<<<<<< ours
# JSONata Analysis-Result Queries

## Outcome

Replace the fixed `result_count`/`mistake_count` analysis-result query enum and the active-page Practice/Racing filter with one JSONata query system. The same evaluator powers:

- `query_analysis_result` AI tool calls;
- built-in active-page presets;
- page-label quick-filter presets derived from the active page's normalized `elements[].labels`; and
- custom expressions entered in the Analysis Results UI.

The evaluator receives only the normalized active page:
=======
# JSONata Analysis-Result and Overall-Trend Queries

## Outcome

Replace the fixed `result_count`/`mistake_count` analysis-result query enum, the active-page Practice/Racing filter, and the imperative Overall Trends aggregation with one JSONata query system. The same evaluator powers:

- `query_analysis_result` AI tool calls;
- a small, fixed set of active-page query templates, where each template defines both which results are shown and their order; and
- custom expressions entered in the Analysis Results UI; and
- generated cross-page queries for the Overall Trends Training/Racing views.

The evaluator accepts one of two explicit, JSON-safe roots. AI calls and active-page queries receive only the normalized active page:
>>>>>>> theirs

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

<<<<<<< ours
=======
Overall Trends queries receive all retained normalized pages:

```json
{
  "pages": [
    {
      "id": "...",
      "createdAt": 0,
      "sourceIndex": 0,
      "baseline": {
        "lap": 1,
        "lapTimeMs": 0,
        "track": "...",
        "car": "..."
      },
      "elements": []
    }
  ]
}
```

Each trend page uses the same normalized element shape shown above. `sourceIndex` is assigned before evaluation and is used only as the stable tie-breaker when `createdAt` values match.

>>>>>>> theirs
This is a breaking replacement. Do not translate old query names, accept a legacy enum alongside expressions, or keep the old active-page filter as a fallback.

## Success Criteria

- Any valid JSONata expression can query the full normalized active page and return its actual JSON scalar, object, array, or `null` result.
<<<<<<< ours
- The active-page UI derives its count, cards, sorting, and frequency graph from the last successfully applied element query.
- Training, Racing, All, and page-label quick-filter choices are presets that contain ordinary JSONata; they do not use a parallel filtering implementation.
- Syntax, runtime, resource-limit, and UI-result-shape failures are actionable and never replace a valid active-page result set with partial data.
- Overall Trends remains cross-page and taxonomy-specific, with its own Training/Racing state.
=======
- The active-page UI derives its count, card order, and frequency graph from the last successfully applied element query.
- The active-page UI has one query-template selector. It does not expose separate filter and sort controls.
- Each predefined template is an ordinary JSONata expression that performs its complete filter-and-order behavior; templates are not composable filter or sort fragments.
- Overall Trends evaluates generated JSONata against the normalized cross-page root; its lap rows, parent-mistake totals, and per-category counts do not come from a parallel imperative filter/aggregation path.
- The Overall Trends Training/Racing selector chooses a complete generated trend expression. It remains independent of the active-page template/custom-query state.
- Syntax, runtime, resource-limit, active-page-result-shape, and trend-result-shape failures are actionable and never replace a valid result with partial data from the same input generation.
- Overall Trends remains cross-page and taxonomy-specific, with its own Training/Racing state and a strict trend-result contract.
>>>>>>> theirs
- No production code special-cases `result_count` or `mistake_count` after migration.

## 1. Shared Evaluator and Contracts

Add exact frontend dependencies consistent with the repository's pinned dependency style:

- [`jsonata@2.2.2`](https://www.npmjs.com/package/jsonata), which exposes the `timeout`, `stack`, and `sequence` evaluator options;
- `@uiw/react-codemirror@4.25.11`;
- `@codemirror/language@6.12.4`;
- `@codemirror/autocomplete@6.20.3`; and
- `@codemirror/lint@6.9.7`.

<<<<<<< ours
Create a pure `analysisResultsQuery` module rather than putting evaluator logic in the React chart or AI command registry. It owns:

The `analysisResultsQuery` contract is explicit: its input is the query, `{ query: string }`, and its output is `{ status: "ready", data: JsonValue }`.

- recursive `JsonValue` types;
- `{ query: string }` and `{ status: "ready", data: JsonValue }` query types;
- expression compilation and evaluation;
- structured JSONata error normalization;
- JSON-safe output validation/cloning;
- preset construction and string escaping; and
- conversion of a UI query result into canonical active-page elements.
=======
Create a pure `analysisResultsQuery` module rather than putting evaluator logic in the React chart or AI command registry. The public AI-tool contract remains explicit: its input is `{ query: string }`, and its output is `{ status: "ready", data: JsonValue }`. Internally, the evaluator accepts the query plus either a validated `ActivePageQueryInput` or a validated `OverallTrendQueryInput`.

The module owns:

- recursive `JsonValue` types;
- `{ query: string }` and `{ status: "ready", data: JsonValue }` AI query types;
- active-page and cross-page trend input types/normalizers;
- expression compilation and evaluation;
- structured JSONata error normalization;
- JSON-safe output validation/cloning;
- predefined active-page and Overall Trends expression construction, taxonomy-aware mistake matching, and string escaping;
- conversion of a UI query result into canonical active-page elements; and
- strict validation/conversion of an Overall Trends query result into canonical trend rows and categories.
>>>>>>> theirs

Both the chart handle and command registry import the shared types. Remove the current chart-to-`ai-command-registry` type dependency.

### Evaluator behavior

- Reject a missing, non-string, or whitespace-only expression. Use the original untrimmed string for compilation so diagnostic positions still align with the editor.
<<<<<<< ours
- Compile with JSONata 2.2.2 guardrails `{ timeout: 100, stack: 64, sequence: 10000 }` and evaluate asynchronously against `{ elements }`.
- Do not bind application functions, browser globals, alternate roots, or hidden page data.
- Recursively validate and detach the normalized `{ elements }` input with the same JSON-safety rules before evaluation; `metadata` is typed as `unknown` today, so circular or executable values must not enter JSONata.
- Normalize a top-level `undefined`/empty-sequence result to `null`.
- Recursively validate the result as JSON-safe. Reject functions, symbols, bigint values, non-finite numbers, nested `undefined`, cycles, and other non-serializable values instead of silently dropping or coercing them. Return a detached JSON-safe value so JSONata sequence markers and object identities do not leak into tool output.
- Preserve JSONata `code`, one-based `position`, `token`, and message in a normalized error detail object. Convert `position` to CodeMirror offsets only at the UI boundary.
- Use a monotonically increasing evaluation generation in the chart so an older async Apply or page-change evaluation cannot overwrite a newer result. JSONata evaluation is not cancellable; stale completions are ignored.
=======
- Compile with JSONata 2.2.2 guardrails `{ timeout: 100, stack: 64, sequence: 10000 }` and evaluate asynchronously against the one normalized root supplied by the caller.
- Do not bind application functions, browser globals, alternate roots, or hidden page data.
- Recursively validate and detach either normalized root with the same JSON-safety rules before evaluation; `metadata` is typed as `unknown` today, so circular or executable values must not enter JSONata. Reject a root whose discriminator/shape does not match the calling surface rather than making both `elements` and `pages` available at once.
- Normalize a top-level `undefined`/empty-sequence result to `null`.
- Recursively validate the result as JSON-safe. Reject functions, symbols, bigint values, non-finite numbers, nested `undefined`, cycles, and other non-serializable values instead of silently dropping or coercing them. Return a detached JSON-safe value so JSONata sequence markers and object identities do not leak into tool output.
- Preserve JSONata `code`, one-based `position`, `token`, and message in a normalized error detail object. Convert `position` to CodeMirror offsets only at the UI boundary.
- Use separate monotonically increasing active-page and Overall Trends evaluation generations in the chart so an older async Apply, page change, taxonomy refresh, or trend-selection evaluation cannot overwrite a newer result. JSONata evaluation is not cancellable; stale completions are ignored.
>>>>>>> theirs

JSONata's official guardrails are cooperative evaluator checks. Confirm the 100 ms failure with a real non-terminating/tail-recursive expression in tests. If that test can still block past the limit in the browser build, move evaluation behind a terminable Web Worker before shipping rather than advertising an unenforced deadline.

### UI element-result contract

<<<<<<< ours
AI tool calls may return any `JsonValue`. Active-page filtering is intentionally narrower:
=======
AI tool calls may return any `JsonValue`. Active-page element rendering is intentionally narrower:
>>>>>>> theirs

- Accept `null` as zero matches.
- Accept one string ID, one object with a string `id`, or an array/JSONata sequence of those values. This handles JSONata singleton collapsing without requiring special query syntax.
- Resolve every ID against the current canonical `elements` map. A projected object is only an identifier carrier; render the canonical element, not arbitrary projected fields.
- Reject the whole UI result if any item has no usable ID, references an ID outside the active page, is nested unexpectedly, or has another scalar type. Do not partially apply valid members of a mixed result.
- Remove duplicate IDs while preserving query-result order.

<<<<<<< ours
=======
### Overall Trends result contract

Overall Trends uses a narrower result contract than the AI tool. A generated trend expression returns:

```ts
{
    laps: Array<{
        pageId: string;
        lap: number;
        lapTimeMs: number | null;
        totalCount: number;
        categoryCounts: Array<{ id: string; label: string; count: number }>;
    }>;
    categories: Array<{ id: string; label: string; occurrences: number }>;
}
```

- Require exactly one lap row for every current trend-input page, in `createdAt`/`sourceIndex` order. Resolve `pageId` against the current page map and rebuild display labels from canonical page data.
- Require finite, non-negative integer counts; a finite positive `lap`, a finite non-negative `lapTimeMs` or `null`; and unique category IDs within every array.
- Require category IDs and display labels to match the taxonomy values embedded in the selected generated query. Every lap's `categoryCounts` contains each recognized canonical child category exactly once, including zero counts; top-level `categories` contains exactly the positive-occurrence categories in deterministic label order. Do not accept projected labels as new taxonomy/UI content.
- Require each top-level category occurrence total to equal the sum of that category's per-lap counts, and each per-lap category count to be no greater than that lap's parent-mistake `totalCount`.
- Reject the whole result on an unknown/missing/duplicate page, unknown category, invalid scalar, inconsistent aggregate, or unexpected nesting. Never partially graph a malformed trend result.
- Treat lap-time best-fit values, direction descriptions, graph specs, and selected-category presentation as deterministic TypeScript derivations of the validated JSONata result; JSONata owns cross-page ordering, taxonomy selection, and aggregation.

>>>>>>> theirs
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

<<<<<<< ours
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
=======
## 3. Combined Query Templates and State

The active-page UI has one **View** selector containing a small, fixed catalog of complete query templates. Do not expose one selector for filtering and another for sorting, and do not model filter and sort templates as pieces that users combine. Each option answers a recognizable question and owns one JSONata expression that both selects the elements and returns them in display order.

Use stable template keys, separate from display labels and generated expressions:

| Key | Display label | Complete behavior |
| --- | --- | --- |
| `all-results` | All results | Show every element in source order. |
| `mistakes` | Mistakes | Show elements matching either a recognized Practice/Training or Racing parent label, in source order. |
| `common-label-mistakes` | Most common label in mistakes | Show recognized Training or Racing mistakes only. Order elements by the page-wide frequency of their most common recognized non-parent mistake label, most frequent first. |
| `time-lost-mistakes` | Most time lost in mistakes | Show recognized Training or Racing mistakes only. Put finite `timeGap.deltaMs` values first, largest to smallest, and keep source order for ties and missing/invalid values. |

`all-results` is exactly `elements`. Generate the other expressions from the recognized taxonomy values. A recognized parent value includes the canonical ID, canonical fallback name, and current resolved name for the applicable Training or Racing parent. For common-label ordering, recognize the corresponding child IDs and resolved names, count each child label at most once per element, exclude parent-category labels, use case-insensitive label text followed by exact text as the deterministic tie-breaker, and then use source order. Escape every generated JSONata string literal with `JSON.stringify` and deduplicate values before building an expression.

The template expression itself must perform the filtering and ordering. After evaluation, the UI only resolves returned IDs to canonical active-page elements while preserving query-result order; it must not apply another template-specific filter or sort in React. Keep template construction in the shared query module so the dropdown, editor, and tests use the exact same expressions.

Treat this as a replacement of the current active-page filter/sort model, not an extension of it. Do not carry `AnalysisResultsMainLabelFilter`, `MainLabelFilterOption.sortDisplayName`, or the old `MSP`/`MSR` selection into the active-page query state. The remaining Training/Racing option model belongs exclusively to Overall Trends and contains no active-page sort metadata.

Do not generate templates for individual page labels. Users who need a label-specific view can edit a custom query, aided by completion for labels on the active page. Do not add separate options such as “Mistakes only,” “Most common label,” or “Most time lost” that can be combined into an implicit query.

State transitions must be explicit:

- `mistakes` is the initial active-page template.
- Selecting a template immediately evaluates its complete expression and copies that exact expression into the editor.
- Typing changes only the draft and marks it dirty; it does not change cards or graphs.
- Apply or Ctrl+Enter compiles/evaluates the draft. Only a successful element-shaped UI result commits the expression and matched IDs; a custom success changes the selector to Custom.
- Reset restores the selected template's complete expression. When Custom is selected, Reset restores the last successfully applied custom expression.
- A failed Apply leaves the previously applied expression and current matched elements intact and focuses the diagnostic.
- When active-page data changes, re-evaluate the committed expression against the new canonical page. Never render element objects retained from a prior page. If this automatic re-evaluation fails, show the error and fail closed to zero matches for the new page.
- If taxonomy label resolution changes while a taxonomy-dependent template is selected, regenerate and reapply that complete template. A committed custom expression is never rewritten.

## 4. Query Editor

Add an “Edit query” panel to the active-page screen only. Overall Trends uses the same evaluator internally but keeps its compact Training/Racing control; exposing arbitrary custom trend-result authoring is outside this change because that screen requires the stricter multi-series result contract above.
>>>>>>> theirs

Implement a local CodeMirror extension with `StreamLanguage.define` and the documented [custom-language APIs](https://codemirror.net/examples/lang-package/):

- JSONata token highlighting, including C-style comments;
- bracket matching and line numbers;
<<<<<<< ours
- completion for `elements`, normalized element fields, the exact label strings used by the active page's page-label quick filters, and common JSONata functions;
=======
- completion for `elements`, normalized element fields, the exact label strings observed on the active page, and common JSONata functions;
>>>>>>> theirs
- debounced compile-only syntax diagnostics while editing; and
- runtime/resource/UI-shape diagnostics on Apply.

Keep editor concerns in a small `JsonataQueryEditor` component and a separately tested `jsonataCodeMirror` extension. Cancel pending diagnostic debounce work on draft change/unmount. Clamp JSONata error positions to the document length and highlight the token when available.

The UI should expose loading/evaluating state, disable duplicate Apply submissions, and retain the draft text after an error so it can be corrected.

## 5. Active-Page Derivations

After a successful UI query, use canonical `matchedElements` as the only input to:

- the “N of M total” count;
<<<<<<< ours
- cards;
- empty state;
- sorting; and
- the active-page frequency graph.

Preserve taxonomy-aware behavior for the Training and Racing presets: their graph and “most frequent” sort count recognized child labels and omit parent-category labels. All, page-label quick filters, and Custom count all exact labels in the matched set.
=======
- cards, in exact query-result order;
- empty state;
- the active-page frequency graph.

Preserve taxonomy-aware graph behavior for the three mistake templates: count recognized child labels and omit parent-category labels. All results and Custom count all exact labels in the matched set.

Delete the active-page filter selector, sort selector, `sortMode`, and `sortAnalysisResults`. There is no post-query sort layer: predefined templates encode their order in JSONata, and a custom query's returned order is the displayed order. Stable tie-breakers for predefined templates therefore belong in their generated expressions and their tests.

The active-page taxonomy maps may still be used to generate mistake-template expressions and their frequency graph, but they must not filter or reorder `matchedElements` after evaluation. Build the combined Training-and-Racing recognition data once and share it with template generation and graph aggregation; build the selected single-category recognition catalog for the Overall Trends expression separately.
>>>>>>> theirs

Rename active-page concepts that are no longer mistake-specific:

- `filteredElements` -> `matchedElements`;
<<<<<<< ours
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
=======
- active-page `buildMistakeFrequencyData` -> a generic label-frequency helper;
- `getMistakeFrequencyGraphHeight` -> `getLabelFrequencyGraphHeight`;
- active-page `mistakeFrequency*` variables/test IDs/text -> `labelFrequency*`;
- remove the active-page `graphSubject`/Training-or-Racing graph copy in favor of query-aware label-frequency copy; and
- active-page “No Training/Racing Mistake results” copy -> query-aware empty-state copy.

## 6. Overall Trends Through JSONata

The current chart shares `mainLabelFilter` between active-page filtering and Overall Trends and builds trend counts with `buildMistakeTrendData`. Replace both arrangements after the shared evaluator is working:

- Rename the remaining trend taxonomy type/options/state to `MistakeTrendParent`, `TREND_PARENT_OPTIONS`, and `trendParent`. Drop `sortDisplayName`; it existed only for the deleted active-page sort control.
- Build one complete Overall Trends JSONata expression for each Training/Racing selection. The generator receives the applicable recognized parent aliases plus canonical child IDs, fallback names, and resolved names. It escapes/deduplicates every literal exactly as the active-page template generator does.
- Evaluate the selected expression against `{ pages }`, not against the active page. The query sorts by `createdAt` and `sourceIndex`, emits the lap-time row for every page, selects recognized parent-mistake elements, counts each recognized child category at most once per element, emits zero counts where a category is absent on a lap, and emits cross-page occurrence totals.
- Keep the lap-time series inside the same complete trend expression so every data series on Overall Trends comes from the same normalized root and evaluation generation. Do not special-case the lap-time panel with a second raw-pages data path.
- Validate the result with the strict Overall Trends contract before committing it. Use one monotonically increasing trend generation so stale evaluations from a rapid Training/Racing change, taxonomy refresh, or page update cannot win.
- On Training/Racing selection, taxonomy resolution, or retained-page changes, regenerate and evaluate the selected trend expression. Preserve the last valid result only while its input generation is still current; if evaluation for new pages fails, show one actionable trend error and fail closed to an empty trend result rather than graphing rows retained from old pages.
- Feed `buildLapTimeTrendData` (narrowed to consume validated lap rows), trend-direction helpers, graph specs, and category selection from the validated query result. Category selection keeps the current ID when it still exists and otherwise selects the highest-occurrence category with the existing deterministic label tie-breaker.
- Keep `trendParent` fully independent from active-page template/custom-query state. Active-page selection must not regenerate the trend expression, and trend selection must not overwrite the active query.
- Keep the generated Overall Trends expression internal in this iteration. The compact selector is the authoring surface; the expression must still be exported/testable from the shared query module so it cannot drift into hidden React filtering.

This preserves the recognizable Overall Trends UI while moving all cross-page ordering, taxonomy selection, and count aggregation to JSONata.

## 7. Legacy Cleanup After Both JSONata Migrations

Start this phase only after the active-page and Overall Trends JSONata paths are implemented, tested, and driving the UI. Delete the legacy paths rather than retaining fallback behavior.
>>>>>>> theirs

Delete rather than deprecate or wrap:

- `AnalysisResultQueryData` and its `result_count`/`mistake_count` keyed generic types;
- enum-specific `validateAnalysisResultQueryArguments` branches and messages;
- the chart's `analysisResultQueryData` count memo and indexed lookup;
- `countAnalysisResultMistakes` once its query-only caller is removed;
- backend enum/schema descriptions and enum-based registry expectations;
<<<<<<< ours
- old active-page `mainLabelFilter` filtering, select, and Practice/Racing-only empty-state logic;
- old active-page mistake-frequency naming and the local sort-mode value it made obsolete; and
=======
- old active-page `mainLabelFilter` filtering, `selectedFilter`, selector, and Practice/Racing-only count/empty-state/graph copy;
- `AnalysisResultsSortMode`, `sortMode`, `sortAnalysisResults`, `IndexedAnalysisResult`, `sortedElements`, the separate active-page sort selector, and obsolete sort-specific test expectations;
- `MainLabelFilterOption.sortDisplayName` and category-specific sort-option wording;
- the old active-page `recognizedParentLabels`/single-category `recognizedSubLabels` path as a second filter/sort implementation; replace it with combined taxonomy data used only by template generation and graph aggregation;
- `buildMistakeTrendData` and its imperative per-page parent filtering, sub-label loops, category totals, and sorting once the strict Overall Trends result adapter is in use;
- the Overall Trends `recognizedParentLabels`/`recognizedSubLabels` filtering memos and any raw-`chronologicalPages` count derivation that duplicates the generated trend expression;
- the shared `mainLabelFilter`, `AnalysisResultsMainLabelFilter`, `MAIN_LABEL_FILTER_OPTIONS`, `selectedFilter`, and `graphSubject` names after the retained selector/state/copy have moved to trend-specific names;
- any separate lap-time raw-pages memo that bypasses the complete trend query; retain the regression/presentation helper only after narrowing it to validated query rows;
- old active-page mistake-frequency names, including `buildMistakeFrequencyData`, `getMistakeFrequencyGraphHeight`, `mistakeFrequency*`, and the `mistake-frequency-graph` test ID;
- obsolete `.sortControl`/`.sortSelect` CSS and the two-column responsive rule that existed for the filter-plus-sort pair; rename generic filter CSS where necessary so the remaining Overall Trends selector and new View selector have clear ownership;
- `selectSortMode`, active-page uses of `selectMainLabel`, and tests asserting `Original order`, `most-frequent-sub-label`, `most-time-lost`, or dynamic Training/Racing sort wording; and
>>>>>>> theirs
- tests/mocks that expect the aliases to return numbers or expect only Training/Racing active-page options.

Keep only where still semantically owned:

<<<<<<< ours
- `getAnalysisResultMistakeParentLabels`, because presets and Overall Trends still need taxonomy aliases;
- mistake terminology inside the cross-page Overall Trends implementation;
- `mistake_count` fields in `user-summary-ai-tools.ts`, which are a separate user-summary data contract; and
=======
- `getAnalysisResultMistakeParentLabels`, because predefined templates and Overall Trends still need taxonomy aliases;
- taxonomy child-resolution and label-comparison behavior where it is still used by expression generation or active-page label-frequency aggregation, but never as a post-query card sorter or parallel trend aggregator;
- the Overall Trends Training/Racing selector, renamed to trend-specific concepts and wired only to generated JSONata expressions;
- lap-time regression, trend-direction, graph-spec, accessibility-copy, and category-selection presentation helpers that consume validated JSONata output;
- mistake terminology inside the cross-page Overall Trends implementation;
- `mistake_count` fields in `user-summary-ai-tools.ts`, which are a separate user-summary data contract; and
- the Goal determination test that rejects a legacy top-level `mistake_count`; it is a negative envelope regression, not alias support;
>>>>>>> theirs
- `analysisResultsModel` input normalization for existing upstream array/`items`, snake_case, and camelCase analysis payloads. Removing ingestion compatibility is a separate migration and is not required by the query API clean break.

After implementation, search production frontend/backend code for `result_count` and `mistake_count`. The analysis-result aliases should have no production matches; unrelated user-summary fields and an explicit negative no-alias regression test are acceptable.

## 8. Implementation Order

<<<<<<< ours
1. Add dependencies and the pure evaluator/types/error normalizer.
2. Add evaluator, JSON-safety, UI-result resolution, preset-generation, and no-alias unit tests.
3. Replace frontend AI query types/validation/dispatch with the expression contract.
4. Update backend registry schema, descriptions, goal guidance, and tests.
5. Split trend state from active-page state.
6. Add query presets/state and drive all active-page derivations from `matchedElements`.
7. Add the CodeMirror editor and diagnostics.
8. Remove orphaned legacy helpers/names/tests and run the repository-wide alias search.
=======
1. Add dependencies and the pure evaluator, input/output types, input normalizers, JSON-safety checks, and error normalizer.
2. Add evaluator, UI element-result resolution, active-page template generation/ordering, and no-alias unit tests.
3. Replace frontend AI query types/validation/dispatch with the expression contract.
4. Update backend registry schema, descriptions, goal guidance, and tests.
5. Add the single combined-template selector/query state and drive all active-page derivations from ordered `matchedElements`.
6. Add the active-page CodeMirror editor and diagnostics.
7. Add the normalized `{ pages }` root, strict trend-result validator, generated Training/Racing trend expressions, and focused tests.
8. Split `trendParent` from active-page state and switch every Overall Trends data series to the validated JSONata result, including page/taxonomy refresh and stale-generation behavior.
9. Run active-page and Overall Trends migration tests with the legacy implementations still available only as test oracles where useful; compare results across representative multi-page fixtures.
10. Remove the orphaned active-page filter/sort code and imperative Overall Trends aggregation listed in the cleanup checklist, delete the temporary test oracles, then run repository-wide alias and legacy-symbol searches.
>>>>>>> theirs

## 9. Test Plan

### Pure evaluator

- scalar, object, array, projection, aggregation, label filtering, singleton collapse, empty sequence -> `null`, and comments;
<<<<<<< ours
=======
- isolation and shape validation for both `{ elements }` and `{ pages }` roots, including normalized `lapTimeMs` and deterministic `sourceIndex` handling;
>>>>>>> theirs
- rejection of non-JSON-safe/cyclic normalized input as well as unsafe output;
- syntax/runtime errors with code/position/token;
- timeout, stack, sequence, and JSON-safe result failures;
- function-valued, cyclic/non-finite/nested-undefined outputs;
- IDs, projected objects, unknown IDs, mixed invalid results, duplicates, ordering, and `null` UI results;
<<<<<<< ours
- special-character/Unicode label escaping, exact-value deduplication, case-distinct values, and collisions with built-in display names; and
- `result_count`/`mistake_count` produce no legacy count behavior.

=======
- special-character/Unicode taxonomy-value escaping, exact-value deduplication, case-distinct values, all four template results, and deterministic common-label/time-lost ordering; and
- `result_count`/`mistake_count` produce no legacy count behavior.

### Overall Trends queries

- Training and Racing expressions each run against multi-page fixtures with duplicate timestamps, missing lap times, empty pages, parent aliases, resolved child names, and labels that contain quotes/Unicode;
- pages are emitted in `createdAt`/`sourceIndex` order, every page has one lap row, and absent categories produce zero per-lap counts;
- a category label appearing more than once on one element counts once, while separate matching elements count separately;
- total parent-mistake counts, per-category lap counts, and cross-page occurrence totals match the imperative fixture oracle before that oracle is deleted;
- strict rejection of unknown/missing/duplicate pages, unknown/duplicate categories, invalid/negative/fractional counts, inconsistent totals, and malformed nesting;
- Training/Racing changes, taxonomy refreshes, retained-page updates, evaluator errors, and stale async completions commit or fail closed as specified; and
- lap-time regression/direction and category fallback selection consume validated JSONata rows without reading raw pages.

>>>>>>> theirs
### AI/backend

- missing, blank, non-string, and extra arguments;
- actual JSON-safe scalar/object/array/null output;
- structured query error details survive the frontend failed-tool frame;
- backend schema has no enum and advertises expression examples in every existing eligible session context; and
- goal determination accepts finite numeric `data` and rejects nonnumeric query results.

### UI

<<<<<<< ours
- All, Training, Racing, every distinct exact active-page label, and Custom, including the separate **Page labels** grouping;
- preset selection, dirty draft, Apply, Ctrl+Enter, Reset, and last-valid preservation;
- compile debounce, runtime/resource/UI-shape diagnostics, focus, and stale async result suppression;
- active-page switching, element append/update/remove, page-label quick-filter rebuilding, disappearing selected page labels becoming Custom, and taxonomy refresh;
- count, cards, empty state, Original/label-frequency/time-lost sorting, and frequency graphs all use `matchedElements`; and
- Overall Trends still opens/navigates as before and its Training/Racing state is independent.
=======
- the four fixed combined templates and Custom in one selector, with no separate filter or sort control;
- template selection, dirty draft, Apply, Ctrl+Enter, Reset, and last-valid preservation;
- compile debounce, runtime/resource/UI-shape diagnostics, focus, and stale async result suppression;
- active-page switching, element append/update/remove, taxonomy-dependent template regeneration, and taxonomy refresh without rewriting Custom;
- count, cards, empty state, query-defined card order, and frequency graphs all use `matchedElements`;
- the active page has no `Sort by` control and no Training/Racing `Showing` control, while Overall Trends still has its independent Training/Racing selector backed by generated JSONata;
- “Most common label in mistakes” filters to recognized mistakes and applies its frequency ordering as one operation, while “Most time lost in mistakes” does the same for time loss; and
- Overall Trends still opens/navigates as before, every plotted series comes from the validated trend query result, and its Training/Racing state is independent.
>>>>>>> theirs

## 10. Verification

Run from `acla_front`:

```powershell
npx tsc --noEmit
npm test -- --watchAll=false --runInBand AnalysisResultsChart analysisResultsQuery ai-command-registry Goal use-voice-conversation
npm run build
npm ls jsonata @codemirror/state @codemirror/view @codemirror/language @codemirror/autocomplete @codemirror/lint
```

Run the targeted backend registry/controller tests from `acla_backend`, then run the normal backend test suite if the targeted set passes.

<<<<<<< ours
Finally verify the lockfile has one compatible CodeMirror 6 dependency tree and search both projects for obsolete alias/filter production code.

## Assumptions

- Query scope is always the currently active normalized Analysis Results page, even while the pagination UI is displaying Overall Trends.
- Custom expressions apply explicitly, not live while typing.
- Page-label quick filters are ephemeral, active-page-only shortcuts over exact normalized label strings; they are not saved user presets or taxonomy aliases.
- The active-page UI accepts only results resolvable to active element IDs; the AI tool remains unrestricted to any JSON-safe result.
=======
Finally verify the lockfile has one compatible CodeMirror 6 dependency tree and search both projects for obsolete alias/filter production code. In the frontend, production matches for `AnalysisResultsSortMode`, `sortAnalysisResults`, `IndexedAnalysisResult`, `sortedElements`, `sortMode`, `sortControl`, `sortSelect`, `mistake-frequency-graph`, and `buildMistakeTrendData` must be zero. `mainLabelFilter`, `AnalysisResultsMainLabelFilter`, `MAIN_LABEL_FILTER_OPTIONS`, and imperative Overall Trends `recognizedParentLabels`/`recognizedSubLabels` filtering should also be zero after the trend-specific JSONata migration. Classify any remaining `result_count`, `mistake_count`, or `filteredElements` match rather than deleting it blindly: the documented user-summary fields, negative regressions, and unrelated local transformations are allowed.

## Assumptions

- AI and custom-editor query scope is always the currently active normalized Analysis Results page, even while the pagination UI is displaying Overall Trends. Generated Overall Trends queries are the only queries that receive the normalized cross-page `{ pages }` root.
- Custom expressions apply explicitly, not live while typing.
- The built-in catalog is intentionally fixed and small. Individual labels are available through custom JSONata and editor completion, not generated dropdown choices.
- The active-page UI accepts only results resolvable to active element IDs; the AI tool remains unrestricted to any JSON-safe result.
- Overall Trends expressions are generated, fixed-shape queries selected by Training/Racing; arbitrary custom trend expressions and a trend editor are outside this iteration.
>>>>>>> theirs
- This migration intentionally breaks the old analysis-result query aliases and active-page filter implementation. It does not rename unrelated domain fields or remove upstream analysis-payload normalization.
- Existing overlapping worktree changes are preserved and extended without resetting or cleaning unrelated edits.
