# JSONata Analysis-Result Queries

## Summary

Replace the fixed `result_count`/`mistake_count` query enum with real JSONata expressions evaluated against the normalized active analysis page:

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

Use the same evaluator for AI queries, predefined filters, observed-label filters, and custom UI expressions.

## Implementation Changes

- Add the official [`jsonata`](https://www.npmjs.com/package/jsonata) evaluator. Add [`@uiw/react-codemirror`](https://www.npmjs.com/package/%40uiw/react-codemirror) with `@codemirror/language`, `@codemirror/autocomplete`, and `@codemirror/lint` for the rich editor.
- Implement a local JSONata CodeMirror extension because no maintained JSONata-specific editor package exists:
  - JSONata syntax highlighting, bracket matching, line numbers, and comments.
  - Completion for `elements`, analysis-element fields, observed labels, and common JSONata functions.
  - Debounced syntax diagnostics and runtime diagnostics on Apply.
  - Use CodeMirror’s documented custom-language APIs rather than adding a niche grammar package. [CodeMirror language-package guidance](https://codemirror.net/examples/lang-package/)
- Centralize JSONata compilation/evaluation:
  - Use active normalized `{ elements }` only.
  - Apply JSONata 2.2 guardrails: 100 ms execution limit, stack depth 64, and generated range/sequence limit 10,000.
  - Return JSON-safe values; normalize an undefined result to `null`, and reject functions or other non-serializable output.
  - Preserve structured JSONata error position/code information for editor diagnostics and AI tool errors.

## UI and Tool Contracts

- Change `query_analysis_result` to accept exactly `{ query: string }` and return `{ status: "ready", data: JsonValue }`, where `data` is the actual JSONata scalar, object, or array.
- Replace the backend enum schema and descriptions with the expression contract and examples such as `$count(elements)` and `elements[labels[$ = "Lockup"]].{ "id": id, "section": section }`.
- Remove the legacy `result_count` and `mistake_count` aliases; callers use `$count(elements)` and label predicates instead. Numeric goal determinations must supply an expression that returns a number.
- Replace hard-coded active-page filtering with JSONata presets:
  - All: `elements`
  - Training and Racing: generated predicates containing the recognized parent-label IDs/names.
  - One alphabetically sorted preset per distinct label observed on the active page, with JSONata-safe escaping.
  - Default remains Training to preserve current initial behavior.
- Add an “Edit query” CodeMirror panel. Selecting any preset displays its exact expression; editing creates a custom draft. Apply or Ctrl+Enter commits it, while Reset restores the selected preset. Invalid drafts retain the last successfully applied card list.
- For UI filtering, accept JSONata results containing active-page element objects or their IDs, map them back to canonical elements, remove duplicate IDs while preserving order, and treat null/undefined as zero matches. Reject projections that cannot identify active elements.
- Re-evaluate the applied expression when active-page data changes. Refresh observed-label presets accordingly.
- Drive the result count, cards, frequency graph, and sorting from the matched elements. Preserve current taxonomy-specific graph behavior for Training/Racing; All, observed-label, and custom queries calculate frequencies from labels in the matched set.
- Keep the cross-page Overall Trend screen on its existing Training/Racing controls because JSONata scope is explicitly the active page.

## Test Plan

- Unit-test JSONata scalar, projection, aggregation, label filtering, singleton/sequence behavior, undefined normalization, escaping, guardrail failures, and structured syntax/runtime errors.
- Test AI validation for empty queries, extra arguments, actual JSON-safe return values, and numeric goal determinations.
- Test every static preset and dynamically observed label, duplicate display names, special-character labels, active-page switching, and result mutations.
- Test the CodeMirror editor’s preset synchronization, completions, diagnostics, Apply/Ctrl+Enter/Reset behavior, and preservation of the last valid filter.
- Confirm Training/Racing counts, cards, sorting, frequency graphs, and Overall Trend behavior remain compatible.
- Run frontend type-check/tests and backend tool-registry/controller tests; verify the lockfile contains one compatible CodeMirror dependency tree.

## Assumptions

- Query scope is the currently active analysis-results page.
- AI queries return the actual JSONata result, while UI filters additionally require resolvable element objects or IDs.
- Custom filters apply explicitly, not live while typing.
- Existing overlapping worktree changes are preserved and extended without resetting or cleaning unrelated edits.
