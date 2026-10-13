# Recorded session data

Recorded Session uses source loaders plus a scoped React Context provider. The
provider is mounted once inside `SessionAnalysisProvider`, above the visualizations.

```text
Cloud download chunks -> loadCloudTelemetry -----+
                                                +-> RecordedSessionDataProvider -> consumers
Local IBT importer -> selected session's data ---+
```

`useRecordedSessionData()` returns `{ sessionId, source, status, table, message }`.
`table` is a readonly array of source rows, available when `status === 'ready'`.
An empty source is a ready empty table. Loading and error states expose no partial
rows. Components outside the React subtree can read the same snapshot through the
registered `SessionAnalysisHandle.getRecordedSessionData()` method.

```tsx
const { status, table } = useRecordedSessionData();
// Derive chart data separately; never mutate the shared table or nested values.
const frames = useMemo(
    () => status === 'ready' ? parseTelemetryFrames(table) : [],
    [status, table],
);
```

The provider does no cleaning: no filtering, sorting, deduplication, coercion,
missing-value filling, resampling, unit conversion, or derived columns. Cloud
chunks are concatenated in their source order; local rows are retained directly.
Field names, values, nested structures, duplicate rows, and source indices survive.
Readonly types describe the consumer contract; source objects are not cloned or
recursively frozen. Consumers must also treat nested values as read-only.

Map frame parsing and playback timing adjustments belong to the map's derived
view. Analysis-results consumers also read this table rather than the empty
`sessionSelected.data` field on cloud session metadata.

The table stays available while panels mount and unmount. Selecting another
session replaces it and aborts the previous download; late results are ignored.
Only the current session is retained, with no unbounded session cache. Transport
envelopes are checked, but individual rows are not cleaned or schema-validated here.

The local source boundary is the **existing IBT importer output**, which already
converts native iRacing channels into the application's standard telemetry schema.
This provider preserves that output; it does not make the existing importer a
lossless native IBT decoder. Native-channel preservation would require changing
the IBT decoder/import transport and adding separate visualization adapters.
