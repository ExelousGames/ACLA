# Renderer audio

Import the singleton from `services/audio`. Keep prepared URLs, `TtsPack` objects,
and replay decisions in the component that owns them. Stop that component's
handles when replacing its content or unmounting.

```ts
const playback = audioManager.play({
    url: savedPack.audioDataUrl,
    type: 'voice',
    priority: 50,
    volume: 0.9,
    onComplete: (outcome) => { /* update this caller's presentation */ },
});
const outcome = await playback.finished;
```

`finished` always resolves with `completed`, `replaced`, `rejected`, `cancelled`,
or `failed` (including an `Error`). `stop()` cancels just that handle. Terminal
notifications run asynchronously, once, after resources have been released.

There is one admitted request per type (`voice`, `music`, `alert`). A higher
priority, or a newer equal priority, replaces the incumbent. Lower priorities
are rejected. Other types continue independently. Audible voice multiplies
music's configured volume by 0.2; voice becoming idle or terminal restores it.

```ts
const stream = audioManager.createStream({
    type: 'voice', priority: 50, format: 'pcm16', sampleRate: 24000,
    onStart: () => { /* audible */ },
    onIdle: () => { /* no longer audible; still admitted */ },
});
stream.enqueuePcm16(chunk); // Mono little-endian PCM16 ArrayBuffer; copied on enqueue.
stream.finish();            // Close input and finish after all scheduled audio ends.
await stream.finished;
```

For sequential base64 WAV chunks, use `format: 'wav'` and
`enqueueBase64Wav(base64)`. Enqueue methods return `false` for invalidated or
finished handles. Streams first compete when a nonempty chunk arrives and keep
admission through silence until `finish()`, `stop()`, replacement, or failure.
Discarded streams never compete again. Explicit replay requires a new handle;
the caller's saved assets remain intact.

The voice conversation hook owns one stream until stop/restart. It retains
microphone capture and networking, and never recreates a discarded stream merely
because more output arrives. Voice callers default to priority 50 and expose
priority/volume overrides.

`DriverExpertComparisonPresentation` renders display data only. The overlay
emits `replay_started` and `replay_complete`; `AnalysisResultsChart` owns cached
narration, its handle, and the operation that waits for both visual completion
and any audio terminal outcome. Static completion without a start event skips
narration. Local narrated graphs use `Tts`, which preserves `play()`/`stop()`.
No audio asset is published in the overlay snapshot and no audio IPC is needed.

## Verification

Run the audio, TTS, comparison/chart, overlay, voice and workflow tests and the
TypeScript checker:

```sh
npm test -- --watchAll=false --runInBand --testPathPattern='audio-manager|tts|DriverExpertComparison|AnalysisResultsChart|use-voice-conversation|AiOverlayManager|overlay-display-client|WorkflowPanel'
npx tsc --noEmit
```

For a manual listening check, prepare a comparison's narration in the main
window, display it in the overlay, and minimize the main window. Confirm speech
starts once with the animation, remains audible while minimized, and stops when
the presentation is replaced or cancelled. Repeat with reduced motion (no
speech), and with an explicit replay of the retained asset. In a browser, start
playback through a user gesture and verify the same arbitration and restoration.
