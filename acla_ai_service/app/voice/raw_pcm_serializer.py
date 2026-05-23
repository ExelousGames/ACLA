"""Raw PCM16 mono serializer for Pipecat's WebSocket transport.

Pipecat 1.2.1's ``FastAPIWebsocketInputTransport._receive_messages`` drops
every inbound message when ``params.serializer is None`` — the
``if not self._params.serializer: continue`` guard skips raw bytes silently.
That made our previous ``serializer=None`` config a no-op: the pipeline
booted, the WS accepted, mic bytes arrived… and then sat there until the
idle-timeout fired with the LLM never seeing a single audio frame.

The wire format we want is the simplest possible: raw little-endian PCM16
mono in both directions (mic in / Kokoro out). This serializer just wraps
those bytes in/out of ``InputAudioRawFrame`` / ``OutputAudioRawFrame`` so
Pipecat's transport will actually push them into the pipeline.

Text frames are already filtered upstream by
:class:`app.api.voice._TextFilteringWebSocket` (the tool-relay channel), so
this serializer only ever sees binary payloads.
"""

from __future__ import annotations

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class RawPCMSerializer(FrameSerializer):
    """Identity-ish serializer for raw PCM16 mono audio over a WS."""

    def __init__(self) -> None:
        super().__init__()
        # Defaults match the pipeline config; overwritten in setup() once
        # Pipecat propagates the real StartFrame rates.
        self._in_sample_rate = 16000
        self._out_sample_rate = 24000

    async def setup(self, frame: StartFrame) -> None:
        self._in_sample_rate = frame.audio_in_sample_rate
        self._out_sample_rate = frame.audio_out_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        # We only forward TTS audio. Everything else (control frames,
        # transcripts, etc.) stays inside the pipeline.
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        # Text frames are routed off the audio path by _TextFilteringWebSocket
        # before reaching here, but be defensive.
        if not isinstance(data, (bytes, bytearray)):
            return None
        return InputAudioRawFrame(
            audio=bytes(data),
            sample_rate=self._in_sample_rate,
            num_channels=1,
        )
