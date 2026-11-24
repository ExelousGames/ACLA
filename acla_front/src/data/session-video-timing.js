// The last box in a synchronized MP4 is a UUID box containing the telemetry
// reader time, in seconds, at the first video frame. The file stays playable
// by MP4 readers that ignore application metadata.
const VIDEO_TIMING_BYTES = 32;
const HEADER = Uint8Array.of(
  0, 0, 0, VIDEO_TIMING_BYTES, 0x75, 0x75, 0x69, 0x64, // size and 'uuid'
  0x5a, 0xf9, 0x4e, 0x89, 0x71, 0xb5, 0x4c, 0xc4, 0x80, 0x66, 0x76, 0x6b, 0x62, 0x35, 0x9f, 0x85,
);

/** @param {number} telemetryStartSeconds */
function encodeVideoTelemetryStart(telemetryStartSeconds) {
  if (!Number.isFinite(telemetryStartSeconds) || telemetryStartSeconds <= 0) {
    throw new TypeError('Invalid video telemetry start time.');
  }
  const bytes = new Uint8Array(VIDEO_TIMING_BYTES);
  bytes.set(HEADER);
  new DataView(bytes.buffer).setFloat64(HEADER.length, telemetryStartSeconds);
  return bytes;
}

/** @param {Uint8Array} bytes */
function readVideoTelemetryStart(bytes) {
  if (bytes.byteLength !== VIDEO_TIMING_BYTES || !HEADER.every((byte, index) => bytes[index] === byte)) return null;
  const seconds = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getFloat64(HEADER.length);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
}

module.exports = { VIDEO_TIMING_BYTES, encodeVideoTelemetryStart, readVideoTelemetryStart };
