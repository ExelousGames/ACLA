'use strict';

const { DESKTOP_GAME_SET } = require('./recording-protocol');
const { isPlainObject, validateLiveTelemetryRow } = require('../../src/data/live-telemetry-dataset');

function validateSourceFrame(frame, expectedGame) {
  if (!isPlainObject(frame) || !DESKTOP_GAME_SET.has(frame.game)) {
    return { ok: false, error: 'Source frame must contain a recognized game.' };
  }
  if (expectedGame !== undefined && frame.game !== expectedGame) {
    return { ok: false, error: `Source frame game ${frame.game} does not match ${expectedGame}.` };
  }
  const sampleResult = validateLiveTelemetryRow(frame.sample);
  if (!sampleResult.ok) return sampleResult;
  if (Object.keys(frame).some((key) => key !== 'game' && key !== 'sample')) {
    return { ok: false, error: 'Source frame contains unsupported transport fields.' };
  }
  return { ok: true, value: frame };
}

function assertSourceFrame(frame, expectedGame) {
  const result = validateSourceFrame(frame, expectedGame);
  if (!result.ok) throw new TypeError(result.error);
  return frame;
}

module.exports = { assertSourceFrame, validateSourceFrame };
