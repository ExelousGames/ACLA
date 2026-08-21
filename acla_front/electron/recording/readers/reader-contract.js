'use strict';

const { validateSourceFrame } = require('../telemetry-contract');

function validateReaderEvent(event, expectedGame) {
  if (!event || typeof event !== 'object' || Array.isArray(event)) {
    return { ok: false, error: 'Reader event must be an object.' };
  }
  if (event.type === 'fatal') {
    return typeof event.error === 'string' && event.error.length > 0
      ? { ok: true, value: event }
      : { ok: false, error: 'Reader fatal event requires an error message.' };
  }
  if (event.type === 'frame') return validateSourceFrame(event.frame, expectedGame);
  return { ok: false, error: `Unknown reader event type: ${String(event.type)}` };
}

function assertReader(reader) {
  if (!reader || typeof reader !== 'object'
    || typeof reader.game !== 'string'
    || typeof reader.start !== 'function'
    || typeof reader.stop !== 'function') {
    throw new TypeError('Telemetry reader does not implement the reader lifecycle contract.');
  }
  return reader;
}

module.exports = { assertReader, validateReaderEvent };
