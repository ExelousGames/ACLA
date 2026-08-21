'use strict';

const DESKTOP_GAMES = Object.freeze(['acc', 'ac', 'iracing']);
const DESKTOP_GAME_SET = new Set(DESKTOP_GAMES);

const START_FAILURE_TYPES = new Set([
  'malformed-recording-game',
  'unknown-recording-game',
  'unsupported-recording-game',
]);

function recordingStartFailure(type, message) {
  if (!START_FAILURE_TYPES.has(type)) {
    throw new TypeError(`Unknown recording start failure type: ${String(type)}`);
  }
  if (typeof message !== 'string' || !message.trim()) {
    throw new TypeError('Recording start failure message must be a non-empty string');
  }
  return { ok: false, error: { type, message } };
}

function validateRecordingStartConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)
    || Object.keys(config).length !== 1
    || typeof config.game !== 'string' || !config.game.trim()) {
    return recordingStartFailure(
      'malformed-recording-game',
      'A recording game identifier is required.',
    );
  }
  if (!DESKTOP_GAME_SET.has(config.game)) {
    return recordingStartFailure(
      'unknown-recording-game',
      `The simulator identifier "${config.game}" is not recognized.`,
    );
  }
  return null;
}

function isRecordingStartFailure(value) {
  return Boolean(
    value
    && typeof value === 'object'
    && !Array.isArray(value)
    && Object.keys(value).length === 2
    && value.ok === false
    && value.error
    && typeof value.error === 'object'
    && !Array.isArray(value.error)
    && Object.keys(value.error).length === 2
    && START_FAILURE_TYPES.has(value.error.type)
    && typeof value.error.message === 'string'
    && value.error.message.length > 0,
  );
}

function isReaderLaunchConfigResult(value) {
  if (!value || typeof value !== 'object') return false;
  if (value.ok === false) {
    return value.error?.type === 'unsupported-recording-game'
      && typeof value.error.message === 'string'
      && value.error.message.length > 0;
  }
  return value.ok === true
    && value.config
    && DESKTOP_GAME_SET.has(value.config.game)
    && typeof value.config.readerEntryPath === 'string'
    && value.config.readerEntryPath.length > 0
    && value.config.readerOptions
    && typeof value.config.readerOptions === 'object'
    && !Array.isArray(value.config.readerOptions);
}

module.exports = {
  DESKTOP_GAMES,
  DESKTOP_GAME_SET,
  START_FAILURE_TYPES,
  isReaderLaunchConfigResult,
  isRecordingStartFailure,
  recordingStartFailure,
  validateRecordingStartConfig,
};
