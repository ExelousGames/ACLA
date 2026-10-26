'use strict';

const { assertLiveTelemetryRow, isPlainObject } = require('../../../../src/data/live-telemetry-dataset');

function flattenFields(source, sample, prefix = '') {
  for (const [key, value] of Object.entries(source)) {
    if (key.startsWith('_')) continue;
    const field = prefix ? `${prefix}_${key}` : key;
    if (isPlainObject(value)) {
      flattenFields(value, sample, field);
      continue;
    }
    // Preserve the extractor's ASCII/NUL cleanup and omission of blank fields.
    const cleaned = typeof value === 'string' ? value.replace(/[^\x01-\x7f]/g, '') : value;
    if (cleaned == null
      || (typeof cleaned === 'string' && !cleaned.trim())
      || (Array.isArray(cleaned) && cleaned.length === 0)) continue;
    sample[field] = cleaned;
  }
}

function swapYZAxes(sample) {
  // Normalize every ACC vector from (x, y, z) to (x, z, y) at the adapter boundary.
  return Object.fromEntries(Object.entries(sample).map(([field, value]) => {
    if (field === 'Graphics_car_coordinates') {
      return [field, value.map(({ x, y, z }) => ({ x, y: z, z: y }))];
    }
    const adjustedField = field.replace(/_([yz])$/, (_, axis) => axis === 'y' ? '_z' : '_y');
    return [adjustedField, value];
  }));
}

class AccAdapter {
  adapt(packet, elapsedMilliseconds) {
    if (!isPlainObject(packet) || packet.type !== 'sample'
      || !isPlainObject(packet.sharedMemory) || !isPlainObject(packet.broadcasting)) {
      throw new TypeError('ACC capture must contain shared-memory and Broadcasting objects.');
    }

    let sample = {};
    flattenFields(packet.sharedMemory, sample);
    const { positions, sessionTime } = packet.broadcasting;
    if (sample.Graphics_status === 2 && positions != null) {
      sample.Graphics_normalized_positions = positions;
      if (typeof sessionTime === 'number' && Number.isFinite(sessionTime) && sessionTime >= 0) {
        sample.Graphics_game_time = sessionTime / 1000;
      }
    }
    // Validate source fields before a timestamp can make an empty capture look valid.
    assertLiveTelemetryRow(sample);
    sample = swapYZAxes(sample);
    if (elapsedMilliseconds !== undefined) {
      sample.Graphics_analyst_session_time = elapsedMilliseconds / 1000;
    }
    return assertLiveTelemetryRow(sample);
  }
}

module.exports = { AccAdapter };
