'use strict';

const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const validLocation = (lat, lon, alt) => [lat, lon, alt].every(finite)
  && Math.abs(lat) <= 90 && Math.abs(lon) <= 180;
const RADIANS = Math.PI / 180;
const WGS84_A = 6378137;
const WGS84_F = 1 / 298.257223563;
const WGS84_E2 = WGS84_F * (2 - WGS84_F);

// WGS84 geodetic -> Earth-centered, Earth-fixed meters. Altitude uses the
// simulator's reported datum for both car and reference (no assumed geoid offset).
// https://proj.org/en/stable/operations/conversions/cart.html
function ecef(lat, lon, alt) {
  const phi = lat * RADIANS, lambda = lon * RADIANS;
  const sinPhi = Math.sin(phi), cosPhi = Math.cos(phi);
  const radius = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinPhi * sinPhi);
  return {
    x: (radius + alt) * cosPhi * Math.cos(lambda),
    y: (radius + alt) * cosPhi * Math.sin(lambda),
    z: (radius * (1 - WGS84_E2) + alt) * sinPhi,
  };
}

function measurement(value, units) {
  if (finite(value)) return value;
  const match = typeof value === 'string'
    && value.match(/^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(deg|m)\s*$/);
  return match && units.includes(match[2]) ? Number(match[1]) : undefined;
}

function createPositionReference(weekend = {}) {
  // Native session YAML labels TrackLatitude/TrackLongitude with "m" even
  // though their values are decimal degrees. This exception is metadata-only;
  // the disk Lat/Lon descriptors must still declare "deg".
  const lat = measurement(weekend.TrackLatitude, ['deg', 'm']);
  const lon = measurement(weekend.TrackLongitude, ['deg', 'm']);
  const alt = measurement(weekend.TrackAltitude, ['m']);
  if (!validLocation(lat, lon, alt)) return undefined;
  return { origin: ecef(lat, lon, alt), sinLat: Math.sin(lat * RADIANS), cosLat: Math.cos(lat * RADIANS),
    sinLon: Math.sin(lon * RADIANS), cosLon: Math.cos(lon * RADIANS) };
}

function playerPosition(values, variableUnits, reference) {
  const { Lat: lat, Lon: lon, Alt: alt } = values;
  if (!reference || !validLocation(lat, lon, alt)
    || variableUnits.get('Lat') !== 'deg' || variableUnits.get('Lon') !== 'deg'
    || variableUnits.get('Alt') !== 'm') return undefined;
  const point = ecef(lat, lon, alt);
  const dx = point.x - reference.origin.x, dy = point.y - reference.origin.y, dz = point.z - reference.origin.z;
  const { sinLat, cosLat, sinLon, cosLon } = reference;
  // ECEF -> local tangent plane, ordered for the application's Y-up XYZ:
  // x = east, y = up, z = north. The origin comes only from track metadata,
  // never the first sample, so laps and separate stints share a reference.
  // https://proj.org/en/stable/operations/conversions/topocentric.html
  const position = {
    x: -sinLon * dx + cosLon * dy,
    y: cosLat * cosLon * dx + cosLat * sinLon * dy + sinLat * dz,
    z: -sinLat * cosLon * dx - sinLat * sinLon * dy + cosLat * dz,
  };
  return Object.values(position).every(finite) ? position : undefined;
}

module.exports = { createPositionReference, playerPosition };
