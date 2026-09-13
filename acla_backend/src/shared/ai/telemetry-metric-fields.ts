export const TELEMETRY_METRIC_FIELD_DEFINITIONS = [
  {
    name: 'Physics_speed_kmh',
    description: 'Vehicle speed in kilometres per hour.',
  },
  {
    name: 'Physics_gear',
    description: 'Currently selected transmission gear.',
  },
  {
    name: 'Physics_rpm',
    description: 'Engine speed in revolutions per minute.',
  },
  {
    name: 'Physics_brake',
    description: 'Brake pedal input from 0 (released) to 1 (fully applied).',
  },
  {
    name: 'Physics_gas',
    description: 'Throttle pedal input from 0 (released) to 1 (fully applied).',
  },
  {
    name: 'Graphics_normalized_car_position',
    description:
      'Normalized car position around the lap from 0 (start line) to 1 (finish line).',
  },
] as const;

export const TELEMETRY_METRIC_FIELD_SCHEMA = {
  type: 'string',
  enum: TELEMETRY_METRIC_FIELD_DEFINITIONS.map(({ name }) => name),
  description: TELEMETRY_METRIC_FIELD_DEFINITIONS.map(
    ({ name, description }) => `${name}: ${description}`,
  ).join(' '),
} as const;
