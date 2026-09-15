'use strict';

const fs = require('fs');
const path = require('path');
const { IRacingAdapter, IRACING_VARIABLES, IRACING_FIELD_COVERAGE } = require('../electron/recording/readers/iracing/iracing-adapter');

const filename = process.argv[2];
const capture = JSON.parse(fs.readFileSync(filename, 'utf8'));
const adapter = new IRacingAdapter();
const observed = new Set();
const rows = [];
for (const packet of capture.samples) {
  // Match the production reader's allowlist before checking the actual adapter.
  const values = Object.fromEntries(Object.entries(packet.values).filter(([name]) => IRACING_VARIABLES.includes(name)));
  const row = adapter.adapt({ ...packet, values });
  Object.keys(row).forEach((field) => observed.add(field));
  rows.push(row);
}
const native = Object.keys(capture.channels);
const mapped = Object.keys(IRACING_FIELD_COVERAGE).filter((field) => IRACING_FIELD_COVERAGE[field].supported);
const last = capture.samples.at(-1)?.values || {};
const selected = (names) => Object.fromEntries(names.filter((name) => name in last).map((name) => [name, last[name]]));
const info = adapter.session;
const player = info.DriverInfo?.Drivers?.find((driver) => driver.CarIdx === info.DriverInfo.DriverCarIdx);
const session = info.SessionInfo?.Sessions?.find((item) => item.SessionNum === last.SessionNum);
const result = {
  capturedAt: capture.capturedAt, elapsedSeconds: capture.elapsedSeconds,
  advertisedHz: capture.advertisedHz, sampleCount: capture.sampleCount,
  observedTickHz: (capture.lastTick - capture.firstTick) / capture.elapsedSeconds,
  nativeChannels: native.length,
  changingChannels: native.filter((name) => capture.channels[name].changes > 0).length,
  track: info.WeekendInfo?.TrackDisplayName,
  config: info.WeekendInfo?.TrackConfigName,
  length: info.WeekendInfo?.TrackLength,
  car: player?.CarScreenName, sessionType: session?.SessionType,
  state: selected(['IsOnTrack', 'IsOnTrackCar', 'IsInGarage', 'IsReplayPlaying', 'OnPitRoad', 'PlayerTrackSurface', 'SessionState']),
  latest: selected(['Speed', 'RPM', 'Gear', 'Throttle', 'Brake', 'BrakeRaw', 'Clutch', 'FuelLevel', 'Lap', 'LapCompleted', 'AirTemp', 'TrackTempCrew', 'Precipitation', 'TrackWetness']),
  app: {
    requestedNativeChannels: IRACING_VARIABLES.length,
    presentRequestedChannels: IRACING_VARIABLES.filter((name) => native.includes(name)).length,
    missingRequestedChannels: IRACING_VARIABLES.filter((name) => !native.includes(name)),
    catalogFields: Object.keys(IRACING_FIELD_COVERAGE).length,
    declaredMappedFields: mapped.length,
    observedMappedFields: observed.size,
    latestMappedFields: Object.keys(rows.at(-1) || {}).length,
    mappedButNotObserved: mapped.filter((field) => !observed.has(field)),
    nativeChannelsNotRequested: native.filter((name) => !IRACING_VARIABLES.includes(name)),
  },
  sessionMetadataSections: Object.keys(info),
};

const format = (value) => {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : Number(value.toFixed(6)).toString();
  if (Array.isArray(value)) return `[${value.slice(0, 6).map(format).join(', ')}${value.length > 6 ? ', ...' : ''}]`;
  return String(value ?? 'unavailable').replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
};
const lines = [
  '# Measured live iRacing telemetry', '',
  `Captured: ${capture.capturedAt}; ${capture.elapsedSeconds.toFixed(2)} seconds; read-only shared memory.`, '',
  `- Car: ${result.car}; track: ${result.track}, ${result.config}; session: ${result.sessionType}.`,
  `- SDK: ${capture.nativeChannelCount} channels, ${capture.advertisedHz} Hz advertised, ${capture.sampleCount} distinct snapshots, ${result.observedTickHz.toFixed(2)} observed ticks/s.`,
  `- ${result.changingChannels} channels changed during the capture. A present or constant channel is not proof of a fresh physical measurement.`,
  `- Application requests ${IRACING_VARIABLES.length} native channels; ${result.app.presentRequestedChannels} are present.`,
  `- Application declares ${mapped.length}/${result.app.catalogFields} standardized fields mapped; ${observed.size} were emitted across the captured samples.`,
  '- Mapped counts include static metadata and derived fields. Raw and standardized counts are different schemas.',
  '- This check exercised the existing Python shared-memory reader and JavaScript adapter; it did not verify the running UI or recording writer.',
  '- Availability is specific to this car/session. A stationary capture does not establish on-track freshness for constant channels.', '',
  '## Session state', '', '```json', JSON.stringify(result.state, null, 2), '```', '',
  '## Requested native channels absent from this session', '',
  result.app.missingRequestedChannels.map((name) => `- \`${name}\``).join('\n') || 'None.', '',
  '## Mapped standard fields not emitted during this capture', '',
  result.app.mappedButNotObserved.map((name) => `- \`${name}\`: ${IRACING_FIELD_COVERAGE[name].source}`).join('\n') || 'None.', '',
  'These may require a cockpit state, a valid reference lap, race finish, a particular source channel, or longer uninterrupted history.', '',
  '## All native channels', '',
  'Array previews show the first six entries; full values and all captured samples are in the adjacent JSON file. For countAsTime channels, the existing reader returns the newest sub-sample.', '',
  '| Channel | Type/count | Unit | Latest value | Changes | SDK description |',
  '| --- | --- | --- | --- | ---: | --- |',
];
for (const name of native.sort()) {
  const channel = capture.channels[name];
  lines.push(`| ${name} | ${channel.type}[${channel.count}]${channel.countAsTime ? ' time' : ''} | ${format(channel.unit)} | ${format(channel.latest)} | ${channel.changes} | ${format(channel.description)} |`);
}
lines.push('', '## Standardized fields emitted by the existing adapter', '', '| Field | Latest value |', '| --- | --- |');
for (const field of [...observed].sort()) lines.push(`| ${field} | ${format(rows.at(-1)?.[field])} |`);
const reportPath = path.resolve(filename.replace(/\.json$/, '.md'));
fs.writeFileSync(reportPath, lines.join('\n') + '\n');
fs.writeFileSync(filename.replace(/\.json$/, '-summary.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ ...result, reportPath }, null, 2));
