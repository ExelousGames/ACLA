'use strict';

const { MIGRATION: YZ_MIGRATION, main } = require('./migrate-acc-yz');

const MIGRATION = 'acc_orientation_v1';
const ROTATION_FIELDS = ['Physics_heading', 'Physics_pitch', 'Physics_roll',
  'Physics_local_angular_vel_x', 'Physics_local_angular_vel_y', 'Physics_local_angular_vel_z'];

function convertRow(row) {
  if (!row || typeof row !== 'object' || Array.isArray(row)) {
    throw new Error('Telemetry rows must be objects.');
  }
  const converted = { ...row };
  // The previous migration already applied P(x,y,z)=(x,z,y). det(P)=-1:
  // Euler angles change sign and axial angular velocity uses -P*w.
  // Only the signs remain to fix; swapping axes again would undo acc_yz_v1.
  for (const field of ROTATION_FIELDS) {
    if (!Object.hasOwn(row, field) || row[field] === null) continue;
    if (!Number.isFinite(row[field])) throw new Error(`Invalid ${field}; refusing to coerce rotation data.`);
    converted[field] = row[field] === 0 ? 0 : -row[field];
  }
  return converted;
}

function transform(buffer) {
  const rows = JSON.parse(buffer.toString('utf8'));
  if (!Array.isArray(rows)) throw new Error('Telemetry chunk must be a JSON array.');
  return { buffer: Buffer.from(JSON.stringify(rows.map(convertRow))), rows: rows.length };
}

const orientationMigration = {
  id: MIGRATION,
  transform,
  validateSession(session) {
    const previous = session.telemetryMigrations?.[YZ_MIGRATION];
    if (!previous || !Array.isArray(previous.originalChunkFileIds)
      || previous.originalChunkFileIds.length !== session.dataChunkFileIds?.length
      || (session.totalDataPoints != null && previous.rows !== session.totalDataPoints)) {
      throw new Error(`Session ${session._id}: requires a completed ${YZ_MIGRATION}; coordinate convention is unknown.`);
    }
  },
  validateFile(file, session, before, index) {
    const source = file?.metadata?.telemetryMigration;
    const originalId = session.telemetryMigrations[YZ_MIGRATION].originalChunkFileIds[index];
    if (!file || !(file.uploadDate instanceof Date) || file.uploadDate >= before
      || source?.id !== YZ_MIGRATION || !source.sessionId?.equals(session._id)
      || !source.sourceFileId?.equals(originalId)) {
      throw new Error(`Session ${session._id}: missing, recent, or mismatched ${YZ_MIGRATION} chunk at index ${index}.`);
    }
  },
};

if (require.main === module) main(orientationMigration, 'migrate-acc-orientation.js').catch(error => {
  console.error(error.message.replace(/mongodb(?:\+srv)?:\/\/\S+/g, '[redacted URI]'));
  process.exitCode = 1;
});

module.exports = { MIGRATION, ROTATION_FIELDS, convertRow, transform, orientationMigration };
