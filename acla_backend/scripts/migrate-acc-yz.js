'use strict';

const { MongoClient, GridFSBucket } = require('mongodb');
const { parseArgs } = require('node:util');
const { finished } = require('node:stream/promises');

const MIGRATION = 'acc_yz_v1';
const MARKER = `telemetryMigrations.${MIGRATION}`;
const BUCKET = 'racing_sessions';

function swapRow(row) {
  if (!row || typeof row !== 'object' || Array.isArray(row)) {
    throw new Error('Telemetry rows must be objects.');
  }
  return Object.fromEntries(Object.entries(row).map(([field, value]) => {
    if (field === 'Graphics_car_coordinates') {
      if (!Array.isArray(value) || value.some(point => !point
        || !['x', 'y', 'z'].every(axis => Number.isFinite(point[axis])))) {
        throw new Error('Invalid Graphics_car_coordinates; refusing to discard data.');
      }
      return [field, value.map(point => ({ ...point, y: point.z, z: point.y }))];
    }
    return [field.replace(/_([yz])$/, (_, axis) => axis === 'y' ? '_z' : '_y'), value];
  }));
}

async function download(bucket, id) {
  const buffers = [];
  for await (const buffer of bucket.openDownloadStream(id)) buffers.push(buffer);
  return Buffer.concat(buffers);
}

function transform(buffer) {
  const rows = JSON.parse(buffer.toString('utf8'));
  if (!Array.isArray(rows)) throw new Error('Telemetry chunk must be a JSON array.');
  return { buffer: Buffer.from(JSON.stringify(rows.map(swapRow))), rows: rows.length };
}

function validateCounts(session, rows) {
  if (session.totalDataPoints != null && session.totalDataPoints !== rows) {
    throw new Error(`Session ${session._id}: expected ${session.totalDataPoints} rows, found ${rows}.`);
  }
}

// Originals are never deleted. New chunks are verified before a single atomic
// session update publishes the complete ordered list together with its marker.
async function migrateSession(sessions, bucket, session, files, before, log) {
  const replacements = [];
  let rows = 0;
  for (let index = 0; index < files.length; index++) {
    const file = files[index];
    const converted = transform(await download(bucket, file._id));
    const upload = bucket.openUploadStream(`${file.filename}.${MIGRATION}.json`, {
      chunkSizeBytes: file.chunkSize,
      metadata: {
        ...file.metadata,
        telemetryMigration: { id: MIGRATION, sourceFileId: file._id, sessionId: session._id },
      },
    });
    const completion = finished(upload);
    upload.end(converted.buffer);
    await completion;
    replacements.push(upload.id);
    if (!converted.buffer.equals(await download(bucket, upload.id))) {
      throw new Error(`Verification failed for new chunk ${upload.id}. Session remains unchanged.`);
    }
    rows += converted.rows;
    if ((index + 1) % 25 === 0) log({ sessionId: session._id, stagedChunks: index + 1, totalChunks: files.length });
  }
  validateCounts(session, rows);
  const result = await sessions.updateOne({
    _id: session._id,
    game_recorded_from: 'acc',
    dataChunkFileIds: session.dataChunkFileIds,
    totalDataPoints: session.totalDataPoints ?? { $exists: false },
    [MARKER]: { $exists: false },
  }, {
    $set: {
      dataChunkFileIds: replacements,
      [MARKER]: {
        before, migratedAt: new Date(), rows,
        originalChunkFileIds: session.dataChunkFileIds,
      },
    },
  });
  if (result.modifiedCount !== 1) {
    throw new Error(`Session ${session._id} changed concurrently; staged copies retained, session not overwritten.`);
  }
  return rows;
}

async function verifySession(bucket, session) {
  const marker = session.telemetryMigrations[MIGRATION];
  if (marker.originalChunkFileIds.length !== session.dataChunkFileIds.length) {
    throw new Error(`Session ${session._id}: migrated chunk count differs from backup.`);
  }
  let rows = 0;
  for (let index = 0; index < session.dataChunkFileIds.length; index++) {
    const expected = transform(await download(bucket, marker.originalChunkFileIds[index]));
    if (!expected.buffer.equals(await download(bucket, session.dataChunkFileIds[index]))) {
      throw new Error(`Session ${session._id}, chunk ${index}: migrated data differs from swapped original.`);
    }
    rows += expected.rows;
  }
  validateCounts(session, rows);
  if (rows !== marker.rows) throw new Error(`Session ${session._id}: migration row count differs.`);
  return rows;
}

async function migrate(db, { before, apply = false, verify = false, log = console.log }) {
  if (!(before instanceof Date) || !Number.isFinite(before.getTime())) throw new Error('A valid --before date is required.');
  if (apply && verify) throw new Error('Choose either --apply or --verify.');
  const sessions = db.collection('racingsessions');
  const bucket = new GridFSBucket(db, { bucketName: BUCKET });
  const documents = await sessions.find({ game_recorded_from: 'acc' }).sort({ _id: 1 }).toArray();
  const summary = { mode: verify ? 'verify' : apply ? 'apply' : 'dry-run', before: before.toISOString(),
    candidates: 0, alreadyMigrated: 0, afterCutoff: 0, chunks: 0, sourceBytes: 0, migrated: 0, verified: 0, rows: 0 };
  const plan = [];
  // Preflight every session's metadata before writing any replacement chunks.
  for (const session of documents) {
    const created = session.created_date == null ? session._id.getTimestamp() : new Date(session.created_date);
    if (!Number.isFinite(created.getTime())) throw new Error(`Session ${session._id}: invalid creation date.`);
    if (created >= before) { summary.afterCutoff++; continue; }
    if (session.telemetryMigrations && Object.hasOwn(session.telemetryMigrations, MIGRATION)) {
      summary.alreadyMigrated++;
      if (verify) plan.push({ session });
      continue;
    }
    summary.candidates++;
    if (verify) continue;
    const ids = session.dataChunkFileIds;
    if (!Array.isArray(ids) || !ids.length) throw new Error(`Session ${session._id}: no telemetry chunks.`);
    if (session.totalChunks != null && session.totalChunks !== ids.length) {
      throw new Error(`Session ${session._id}: chunk count differs from metadata.`);
    }
    const metadata = await db.collection(`${BUCKET}.files`).find({ _id: { $in: ids } }).toArray();
    const byId = new Map(metadata.map(file => [file._id.toString(), file]));
    const files = ids.map(id => {
      const file = byId.get(id.toString());
      if (!file || !(file.uploadDate instanceof Date) || file.uploadDate >= before
        || file.metadata?.telemetryMigration) {
        throw new Error(`Session ${session._id}: missing, recent, or already transformed chunk ${id}.`);
      }
      return file;
    });
    summary.chunks += files.length;
    summary.sourceBytes += files.reduce((sum, file) => sum + file.length, 0);
    plan.push({ session, files });
  }
  log({ preflight: summary });
  if (verify && summary.candidates) throw new Error(`${summary.candidates} eligible sessions have not been migrated.`);
  if (!apply && !verify) return summary;
  for (const { session, files } of plan) {
    if (verify) {
      summary.rows += await verifySession(bucket, session);
      summary.verified++;
    } else {
      summary.rows += await migrateSession(sessions, bucket, session, files, before, log);
      summary.migrated++;
    }
    log({ sessionId: session._id, migrated: summary.migrated, verified: summary.verified, rows: summary.rows });
  }
  return summary;
}

async function main() {
  const { values } = parseArgs({ options: {
    before: { type: 'string' }, apply: { type: 'boolean' }, verify: { type: 'boolean' },
    'dry-run': { type: 'boolean' }, help: { type: 'boolean' },
  } });
  if (values.help) {
    console.log('node scripts/migrate-acc-yz.js --before <ISO-8601 UTC date> [--dry-run | --apply | --verify]\n'
      + 'Defaults to metadata-only dry run. Set MONGODB_URI or backend MONGO_CLIENTNAME/MONGO_CLIENTPASSWORD/MONGO_URL.\n'
      + 'Apply retains originals and verifies every replacement before publishing each session.');
    return;
  }
  if (!values.before || !/^\d{4}-\d{2}-\d{2}T.*Z$/.test(values.before)
    || !Number.isFinite(Date.parse(values.before))) throw new Error('--before must be an explicit ISO-8601 UTC date.');
  if ([values.apply, values.verify, values['dry-run']].filter(Boolean).length > 1) throw new Error('Choose one mode.');
  const env = process.env;
  if (!env.MONGODB_URI && !(env.MONGO_CLIENTNAME && env.MONGO_CLIENTPASSWORD && env.MONGO_URL)) {
    throw new Error('Set MONGODB_URI or the backend MongoDB connection environment variables.');
  }
  const uri = env.MONGODB_URI || `mongodb://${encodeURIComponent(env.MONGO_CLIENTNAME)}:${encodeURIComponent(env.MONGO_CLIENTPASSWORD)}@${env.MONGO_URL}:27017/ACLA`;
  const client = new MongoClient(uri, { serverSelectionTimeoutMS: 10000 });
  try {
    await client.connect();
    console.log(JSON.stringify(await migrate(client.db(), {
      before: new Date(values.before), apply: values.apply, verify: values.verify,
      log: entry => console.log(JSON.stringify(entry)),
    })));
  } finally { await client.close(); }
}

if (require.main === module) main().catch(error => {
  console.error(error.message.replace(/mongodb(?:\+srv)?:\/\/\S+/g, '[redacted URI]'));
  process.exitCode = 1;
});

module.exports = { MIGRATION, MARKER, swapRow, transform, migrateSession, verifySession, migrate };
