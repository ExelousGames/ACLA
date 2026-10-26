'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { Readable, Writable } = require('node:stream');
const { MongoClient, ObjectId } = require('mongodb');
const { MIGRATION, MARKER, swapRow, transform, migrateSession, verifySession, migrate } = require('./migrate-acc-yz');

const before = new Date('2026-10-26T00:00:00Z');
const quiet = () => {};

function fixture() {
  const source = [
    [{ Physics_velocity_x: 1, Physics_velocity_y: 2, Physics_velocity_z: -3,
      Graphics_car_coordinates: [{ x: 4, y: 5, z: 6 }] }],
    [{ Physics_local_velocity_y: 0, Physics_gear: 4 }],
  ];
  const files = source.map((rows, index) => ({
    _id: new ObjectId(), filename: `chunk-${index}.json`, chunkSize: 255 * 1024,
    uploadDate: new Date('2026-10-24T00:00:00Z'), metadata: { chunkIndex: index },
    length: Buffer.byteLength(JSON.stringify(rows)),
  }));
  const buffers = new Map(files.map((file, index) => [file._id.toString(), Buffer.from(JSON.stringify(source[index]))]));
  const session = { _id: new ObjectId(), game_recorded_from: 'acc', created_date: new Date('2026-10-24T00:00:00Z'),
    dataChunkFileIds: files.map(file => file._id), totalChunks: 2, totalDataPoints: 2,
    videos: [{ chunkFileIds: [new ObjectId()] }], points: [{ position_x: 5, position_y: 6 }] };
  let updates = 0;
  const sessions = {
    async updateOne(filter, update) {
      updates++;
      assert.equal(filter.game_recorded_from, 'acc');
      assert.deepEqual(filter[MARKER], { $exists: false });
      assert.deepEqual(filter.dataChunkFileIds, files.map(file => file._id));
      if (session.telemetryMigrations?.[MIGRATION]) return { modifiedCount: 0 };
      session.dataChunkFileIds = update.$set.dataChunkFileIds;
      session.telemetryMigrations = { [MIGRATION]: update.$set[MARKER] };
      return { modifiedCount: 1 };
    },
  };
  const bucket = {
    openDownloadStream(id) {
      const buffer = buffers.get(id.toString());
      if (!buffer) throw new Error('Missing chunk');
      return Readable.from([buffer]);
    },
    openUploadStream(filename, options) {
      assert.equal(options.metadata.telemetryMigration.id, MIGRATION);
      const parts = [];
      const stream = new Writable({
        write(chunk, encoding, done) { parts.push(Buffer.from(chunk)); done(); },
        final(done) { buffers.set(stream.id.toString(), Buffer.concat(parts)); done(); },
      });
      stream.id = new ObjectId();
      return stream;
    },
  };
  return { source, files, buffers, session, sessions, bucket, updates: () => updates };
}

test('swaps all vectors and car coordinates while preserving unrelated fields and inputs', () => {
  const row = { Physics_heading: 1, Physics_pitch: 2, Physics_roll: 3, Physics_speed_kmh: 100,
    Graphics_car_coordinates: Array.from({ length: 60 }, (_, i) => ({ x: i, y: -i, z: i + 2 })),
    Graphics_normalized_positions: { 3: 0.25 } };
  for (const vector of ['velocity', 'local_velocity', 'g_force', 'local_angular_vel',
    ...['tyre_contact_point', 'tyre_contact_normal', 'tyre_contact_heading'].flatMap(name =>
      ['front_left', 'front_right', 'rear_left', 'rear_right'].map(wheel => `${name}_${wheel}`))]) {
    Object.assign(row, { [`Physics_${vector}_x`]: -1, [`Physics_${vector}_y`]: 0, [`Physics_${vector}_z`]: -3 });
  }
  const original = structuredClone(row);
  const result = swapRow(row);
  for (const field of Object.keys(row).filter(key => key.endsWith('_y'))) {
    assert.equal(result[field], -3);
    assert.equal(result[field.slice(0, -1) + 'z'], 0);
  }
  assert.deepEqual(result.Graphics_car_coordinates[59], { x: 59, y: 61, z: -59 });
  assert.equal(result.Physics_heading, 1);
  assert.equal(result.Physics_pitch, 2);
  assert.equal(result.Physics_roll, 3);
  assert.equal(result.Physics_speed_kmh, 100);
  assert.deepEqual(result.Graphics_normalized_positions, { 3: 0.25 });
  assert.deepEqual(swapRow(result), row);
  assert.deepEqual(row, original);
});

test('moves partial components without creating missing values', () => {
  assert.deepEqual(swapRow({ Physics_velocity_y: 0, Physics_g_force_z: null }),
    { Physics_velocity_z: 0, Physics_g_force_y: null });
});

test('rejects malformed chunks and coordinates before publishing', () => {
  for (const value of [{ data: [] }, [null], [5], [{ Graphics_car_coordinates: [{ x: 1, y: 2 }] }]]) {
    assert.throws(() => transform(Buffer.from(JSON.stringify(value))));
  }
});

test('publishes ordered verified chunks atomically, preserves originals, videos, and session metadata', async () => {
  const f = fixture();
  const original = JSON.parse(JSON.stringify(f.session));
  const rows = await migrateSession(f.sessions, f.bucket, f.session, f.files, before, quiet);
  assert.equal(rows, 2);
  assert.equal(f.updates(), 1);
  assert.deepEqual(f.session.telemetryMigrations[MIGRATION].originalChunkFileIds, f.files.map(file => file._id));
  assert.deepEqual(JSON.parse(JSON.stringify(f.session.videos)), original.videos);
  assert.deepEqual(f.session.points, original.points);
  assert.equal(f.session.totalDataPoints, 2);
  assert.equal(f.session.totalChunks, 2);
  for (let i = 0; i < f.files.length; i++) {
    assert.deepEqual(JSON.parse(f.buffers.get(f.files[i]._id.toString())), f.source[i]);
    assert.deepEqual(JSON.parse(f.buffers.get(f.session.dataChunkFileIds[i].toString())), f.source[i].map(swapRow));
  }
  assert.equal(await verifySession(f.bucket, f.session), 2);
});

test('a failed later chunk leaves the session on its complete original list and can be retried', async () => {
  const f = fixture();
  const ids = [...f.session.dataChunkFileIds];
  const original = f.buffers.get(ids[1].toString());
  f.buffers.set(ids[1].toString(), Buffer.from('[null]'));
  await assert.rejects(migrateSession(f.sessions, f.bucket, f.session, f.files, before, quiet));
  assert.equal(f.updates(), 0);
  assert.deepEqual(f.session.dataChunkFileIds, ids);
  f.buffers.set(ids[1].toString(), original);
  await migrateSession(f.sessions, f.bucket, f.session, f.files, before, quiet);
  assert.equal(await verifySession(f.bucket, f.session), 2);
});

test('corrupt replacement data and row-count mismatches never publish', async () => {
  const f = fixture();
  const read = f.bucket.openDownloadStream;
  f.bucket.openDownloadStream = id => f.files.some(file => file._id.equals(id))
    ? read(id) : Readable.from([Buffer.from('[]')]);
  await assert.rejects(migrateSession(f.sessions, f.bucket, f.session, f.files, before, quiet), /Verification failed/);
  assert.equal(f.updates(), 0);
  f.bucket.openDownloadStream = read;
  f.session.totalDataPoints = 3;
  await assert.rejects(migrateSession(f.sessions, f.bucket, f.session, f.files, before, quiet), /expected 3 rows/);
  assert.equal(f.updates(), 0);
});

test('a concurrent publish cannot be overwritten or swapped twice', async () => {
  const f = fixture();
  const stale = { ...f.session };
  await migrateSession(f.sessions, f.bucket, f.session, f.files, before, quiet);
  const published = [...f.session.dataChunkFileIds];
  await assert.rejects(migrateSession(f.sessions, f.bucket, stale, f.files, before, quiet), /changed concurrently/);
  assert.deepEqual(f.session.dataChunkFileIds, published);
  assert.equal(await verifySession(f.bucket, f.session), 2);
});

test('verification detects subsequent corruption', async () => {
  const f = fixture();
  await migrateSession(f.sessions, f.bucket, f.session, f.files, before, quiet);
  f.buffers.set(f.session.dataChunkFileIds[0].toString(), Buffer.from('[]'));
  await assert.rejects(verifySession(f.bucket, f.session), /differs from swapped original/);
});

test('dry run scopes ACC, honors cutoff and marker, and handles legacy missing dates without writes', async () => {
  const f = fixture();
  delete f.session.created_date;
  f.session._id = ObjectId.createFromTime(Date.parse('2025-12-09T00:00:00Z') / 1000);
  const marked = { ...f.session, _id: new ObjectId(), created_date: new Date('2026-10-24T00:00:00Z'),
    telemetryMigrations: { [MIGRATION]: { originalChunkFileIds: f.session.dataChunkFileIds } } };
  const recent = { ...f.session, _id: new ObjectId(), created_date: before };
  const db = new MongoClient('mongodb://127.0.0.1:27017').db('unused');
  db.collection = name => {
    if (name === 'racingsessions') return { find(filter) {
      assert.deepEqual(filter, { game_recorded_from: 'acc' });
      return { sort: () => ({ toArray: async () => [f.session, marked, recent] }) };
    } };
    return { find: () => ({ toArray: async () => f.files }) };
  };
  const result = await migrate(db, { before, log: quiet });
  assert.equal(result.candidates, 1);
  assert.equal(result.alreadyMigrated, 1);
  assert.equal(result.afterCutoff, 1);
  assert.equal(result.chunks, 2);
  assert.equal(result.migrated, 0);
  await assert.rejects(migrate(db, { before, verify: true, log: quiet }), /have not been migrated/);
  f.files[0].uploadDate = before;
  await assert.rejects(migrate(db, { before, apply: true, log: quiet }), /recent/);
  f.session.telemetryMigrations = marked.telemetryMigrations;
  const rerun = await migrate(db, { before, apply: true, log: quiet });
  assert.equal(rerun.candidates, 0);
  assert.equal(rerun.alreadyMigrated, 2);
  assert.equal(rerun.migrated, 0);
});
