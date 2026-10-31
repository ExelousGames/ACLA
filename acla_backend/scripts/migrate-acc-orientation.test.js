'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { Readable, Writable } = require('node:stream');
const { MongoClient, ObjectId } = require('mongodb');
const { swapRow, migrateSession, verifySession, migrate } = require('./migrate-acc-yz');
const { MIGRATION, ROTATION_FIELDS, convertRow, transform, orientationMigration } = require('./migrate-acc-orientation');

const before = new Date('2026-10-31T01:37:46Z');
const quiet = () => {};

test('repairs signs without swapping normalized positions or polar vectors again', () => {
  const row = { Physics_heading: 1.2, Physics_pitch: -0.4, Physics_roll: 0.7,
    Physics_local_angular_vel_x: -4, Physics_local_angular_vel_y: -9, Physics_local_angular_vel_z: 2,
    Physics_local_velocity_x: -4, Physics_local_velocity_y: -9, Physics_local_velocity_z: 2,
    Physics_velocity_x: 1, Physics_velocity_y: 3, Physics_velocity_z: 2,
    Physics_g_force_y: 5, Physics_g_force_z: 6, Physics_speed_kmh: 100,
    Graphics_car_coordinates: [{ x: 7, y: 9, z: 8 }], Physics_wheel_angular_s_front_left: 12 };
  const original = structuredClone(row);
  const result = convertRow(row);
  for (const [field, value] of Object.entries(row)) {
    assert.deepEqual(result[field], ROTATION_FIELDS.includes(field) ? -value : value);
  }
  assert.deepEqual(row, original);
});

test('combined migrations preserve right-hand rotations and axial cross products', () => {
  const rx = ([x, y, z], a) => [x, y * Math.cos(a) - z * Math.sin(a), y * Math.sin(a) + z * Math.cos(a)];
  const ry = ([x, y, z], a) => [x * Math.cos(a) + z * Math.sin(a), y, -x * Math.sin(a) + z * Math.cos(a)];
  const rz = ([x, y, z], a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a), z];
  const p = ([x, y, z]) => [x, z, y];
  const cross = ([a, b, c], [x, y, z]) => [b * z - c * y, c * x - a * z, a * y - b * x];
  const close = (actual, expected) => actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-10));
  for (const [h, pitch, roll] of [[0, 0, 0], [Math.PI / 2, 0, 0], [0, Math.PI / 2, 0],
    [0, 0, -Math.PI / 2], [1.2, -0.4, 0.7], [5 * Math.PI / 2, 0.3, -0.2]]) {
    const row = convertRow(swapRow({ Physics_heading: h, Physics_pitch: pitch, Physics_roll: roll,
      Physics_local_angular_vel_x: -4, Physics_local_angular_vel_y: 2, Physics_local_angular_vel_z: -9 }));
    for (const v of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-2, 3, 40]]) {
      close(rz(rx(ry(p(v), row.Physics_roll), row.Physics_pitch), row.Physics_heading), p(ry(rx(rz(v, roll), pitch), h)));
      close(cross(['x', 'y', 'z'].map(axis => row[`Physics_local_angular_vel_${axis}`]), p(v)), p(cross([-4, 2, -9], v)));
    }
  }
});

test('preserves absent and null fields, normalizes zero, and rejects malformed rotations', () => {
  assert.deepEqual(convertRow({ Physics_heading: -0, Physics_roll: null, Physics_local_angular_vel_y: 0 }),
    { Physics_heading: 0, Physics_roll: null, Physics_local_angular_vel_y: 0 });
  for (const field of ROTATION_FIELDS) {
    for (const value of [NaN, Infinity, '1', false, [], {}]) assert.throws(() => convertRow({ [field]: value }), /Invalid/);
  }
  for (const value of [{}, [null], [3], [[]]]) assert.throws(() => transform(Buffer.from(JSON.stringify(value))));
});

function fixture() {
  const source = [[{ Physics_heading: 0.5, Physics_local_angular_vel_z: -2, Physics_velocity_y: 3 }],
    [{ Physics_pitch: -0.1, Graphics_car_coordinates: [{ x: 1, y: 3, z: 2 }] }]];
  const session = { _id: new ObjectId(), game_recorded_from: 'acc', created_date: new Date('2026-10-24T00:00:00Z'),
    totalDataPoints: 2, totalChunks: 2, videos: [{ chunkFileIds: [new ObjectId()] }],
    telemetryMigrations: { acc_yz_v1: { rows: 2, originalChunkFileIds: source.map(() => new ObjectId()) } } };
  const files = source.map((rows, i) => ({ _id: new ObjectId(), filename: `chunk-${i}.json`, chunkSize: 255 * 1024,
    uploadDate: new Date('2026-10-26T19:33:00Z'), length: Buffer.byteLength(JSON.stringify(rows)),
    metadata: { chunkIndex: i, telemetryMigration: { id: 'acc_yz_v1', sessionId: session._id,
      sourceFileId: session.telemetryMigrations.acc_yz_v1.originalChunkFileIds[i] } } }));
  session.dataChunkFileIds = files.map(f => f._id);
  const buffers = new Map(files.map((f, i) => [f._id.toString(), Buffer.from(JSON.stringify(source[i]))]));
  const bucket = {
    openDownloadStream: id => Readable.from([buffers.get(id.toString())]),
    openUploadStream(filename, options) {
      assert.equal(options.metadata.telemetryMigration.id, MIGRATION);
      assert.ok(options.metadata.telemetryMigration.sourceFileId.equals(files[options.metadata.chunkIndex]._id));
      const parts = [];
      const stream = new Writable({ write(chunk, encoding, done) { parts.push(chunk); done(); },
        final(done) { buffers.set(stream.id.toString(), Buffer.concat(parts)); done(); } });
      stream.id = new ObjectId();
      return stream;
    },
  };
  const sessions = { async updateOne(filter, update) {
    assert.deepEqual(filter[`telemetryMigrations.${MIGRATION}`], { $exists: false });
    if (session.telemetryMigrations[MIGRATION]) return { modifiedCount: 0 };
    assert.deepEqual(filter.dataChunkFileIds, session.dataChunkFileIds);
    session.dataChunkFileIds = update.$set.dataChunkFileIds;
    session.telemetryMigrations[MIGRATION] = update.$set[`telemetryMigrations.${MIGRATION}`];
    return { modifiedCount: 1 };
  } };
  const db = new MongoClient('mongodb://127.0.0.1:27017').db('unused');
  const documents = [session];
  db.collection = name => name === 'racingsessions' ? { find(filter) {
    assert.deepEqual(filter, { game_recorded_from: 'acc' });
    return { sort: () => ({ toArray: async () => documents }) };
  } } : { find: () => ({ toArray: async () => files }) };
  return { source, session, files, buffers, bucket, sessions, db, documents };
}

test('preflight requires proven Y/Z provenance and refuses reordered, foreign, or recent chunks', async () => {
  const f = fixture();
  const options = { before, migration: orientationMigration, log: quiet };
  assert.equal((await migrate(f.db, options)).candidates, 1);
  const previous = f.session.telemetryMigrations.acc_yz_v1;
  delete f.session.telemetryMigrations.acc_yz_v1;
  await assert.rejects(migrate(f.db, options), /requires a completed/);
  f.session.telemetryMigrations.acc_yz_v1 = previous;
  const file = f.files[0];
  const original = { ...file, metadata: structuredClone(file.metadata) };
  for (const patch of [{ id: 'unknown' }, { sessionId: new ObjectId() }, { sourceFileId: new ObjectId() }]) {
    const saved = file.metadata.telemetryMigration;
    file.metadata.telemetryMigration = { ...saved, ...patch };
    await assert.rejects(migrate(f.db, options), /mismatched/);
    file.metadata.telemetryMigration = saved;
  }
  file.uploadDate = before;
  await assert.rejects(migrate(f.db, options), /recent/);
  file.uploadDate = original.uploadDate;
  f.session.dataChunkFileIds.reverse();
  await assert.rejects(migrate(f.db, options), /mismatched/);
  f.session.dataChunkFileIds.reverse();
  f.documents.push({ ...f.session, _id: new ObjectId(), created_date: before, telemetryMigrations: {} });
  assert.equal((await migrate(f.db, options)).afterCutoff, 1);
});

test('publishes verified orientation copies, retains both backup generations, and skips reruns', async () => {
  const f = fixture();
  const previous = JSON.stringify(f.session.telemetryMigrations.acc_yz_v1);
  const videos = JSON.stringify(f.session.videos);
  const stale = { ...f.session };
  await migrateSession(f.sessions, f.bucket, f.session, f.files, before, quiet, orientationMigration);
  assert.equal(await verifySession(f.bucket, f.session, orientationMigration), 2);
  assert.equal(JSON.stringify(f.session.telemetryMigrations.acc_yz_v1), previous);
  assert.equal(JSON.stringify(f.session.videos), videos);
  assert.deepEqual(f.session.telemetryMigrations[MIGRATION].originalChunkFileIds, f.files.map(file => file._id));
  f.files.forEach((file, i) => assert.deepEqual(JSON.parse(f.buffers.get(file._id.toString())), f.source[i]));
  const published = [...f.session.dataChunkFileIds];
  await assert.rejects(migrateSession(f.sessions, f.bucket, stale, f.files, before, quiet, orientationMigration), /changed concurrently/);
  assert.deepEqual(f.session.dataChunkFileIds, published);
  const rerun = await migrate(f.db, { before, apply: true, migration: orientationMigration, log: quiet });
  assert.equal(rerun.alreadyMigrated, 1);
  assert.equal(rerun.migrated, 0);
  f.buffers.set(published[0].toString(), Buffer.from('[]'));
  await assert.rejects(verifySession(f.bucket, f.session, orientationMigration), /differs/);
});

test('malformed later rotation leaves session references and markers untouched', async () => {
  const f = fixture();
  const ids = [...f.session.dataChunkFileIds];
  f.buffers.set(ids[1].toString(), Buffer.from('[{"Physics_roll":"1"}]'));
  await assert.rejects(migrateSession(f.sessions, f.bucket, f.session, f.files, before, quiet, orientationMigration), /Invalid/);
  assert.deepEqual(f.session.dataChunkFileIds, ids);
  assert.equal(f.session.telemetryMigrations[MIGRATION], undefined);
});
