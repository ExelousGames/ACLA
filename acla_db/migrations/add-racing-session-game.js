const targetDb = db.getSiblingDB(process.env.MONGO_DATEBASE || 'ACLA');
const racingSessions = targetDb.getCollection('racingsessions');

const backfillFilter = {
    $or: [
        { game_recorded_from: { $exists: false } },
        { game_recorded_from: null },
        { game_recorded_from: '' },
    ],
};
const totalBefore = racingSessions.countDocuments({});
const updateResult = racingSessions.updateMany(
    backfillFilter,
    { $set: { game_recorded_from: 'acc' } },
);

print(`Backfill matched: ${updateResult.matchedCount}`);
print(`Backfill modified: ${updateResult.modifiedCount}`);

const newIndexKey = {
    game_recorded_from: 1,
    session_name: 1,
    map: 1,
    car_name: 1,
    user_id: 1,
};
const newIndexName = racingSessions.createIndex(
    newIndexKey,
    { unique: true },
);
print(`Ensured unique index: ${newIndexName}`);

const oldIndexKey = {
    session_name: 1,
    map: 1,
    car_name: 1,
    user_id: 1,
};
const oldIndex = racingSessions.getIndexes().find(
    (index) => JSON.stringify(index.key) === JSON.stringify(oldIndexKey),
);

if (oldIndex) {
    racingSessions.dropIndex(oldIndex.name);
    print(`Dropped old index: ${oldIndex.name}`);
} else {
    print('Old unique index not present; nothing to drop.');
}

const totalAfter = racingSessions.countDocuments({});
print(`Documents before: ${totalBefore}`);
print(`Documents after: ${totalAfter}`);
