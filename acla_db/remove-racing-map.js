// Run against an existing Docker Mongo volume after removing the map editor.
// Example:
// docker exec -i mongodb_c mongosh -u "$MONGO_ADMINUSERNAME" -p "$MONGO_ADMINPASSWORD" --authenticationDatabase admin ACLA < acla_db/remove-racing-map.js

const mapPermissions = db.permissions.find({ resource: "racing_map" }, { _id: 1 }).toArray();
const mapPermissionIds = mapPermissions.map((permission) => permission._id);

if (mapPermissionIds.length > 0) {
    const rolesResult = db.roles.updateMany(
        { permissions: { $in: mapPermissionIds } },
        { $pull: { permissions: { $in: mapPermissionIds } } }
    );

    const usersResult = db.userinfos.updateMany(
        { permissions: { $in: mapPermissionIds } },
        { $pull: { permissions: { $in: mapPermissionIds } } }
    );

    const permissionsResult = db.permissions.deleteMany({ _id: { $in: mapPermissionIds } });

    print(`Removed racing_map permissions from ${rolesResult.modifiedCount} roles.`);
    print(`Removed racing_map permissions from ${usersResult.modifiedCount} users.`);
    print(`Deleted ${permissionsResult.deletedCount} racing_map permission documents.`);
} else {
    print("No racing_map permissions found.");
}

if (db.getCollectionNames().includes("racingmaps")) {
    db.racingmaps.drop();
    print("Dropped racingmaps collection.");
} else {
    print("No racingmaps collection found.");
}
