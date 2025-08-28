//Only when the database is first created (empty data directory)

db.createCollection("userinfos");
db.createCollection("racingmaps");
db.createCollection("racingsessions");
db.createCollection("permissions");
db.createCollection("roles");

// Insert initial permissions
const permissions = [
    // User permissions
    {
        id: "perm_user_create",
        name: "Create User",
        description: "Permission to create new users",
        action: "create",
        resource: "user",
        createdAt: new Date()
    },
    {
        id: "perm_user_read",
        name: "Read User",
        description: "Permission to view user information",
        action: "read",
        resource: "user",
        createdAt: new Date()
    },
    {
        id: "perm_user_update",
        name: "Update User",
        description: "Permission to update user information",
        action: "update",
        resource: "user",
        createdAt: new Date()
    },
    {
        id: "perm_user_delete",
        name: "Delete User",
        description: "Permission to delete users",
        action: "delete",
        resource: "user",
        createdAt: new Date()
    },
    {
        id: "perm_user_manage",
        name: "Manage User",
        description: "Full permission to manage users",
        action: "manage",
        resource: "user",
        createdAt: new Date()
    },

    // Racing Session permissions
    {
        id: "perm_racing_session_create",
        name: "Create Racing Session",
        description: "Permission to create new racing sessions",
        action: "create",
        resource: "racing_session",
        createdAt: new Date()
    },
    {
        id: "perm_racing_session_read",
        name: "Read Racing Session",
        description: "Permission to view racing sessions",
        action: "read",
        resource: "racing_session",
        createdAt: new Date()
    },
    {
        id: "perm_racing_session_update",
        name: "Update Racing Session",
        description: "Permission to update racing sessions",
        action: "update",
        resource: "racing_session",
        createdAt: new Date()
    },
    {
        id: "perm_racing_session_delete",
        name: "Delete Racing Session",
        description: "Permission to delete racing sessions",
        action: "delete",
        resource: "racing_session",
        createdAt: new Date()
    },
    {
        id: "perm_racing_session_manage",
        name: "Manage Racing Session",
        description: "Full permission to manage racing sessions",
        action: "manage",
        resource: "racing_session",
        createdAt: new Date()
    },

    // Racing Map permissions
    {
        id: "perm_racing_map_create",
        name: "Create Racing Map",
        description: "Permission to create new racing maps",
        action: "create",
        resource: "racing_map",
        createdAt: new Date()
    },
    {
        id: "perm_racing_map_read",
        name: "Read Racing Map",
        description: "Permission to view racing maps",
        action: "read",
        resource: "racing_map",
        createdAt: new Date()
    },
    {
        id: "perm_racing_map_update",
        name: "Update Racing Map",
        description: "Permission to update racing maps",
        action: "update",
        resource: "racing_map",
        createdAt: new Date()
    },
    {
        id: "perm_racing_map_delete",
        name: "Delete Racing Map",
        description: "Permission to delete racing maps",
        action: "delete",
        resource: "racing_map",
        createdAt: new Date()
    },
    {
        id: "perm_racing_map_manage",
        name: "Manage Racing Map",
        description: "Full permission to manage racing maps",
        action: "manage",
        resource: "racing_map",
        createdAt: new Date()
    },

    {
        id: "perm_all_manage",
        name: "Manage All",
        description: "Full permission to manage all system resources",
        action: "manage",
        resource: "all",
        createdAt: new Date()
    },

    //Menu Permissions
    {
        id: 'perm_menu_view',
        name: 'View Main Dashboard',
        description: 'Permission to visit the main dashboard',
        action: 'read',
        resource: 'menu',
        createdAt: new Date()
    },

    // All resources permissions (for super admin)
    {
        id: "perm_all_manage",
        name: "Manage All",
        description: "Full permission to manage all system resources",
        action: "manage",
        resource: "all",
        createdAt: new Date()
    }
];

db.permissions.insertMany(permissions);

// Get permission ObjectIds for role assignment
const userReadPerm = db.permissions.findOne({ id: "perm_user_read" })._id;
const userUpdatePerm = db.permissions.findOne({ id: "perm_user_update" })._id;
const userManagePerm = db.permissions.findOne({ id: "perm_user_manage" })._id;
const racingSessionReadPerm = db.permissions.findOne({ id: "perm_racing_session_read" })._id;
const racingSessionCreatePerm = db.permissions.findOne({ id: "perm_racing_session_create" })._id;
const racingSessionUpdatePerm = db.permissions.findOne({ id: "perm_racing_session_update" })._id;
const racingSessionManagePerm = db.permissions.findOne({ id: "perm_racing_session_manage" })._id;
const racingMapReadPerm = db.permissions.findOne({ id: "perm_racing_map_read" })._id;
const racingMapCreatePerm = db.permissions.findOne({ id: "perm_racing_map_create" })._id;
const racingMapUpdatePerm = db.permissions.findOne({ id: "perm_racing_map_update" })._id;
const racingMapManagePerm = db.permissions.findOne({ id: "perm_racing_map_manage" })._id;
const allManagePerm = db.permissions.findOne({ id: "perm_all_manage" })._id;

// Insert initial roles
const roles = [
    {
        id: "role_super_admin",
        name: "Super Admin",
        description: "Full system access with all permissions",
        permissions: [allManagePerm],
        createdAt: new Date(),
        isActive: true
    },
    {
        id: "role_admin",
        name: "Admin",
        description: "Administrative access to manage users and system resources",
        permissions: [
            userManagePerm,
            racingSessionManagePerm,
            racingMapManagePerm
        ],
        createdAt: new Date(),
        isActive: true
    },
    {
        id: "role_moderator",
        name: "Moderator",
        description: "Moderate racing sessions and maps",
        permissions: [
            userReadPerm,
            userUpdatePerm,
            racingMapCreatePerm,
            racingSessionManagePerm,
            racingMapUpdatePerm,
            racingMapReadPerm
        ],
        createdAt: new Date(),
        isActive: true
    },
    {
        id: "role_user",
        name: "User",
        description: "Basic user access to participate in racing",
        permissions: [
            racingSessionReadPerm,
            racingMapReadPerm
        ],
        createdAt: new Date(),
        isActive: true
    },
    {
        id: "role_guest",
        name: "Guest",
        description: "Read-only access to public content",
        permissions: [
            racingSessionReadPerm,
            racingMapReadPerm
        ],
        createdAt: new Date(),
        isActive: true
    }
];

db.roles.insertMany(roles);

// Create a test user with hashed password
// Note: In a real application, you would use the proper password hashing from your backend
// For testing purposes, we'll create a user with a plaintext password that will be updated through the API
const superAdminRole = db.roles.findOne({ id: "role_super_admin" })._id;

// Create test admin user
db.userinfos.insertOne({
    id: "test_admin_user",
    email: "admin@test.com",
    password: "$2b$10$placeholder", // This will be replaced when user changes password through API
    roles: [superAdminRole],
    permissions: [],
    isActive: true,
    createdAt: new Date(),
    lastLogin: new Date()
});

db.createUser({
    user: 'client',
    pwd: 'clientpass',
    roles: [
        {
            role: 'readWrite',
            db: 'ACLA',
        },
    ],
});