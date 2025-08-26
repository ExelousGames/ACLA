// Permission actions
export const PERMISSION_ACTIONS = {
    CREATE: 'create',
    READ: 'read',
    UPDATE: 'update',
    DELETE: 'delete',
    MANAGE: 'manage'
} as const;

// Permission resources
export const PERMISSION_RESOURCES = {
    USER: 'user',
    RACING_SESSION: 'racing_session',
    RACING_MAP: 'racing_map',
    ALL: 'all'
} as const;

// Common role names
export const ROLES = {
    ADMIN: 'admin',
    USER: 'user',
    MODERATOR: 'moderator'
} as const;

// Helper functions for permission checking
export const createPermission = (action: string, resource: string) => ({
    action,
    resource
});

// Common permission combinations
//use helper function because it ensures consistent structure and reduces errors
export const COMMON_PERMISSIONS = {
    CREATE_USER: createPermission(PERMISSION_ACTIONS.CREATE, PERMISSION_RESOURCES.USER),
    READ_USER: createPermission(PERMISSION_ACTIONS.READ, PERMISSION_RESOURCES.USER),
    UPDATE_USER: createPermission(PERMISSION_ACTIONS.UPDATE, PERMISSION_RESOURCES.USER),
    DELETE_USER: createPermission(PERMISSION_ACTIONS.DELETE, PERMISSION_RESOURCES.USER),
    MANAGE_ALL: createPermission(PERMISSION_ACTIONS.MANAGE, PERMISSION_RESOURCES.ALL),
} as const;
