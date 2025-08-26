// Database initialization script for permissions and roles
// Run this script to set up initial permissions, roles, and admin user

import { PermissionAction, PermissionResource } from '../schemas/permission.schema';

// Sample permissions to create
export const INITIAL_PERMISSIONS = [
    // User permissions
    {
        name: 'Create User',
        description: 'Create new users in the system',
        action: PermissionAction.CREATE,
        resource: PermissionResource.USER
    },
    {
        name: 'Read User',
        description: 'View user information',
        action: PermissionAction.READ,
        resource: PermissionResource.USER
    },
    {
        name: 'Update User',
        description: 'Modify user information',
        action: PermissionAction.UPDATE,
        resource: PermissionResource.USER
    },
    {
        name: 'Delete User',
        description: 'Remove users from the system',
        action: PermissionAction.DELETE,
        resource: PermissionResource.USER
    },
    {
        name: 'Manage Users',
        description: 'Full control over user management',
        action: PermissionAction.MANAGE,
        resource: PermissionResource.USER
    },

    // Racing Session permissions
    {
        name: 'Create Racing Session',
        description: 'Create new racing sessions',
        action: PermissionAction.CREATE,
        resource: PermissionResource.RACING_SESSION
    },
    {
        name: 'Read Racing Session',
        description: 'View racing session data',
        action: PermissionAction.READ,
        resource: PermissionResource.RACING_SESSION
    },
    {
        name: 'Update Racing Session',
        description: 'Modify racing session data',
        action: PermissionAction.UPDATE,
        resource: PermissionResource.RACING_SESSION
    },
    {
        name: 'Delete Racing Session',
        description: 'Remove racing sessions',
        action: PermissionAction.DELETE,
        resource: PermissionResource.RACING_SESSION
    },
    {
        name: 'Manage Racing Sessions',
        description: 'Full control over racing sessions',
        action: PermissionAction.MANAGE,
        resource: PermissionResource.RACING_SESSION
    },

    // Racing Map permissions
    {
        name: 'Create Racing Map',
        description: 'Create new racing maps',
        action: PermissionAction.CREATE,
        resource: PermissionResource.RACING_MAP
    },
    {
        name: 'Read Racing Map',
        description: 'View racing map data',
        action: PermissionAction.READ,
        resource: PermissionResource.RACING_MAP
    },
    {
        name: 'Update Racing Map',
        description: 'Modify racing map data',
        action: PermissionAction.UPDATE,
        resource: PermissionResource.RACING_MAP
    },
    {
        name: 'Delete Racing Map',
        description: 'Remove racing maps',
        action: PermissionAction.DELETE,
        resource: PermissionResource.RACING_MAP
    },
    {
        name: 'Manage Racing Maps',
        description: 'Full control over racing maps',
        action: PermissionAction.MANAGE,
        resource: PermissionResource.RACING_MAP
    },

    // System-wide permissions
    {
        name: 'System Administrator',
        description: 'Full system access',
        action: PermissionAction.MANAGE,
        resource: PermissionResource.ALL
    }
];

// Sample roles to create
export const INITIAL_ROLES = [
    {
        name: 'admin',
        description: 'System Administrator with full access',
        permissions: [
            // Admin gets the system-wide manage permission
            'System Administrator'
        ]
    },
    {
        name: 'moderator',
        description: 'Content moderator with limited admin access',
        permissions: [
            'Read User',
            'Update User',
            'Manage Racing Sessions',
            'Manage Racing Maps'
        ]
    },
    {
        name: 'user',
        description: 'Regular user with basic access',
        permissions: [
            'Read User',
            'Read Racing Session',
            'Read Racing Map',
            'Create Racing Session'
        ]
    }
];

// Sample admin user
export const INITIAL_ADMIN_USER = {
    email: 'admin@acla.com',
    password: 'admin123', // Should be hashed in production
    roles: ['admin'],
    permissions: [], // Admin role provides all permissions
    isActive: true
};
