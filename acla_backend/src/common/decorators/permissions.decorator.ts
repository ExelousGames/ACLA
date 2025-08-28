import { SetMetadata } from '@nestjs/common';
import { PermissionAction, PermissionResource } from '../../schemas/permission.schema';

export const PERMISSIONS_KEY = 'permissions';
export const ROLES_KEY = 'roles';

export interface RequiredPermission {
    action: PermissionAction;
    resource: PermissionResource;
}

export interface RequiredRole {
    roles: string[];
    requireAll?: boolean; // If true, user must have ALL roles. If false, user needs ANY role. Default is false.
}

// Custom decorator to specify required permissions for a route
export const RequirePermissions = (...permissions: RequiredPermission[]) =>
    SetMetadata(PERMISSIONS_KEY, permissions);

// Custom decorator to specify required roles for a route
export const RequireRoles = (roleConfig: RequiredRole) =>
    SetMetadata(ROLES_KEY, roleConfig);
