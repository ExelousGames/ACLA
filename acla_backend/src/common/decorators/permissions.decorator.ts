import { SetMetadata } from '@nestjs/common';
import { PermissionAction, PermissionResource } from '../../schemas/permission.schema';

export const PERMISSIONS_KEY = 'permissions';

export interface RequiredPermission {
    action: PermissionAction;
    resource: PermissionResource;
}

// Custom decorator to specify required permissions for a route
export const RequirePermissions = (...permissions: RequiredPermission[]) =>
    SetMetadata(PERMISSIONS_KEY, permissions);
