import { Injectable } from '@nestjs/common';
import { UserInfo } from '../../schemas/user-info.schema';
import { Permission, PermissionAction, PermissionResource } from '../../schemas/permission.schema';
import { Role } from '../../schemas/role.schema';
import { RequiredPermission } from '../../common/decorators/permissions.decorator';

// Service to handle authorization logic
@Injectable()
export class AuthorizationService {

    /**
     * Check if user has all required permissions
     * @param user - UserInfo object with populated permissions and roles
     * @param requiredPermissions - Array of required permissions to check
     */
    hasPermissions(user: UserInfo, requiredPermissions: RequiredPermission[]): boolean {
        if (!requiredPermissions || requiredPermissions.length === 0) {
            return true;
        }

        // Get all user permissions (direct + from roles)
        const userPermissions = this.getUserAllPermissions(user);

        // Check if user has all required permissions
        return requiredPermissions.every(required =>
            this.hasPermission(userPermissions, required.action, required.resource)
        );
    }

    /**
     * Check if user has a specific permission
     * @param permissions - Array of user's permissions
     * @param action - Required action
     * @param resource - Required resource
     * @return true if user has the permission, false otherwise
     */
    hasPermission(permissions: Permission[], action: PermissionAction, resource: PermissionResource): boolean {
        return permissions.some(permission => {
            // Check for exact match
            if (permission.action === action && permission.resource === resource) {
                return true;
            }

            // Check for manage permission on the resource
            if (permission.action === PermissionAction.MANAGE && permission.resource === resource) {
                return true;
            }

            // Check for manage permission on all resources
            if (permission.action === PermissionAction.MANAGE && permission.resource === PermissionResource.ALL) {
                return true;
            }

            return false;
        });
    }

    /**
     * Get all permissions for a user (direct permissions + permissions from roles)
     * @param user - UserInfo object with populated permissions and roles
     * @return Array of all permissions
     */
    private getUserAllPermissions(user: UserInfo): Permission[] {
        const allPermissions: Permission[] = [];

        // Add direct permissions
        if (user.permissions) {
            user.permissions.forEach(permission => {
                // Permissions should be populated when fetching user
                allPermissions.push(permission as any);
            });
        }

        // Add permissions from roles
        if (user.roles) {
            user.roles.forEach(role => {
                // Roles should be populated when fetching user
                const roleObj = role as any;
                if (roleObj.permissions) {
                    roleObj.permissions.forEach(permission => {
                        allPermissions.push(permission as any);
                    });
                }
            });
        }

        return allPermissions;
    }

    /**
     * Check if user has any of the specified roles
     */
    hasAnyRole(user: UserInfo, roleNames: string[]): boolean {
        if (!user.roles || !roleNames || roleNames.length === 0) {
            return false;
        }

        return user.roles.some(role => {
            const roleObj = role as any;
            return roleNames.includes(roleObj.name);
        });
    }

    /**
     * Check if user has all specified roles
     */
    hasAllRoles(user: UserInfo, roleNames: string[]): boolean {
        if (!user.roles || !roleNames || roleNames.length === 0) {
            return false;
        }

        return roleNames.every(roleName =>
            user.roles.some(role => {
                const roleObj = role as any;
                return roleObj.name === roleName;
            })
        );
    }
}
