import { applyDecorators, UseGuards } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { RequirePermissions, RequiredPermission, RequireRoles, RequiredRole } from './permissions.decorator';
import { PermissionsGuard } from '../guards/permissions.guard';
import { RolesGuard } from '../guards/roles.guard';

export interface AuthOptions {
    permissions?: RequiredPermission[];
    roles?: RequiredRole;
}

/**
 * Combined decorator that applies JWT authentication with optional permission and role checking
 * @param options - Object containing permissions and/or roles requirements
 */
export const Auth = (options: AuthOptions = {}) => {

    // default decorators
    const decorators = [
        UseGuards(AuthGuard('jwt'))
    ];

    // Add permissions guard and decorator if permissions are specified
    if (options.permissions && options.permissions.length > 0) {
        decorators.push(

            // apply permission checking mechanic
            UseGuards(PermissionsGuard),

            // add permission requirements, what kind of permission are required for this operation
            RequirePermissions(...options.permissions)
        );
    }

    // Add roles guard and decorator if roles are specified
    if (options.roles && options.roles.roles && options.roles.roles.length > 0) {
        decorators.push(

            //apply role checking mechanic
            UseGuards(RolesGuard),

            // add role requirements, what kind of roles are required for this operation
            RequireRoles(options.roles)
        );
    }

    return applyDecorators(...decorators);
};

/**
 * Backward compatibility: Auth decorator that accepts permissions directly
 * @param permissions - Array of required permissions
 * @deprecated Use Auth({ permissions: [...] }) instead
 */
export const AuthWithPermissions = (...permissions: RequiredPermission[]) => {
    return Auth({ permissions });
};

/**
 * Auth decorator for role-based access control
 * @param roles - Array of role names
 * @param requireAll - If true, user must have ALL roles. If false, user needs ANY role. Default is false.
 */
export const AuthWithRoles = (roles: string[], requireAll: boolean = false) => {
    return Auth({ roles: { roles, requireAll } });
};

/**
 * Shorthand for requiring admin role only
 */
export const AdminOnly = () => AuthWithRoles(['admin']);

/**
 * Shorthand for requiring admin or moderator role
 */
export const AdminOrModerator = () => AuthWithRoles(['admin', 'moderator']);

/**
 * Shorthand for requiring multiple roles (ALL required)
 */
export const RequireAllRoles = (...roles: string[]) => AuthWithRoles(roles, true);

/**
 * Shorthand for requiring any of the specified roles
 */
export const RequireAnyRole = (...roles: string[]) => AuthWithRoles(roles, false);

/**
 * JWT authentication only (no permission or role checking), more info check jwt.strategy.ts
 */
export const JwtAuth = () => applyDecorators(
    UseGuards(AuthGuard('jwt'))
);

/**
 * Local authentication (for login endpoints), more info check local.strategy.ts
 */
export const LocalAuth = () => applyDecorators(
    UseGuards(AuthGuard('local'))
);
