import { applyDecorators, UseGuards } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { RequirePermissions, RequiredPermission } from './permissions.decorator';
import { PermissionsGuard } from '../guards/permissions.guard';

/**
 * Combined decorator that applies JWT authentication and permission checking
 * @param permissions - Array of required permissions
 */
export const Auth = (...permissions: RequiredPermission[]) => {
    const decorators = [
        UseGuards(AuthGuard('jwt'))
    ];

    // Only add permissions guard and decorator if permissions are specified
    if (permissions && permissions.length > 0) {
        decorators.push(
            UseGuards(PermissionsGuard),
            RequirePermissions(...permissions)
        );
    }

    return applyDecorators(...decorators);
};

/**
 * JWT authentication only (no permission checking)
 */
export const JwtAuth = () => applyDecorators(
    UseGuards(AuthGuard('jwt'))
);

/**
 * Local authentication (for login endpoints)
 */
export const LocalAuth = () => applyDecorators(
    UseGuards(AuthGuard('local'))
);
