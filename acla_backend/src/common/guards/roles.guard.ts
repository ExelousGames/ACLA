import { Injectable, CanActivate, ExecutionContext } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ROLES_KEY, RequiredRole } from '../decorators/permissions.decorator';
import { UserInfoService } from '../../modules/user-info/user-info.service';
import { AuthorizationService } from '../../shared/authorization/authorization.service';

@Injectable()
export class RolesGuard implements CanActivate {
    constructor(
        private reflector: Reflector,
        private userInfoService: UserInfoService,
        private authorizationService: AuthorizationService
    ) { }

    async canActivate(context: ExecutionContext): Promise<boolean> {
        // Get required roles from metadata set by the @RequireRoles decorator
        const requiredRoles = this.reflector.getAllAndOverride<RequiredRole>(
            ROLES_KEY,
            [context.getHandler(), context.getClass()]
        );

        if (!requiredRoles) {
            return true;
        }

        // Get user from request
        const { user } = context.switchToHttp().getRequest();

        // If user is not found in request, deny access
        if (!user || !user.user) {
            return false;
        }

        // User object already contains populated permissions and roles from JWT strategy
        const userWithPermissions = user.user;

        if (!userWithPermissions) {
            return false;
        }

        // Check if user has required roles
        const { roles, requireAll = false } = requiredRoles;
        if (roles && roles.length > 0) {
            return requireAll
                ? this.authorizationService.hasAllRoles(userWithPermissions, roles)
                : this.authorizationService.hasAnyRole(userWithPermissions, roles);
        }

        return true;
    }
}
