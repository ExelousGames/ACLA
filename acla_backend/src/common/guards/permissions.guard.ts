import { Injectable, CanActivate, ExecutionContext } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { PERMISSIONS_KEY, RequiredPermission } from '../decorators/permissions.decorator';
import { UserInfoService } from '../../modules/user-info/user-info.service';
import { AuthorizationService } from '../../shared/authorization/authorization.service';

@Injectable()
export class PermissionsGuard implements CanActivate {
    constructor(
        private reflector: Reflector,
        private userInfoService: UserInfoService,
        private authorizationService: AuthorizationService
    ) { }

    // The canActivate method is called by the NestJS framework to determine if a request should be allowed to proceed
    // Execution context provides details about the current request being processed, it extends ArgumentHost.
    //The ArgumentsHost class provides methods for retrieving the arguments being passed to a handler. It allows choosing the appropriate context (e.g., HTTP, RPC (microservice), or WebSockets) to retrieve the arguments from.
    async canActivate(context: ExecutionContext): Promise<boolean> {
        // Get required permissions from metadata set by the @RequirePermissions decorator
        const requiredPermissions = this.reflector.getAllAndOverride<RequiredPermission[]>(
            PERMISSIONS_KEY,
            [context.getHandler(), context.getClass()]
        );

        if (!requiredPermissions) {
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

        // Check if user has required permissions
        return this.authorizationService.hasPermissions(userWithPermissions, requiredPermissions);
    }
}
