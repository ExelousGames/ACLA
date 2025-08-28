import { Module } from '@nestjs/common';
import { AuthorizationService } from './authorization.service';
import { PermissionsGuard } from '../../common/guards/permissions.guard';
import { RolesGuard } from '../../common/guards/roles.guard';

@Module({
    imports: [],
    providers: [AuthorizationService, PermissionsGuard, RolesGuard],
    exports: [AuthorizationService, PermissionsGuard, RolesGuard],
})
export class AuthorizationModule { }
