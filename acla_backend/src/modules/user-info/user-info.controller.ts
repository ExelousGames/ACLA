import { Body, Controller, Delete, Get, Request, Param, Post, Put } from '@nestjs/common';
import { AuthService } from 'src/shared/auth/auth.service';
import { UserInfo } from '../../schemas/user-info.schema';
import { UserInfoService } from './user-info.service';
import { CreateUserInfoDto, UpdateUserPermissionsDto, UpdateUserRolesDto, UpdateUserPasswordDto } from 'src/dto/user.dto';
import {
    Auth,
    LocalAuth,
    JwtAuth,
    AdminOnly,
    AdminOrModerator,
    RequireAllRoles
} from '../../common/decorators/auth.decorator';
import { PermissionAction, PermissionResource } from '../../schemas/permission.schema';

@Controller('userinfo')
export class UserInfoController {

    constructor(private userinfoService: UserInfoService, private authService: AuthService) { }

    //We are using LocalStrategy located in auth folder. our passport local strategy has a default name of 'local',
    //we refers the name in 'AuthGuard'
    //Passport automatically creates a user object (populated by passport during the passport-local authentication flow), 
    // based on the value we return from the validate() method, 
    // and assigns it to the Request object as req.user
    @LocalAuth()
    @Post('auth/login')
    async login(@Request() req) {
        // return a JWT token for a later access
        return this.authService.giveJWTToken(req.user);
    }

    @LocalAuth()
    @Post('auth/logout')
    async logout(@Request() req) {
        return req.logout();
    }

    // Public registration endpoint (for testing password hashing)
    @Post('register')
    async registerUser(@Body() createUserInfoDto: CreateUserInfoDto): Promise<{ message: string }> {
        try {
            await this.userinfoService.createUser(createUserInfoDto);
            return { message: 'User registered successfully' };
        } catch (error) {
            throw error;
        }
    }

    @Auth({ permissions: [{ action: PermissionAction.CREATE, resource: PermissionResource.USER }] })
    @Post()
    async createUser(@Body() createUserInfoDto: CreateUserInfoDto): Promise<CreateUserInfoDto> {
        try {
            const result = await this.userinfoService.createUser(createUserInfoDto);
            return result;
        } catch (error) {
            throw error;
        }
    }

    @Auth({ permissions: [{ action: PermissionAction.DELETE, resource: PermissionResource.USER }] })
    @Delete(':id')
    deleteUser(@Param('id') id: string): void {
        this.userinfoService.deleteTask(id);
    }

    @Auth({ permissions: [{ action: PermissionAction.UPDATE, resource: PermissionResource.USER }] })
    @Put(':id/permissions')
    async updateUserPermissions(
        @Param('id') userId: string,
        @Body() updatePermissionsDto: UpdateUserPermissionsDto
    ): Promise<UserInfo | null> {
        // Set the userId from the URL parameter
        updatePermissionsDto.userId = userId;

        try {
            const updatedUser = await this.userinfoService.updateUserPermissions(updatePermissionsDto);
            return updatedUser;
        } catch (error) {
            throw error;
        }
    }

    @Auth({ permissions: [{ action: PermissionAction.UPDATE, resource: PermissionResource.USER }] })
    @Put(':id/roles')
    async updateUserRoles(
        @Param('id') userId: string,
        @Body() updateRolesDto: UpdateUserRolesDto
    ): Promise<UserInfo | null> {
        // Set the userId from the URL parameter
        updateRolesDto.userId = userId;

        try {
            const updatedUser = await this.userinfoService.updateUserRoles(updateRolesDto);
            return updatedUser;
        } catch (error) {
            throw error;
        }
    }

    @Auth({ permissions: [{ action: PermissionAction.UPDATE, resource: PermissionResource.USER }] })
    @Put(':id/password')
    async updateUserPassword(
        @Param('id') userId: string,
        @Body() updatePasswordDto: UpdateUserPasswordDto
    ): Promise<{ message: string }> {
        // Set the userId from the URL parameter
        updatePasswordDto.userId = userId;

        try {
            await this.userinfoService.updateUserPassword(updatePasswordDto);
            return { message: 'Password updated successfully' };
        } catch (error) {
            throw error;
        }
    }

    @JwtAuth()
    @Get('profile')
    getProfile(@Request() req) {
        return req.user;
    }

    // Example: Only users with 'admin' role can access this endpoint
    @AdminOnly()
    @Get('admin/users')
    async getAllUsers(): Promise<UserInfo[]> {
        // Implementation would go here
        return [];
    }

    // Example: Users need either 'admin' OR 'moderator' role
    @AdminOrModerator()
    @Get('moderation/dashboard')
    async getModerationDashboard() {
        // Implementation would go here
        return { message: 'Moderation dashboard' };
    }

    // Example: Users need BOTH 'admin' AND 'super-admin' roles
    @RequireAllRoles('admin', 'super-admin')
    @Get('super-admin/settings')
    async getSuperAdminSettings() {
        // Implementation would go here
        return { message: 'Super admin settings' };
    }

    // Example: Combine both permissions and roles
    @Auth({
        permissions: [{ action: PermissionAction.READ, resource: PermissionResource.USER }],
        roles: { roles: ['admin'] }
    })
    @Get('admin/user-management')
    async getUserManagement() {
        // User needs both the READ USER permission AND admin role
        return { message: 'User management interface' };
    }
}
