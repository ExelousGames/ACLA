export class LoginInfoDto {
    email: string;
    password: string;

}

export class CreateUserInfoDto {
    email: string;
    password: string;
}

export class UserProfileDto {
    id: string;
    email: string;
    roles: any[];
    permissions: any[];
    isActive: boolean;
    createdAt: Date;
    lastLogin: Date;
    userId?: string; // From JWT payload
    username?: string; // From JWT payload (email)
}

export class UpdateUserPermissionsDto {
    userId: string;
    permissions: string[]; // Array of permission IDs
}

export class UpdateUserRolesDto {
    userId: string;
    roles: string[]; // Array of role IDs
}

export class UpdateUserPasswordDto {
    userId: string;
    newPassword: string;
}