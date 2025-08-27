export class LoginInfoDto {
    email: string;
    password: string;

}

export class CreateUserInfoDto {
    email: string;
    password: string;
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