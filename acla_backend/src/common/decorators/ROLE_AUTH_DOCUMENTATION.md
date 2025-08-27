# Role-Based Authentication Documentation

This document explains how to use the enhanced authentication system that supports both permissions and roles.

## Overview

The authentication system now supports three main authorization methods:

1. **Permission-based authorization** - Fine-grained access control based on specific actions and resources
2. **Role-based authorization** - Access control based on user roles
3. **Combined authorization** - Using both permissions and roles together

## Available Decorators

### Basic Auth Decorators

```typescript
// JWT authentication only (no permission/role checking)
@JwtAuth()

// Local authentication (for login endpoints)
@LocalAuth()

// Combined auth with options
@Auth({
    permissions: [{ action: PermissionAction.READ, resource: PermissionResource.USER }],
    roles: { roles: ['admin'], requireAll: false }
})
```

### Permission-Based Auth

```typescript
// Single permission
@Auth({ permissions: [{ action: PermissionAction.READ, resource: PermissionResource.USER }] })

// Multiple permissions (ALL required)
@Auth({ permissions: [
    { action: PermissionAction.READ, resource: PermissionResource.USER },
    { action: PermissionAction.UPDATE, resource: PermissionResource.USER }
] })
```

### Role-Based Auth

```typescript
// Require specific roles (ANY of the roles)
@Auth({ roles: { roles: ['admin', 'moderator'], requireAll: false } })

// Require ALL specified roles
@Auth({ roles: { roles: ['admin', 'super-admin'], requireAll: true } })

// Using convenience decorators
@AdminOnly()                           // Requires 'admin' role
@AdminOrModerator()                    // Requires 'admin' OR 'moderator' role
@RequireAllRoles('admin', 'manager')   // Requires BOTH 'admin' AND 'manager' roles
@RequireAnyRole('admin', 'moderator')  // Requires 'admin' OR 'moderator' role
```

### Combined Auth

```typescript
// User needs BOTH the permission AND one of the roles
@Auth({ 
    permissions: [{ action: PermissionAction.READ, resource: PermissionResource.USER }],
    roles: { roles: ['admin'] }
})
```

## Role Configuration

### RequiredRole Interface

```typescript
interface RequiredRole {
    roles: string[];        // Array of role names
    requireAll?: boolean;   // If true, user must have ALL roles. Default: false
}
```

### Examples

```typescript
// User needs ANY of these roles
{ roles: ['admin', 'moderator'], requireAll: false }

// User needs ALL of these roles
{ roles: ['admin', 'super-admin'], requireAll: true }
```

## Usage Examples

### Controller Examples

```typescript
@Controller('example')
export class ExampleController {

    // Only authenticated users (no specific permissions/roles required)
    @JwtAuth()
    @Get('profile')
    getProfile() { /* ... */ }

    // Only admin users
    @AdminOnly()
    @Get('admin/dashboard')
    getAdminDashboard() { /* ... */ }

    // Admin or moderator users
    @AdminOrModerator()
    @Get('moderation/tools')
    getModerationTools() { /* ... */ }

    // Users with specific permission
    @Auth({ permissions: [{ action: PermissionAction.READ, resource: PermissionResource.USER }] })
    @Get('users')
    getUsers() { /* ... */ }

    // Users with both permission and role
    @Auth({ 
        permissions: [{ action: PermissionAction.DELETE, resource: PermissionResource.USER }],
        roles: { roles: ['admin'] }
    })
    @Delete('users/:id')
    deleteUser() { /* ... */ }

    // Users with multiple roles (ALL required)
    @RequireAllRoles('admin', 'super-admin')
    @Get('system/settings')
    getSystemSettings() { /* ... */ }
}
```

## Migration from Old System

### Old Way
```typescript
@Auth({ action: PermissionAction.READ, resource: PermissionResource.USER })
```

### New Way
```typescript
@Auth({ permissions: [{ action: PermissionAction.READ, resource: PermissionResource.USER }] })
```

## Guard Behavior

### PermissionsGuard
- Checks if user has required permissions
- Supports both direct permissions and permissions inherited from roles
- Uses `AuthorizationService.hasPermissions()`

### RolesGuard
- Checks if user has required roles
- Supports both "ANY" and "ALL" role requirements
- Uses `AuthorizationService.hasAnyRole()` or `AuthorizationService.hasAllRoles()`

### Combined Guards
When both permissions and roles are specified, BOTH conditions must be satisfied:
1. User must have ALL required permissions
2. User must satisfy the role requirements (ANY or ALL based on configuration)

## Error Handling

If authorization fails:
- HTTP 403 Forbidden is returned
- No additional error details are provided for security reasons

## Best Practices

1. **Use specific permissions** for fine-grained control
2. **Use roles** for broader access categories
3. **Combine both** when you need both role membership and specific permissions
4. **Use convenience decorators** (`@AdminOnly()`, etc.) for common patterns
5. **Document role requirements** in your API documentation
6. **Keep role names consistent** across your application

## Authorization Service Methods

The system uses these methods from `AuthorizationService`:

```typescript
// Check permissions
hasPermissions(user: UserInfo, requiredPermissions: RequiredPermission[]): boolean

// Check roles
hasAnyRole(user: UserInfo, roleNames: string[]): boolean
hasAllRoles(user: UserInfo, roleNames: string[]): boolean
```

## Database Schema

Ensure your roles are properly set up in the database with the correct permissions. The system relies on populated user documents that include both direct permissions and role-based permissions.
