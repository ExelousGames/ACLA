# Combined Authentication Decorators

## Overview

The `@Auth()` decorator combines JWT authentication and permission checking into a single, convenient decorator. This simplifies controller code and ensures consistent authentication patterns.

## Available Decorators

### `@Auth(...permissions)`
Combines JWT authentication with permission checking.

```typescript
@Auth({ action: PermissionAction.CREATE, resource: PermissionResource.USER })
@Post()
createUser() {}
```

### `@JwtAuth()`
JWT authentication only (no permission checking).

```typescript
@JwtAuth()
@Get('profile')
getProfile() {}
```

### `@LocalAuth()`
Local authentication (for login endpoints).

```typescript
@LocalAuth()
@Post('auth/login')
login() {}
```

## Usage Examples

### Before (Multiple Decorators)
```typescript
@UseGuards(AuthGuard('jwt'), PermissionsGuard)
@RequirePermissions({ action: PermissionAction.CREATE, resource: PermissionResource.USER })
@Post()
createUser() {}
```

### After (Combined Decorator)
```typescript
@Auth({ action: PermissionAction.CREATE, resource: PermissionResource.USER })
@Post()
createUser() {}
```

### Multiple Permissions
```typescript
@Auth(
  { action: PermissionAction.READ, resource: PermissionResource.USER },
  { action: PermissionAction.UPDATE, resource: PermissionResource.USER }
)
@Put(':id')
updateUser() {}
```

## Benefits

1. **Cleaner Code**: Single decorator instead of multiple
2. **Consistency**: Ensures proper guard order
3. **Maintainability**: Easier to update authentication logic
4. **Type Safety**: Maintains all type checking from original decorators
