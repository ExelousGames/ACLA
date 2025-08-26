# Permission System Documentation

This document outlines the implementation and usage of the role-based access control (RBAC) system in the ACLA application.

## Overview

The system implements a flexible permission-based authorization mechanism with the following components:
- **Users**: Have direct permissions and roles
- **Roles**: Collections of permissions that can be assigned to users
- **Permissions**: Define what actions can be performed on what resources

## Backend Implementation

### Schemas

#### Permission Schema
```typescript
{
  id: string,
  name: string,
  description: string,
  action: PermissionAction, // create, read, update, delete, manage
  resource: PermissionResource // user, racing_session, racing_map, all
}
```

#### Role Schema
```typescript
{
  id: string,
  name: string,
  description: string,
  permissions: Permission[],
  isActive: boolean
}
```

#### User Schema
```typescript
{
  id: string,
  email: string,
  password: string,
  roles: Role[],
  permissions: Permission[], // Direct permissions
  isActive: boolean
}
```

### Usage in Controllers

#### Protecting Endpoints with Permissions

```typescript
import { RequirePermissions } from '../../common/decorators/permissions.decorator';
import { PermissionsGuard } from '../../common/guards/permissions.guard';
import { PermissionAction, PermissionResource } from '../../schemas/permission.schema';

@Controller('users')
export class UserController {
  @UseGuards(AuthGuard('jwt'), PermissionsGuard)
  @RequirePermissions({ action: PermissionAction.CREATE, resource: PermissionResource.USER })
  @Post()
  createUser(@Body() createUserDto: CreateUserDto) {
    // Only users with CREATE permission on USER resource can access this
  }

  @UseGuards(AuthGuard('jwt'), PermissionsGuard)
  @RequirePermissions({ action: PermissionAction.DELETE, resource: PermissionResource.USER })
  @Delete(':id')
  deleteUser(@Param('id') id: string) {
    // Only users with DELETE permission on USER resource can access this
  }
}
```

#### Multiple Permissions

```typescript
@RequirePermissions(
  { action: PermissionAction.READ, resource: PermissionResource.USER },
  { action: PermissionAction.UPDATE, resource: PermissionResource.USER }
)
```

### Authorization Service

The `AuthorizationService` provides methods for checking permissions:

```typescript
// Check if user has specific permission
hasPermission(permissions: Permission[], action: PermissionAction, resource: PermissionResource): boolean

// Check if user has all required permissions
hasPermissions(user: UserInfo, requiredPermissions: RequiredPermission[]): boolean

// Check if user has any of the specified roles
hasAnyRole(user: UserInfo, roleNames: string[]): boolean

// Check if user has all specified roles
hasAllRoles(user: UserInfo, roleNames: string[]): boolean
```

## Frontend Implementation

### AuthProvider Enhancements

The `AuthProvider` now includes:
- User profile with permissions and roles
- Permission checking functions
- Role checking functions

```typescript
const { hasPermission, hasRole, userProfile } = useAuth();
```

### Protected Components

#### Using ProtectedComponent

```jsx
import ProtectedComponent from '../components/ProtectedComponent';
import { COMMON_PERMISSIONS } from '../constants/permissions';

<ProtectedComponent 
  requiredPermission={COMMON_PERMISSIONS.CREATE_USER}
  fallback={<div>Access Denied</div>}
>
  <Button>Create User</Button>
</ProtectedComponent>

<ProtectedComponent 
  requiredRole="admin"
  fallback={<div>Admin access required</div>}
>
  <AdminPanel />
</ProtectedComponent>
```

#### Using Higher-Order Component

```jsx
import withAuthorization from '../components/withAuthorization';
import { PERMISSION_ACTIONS, PERMISSION_RESOURCES } from '../constants/permissions';

const AdminPanel = () => <div>Admin Panel Content</div>;

const ProtectedAdminPanel = withAuthorization(AdminPanel, {
  requiredPermission: {
    action: PERMISSION_ACTIONS.MANAGE,
    resource: PERMISSION_RESOURCES.ALL
  },
  fallback: () => <div>Access Denied</div>
});
```

### Permission Constants

Use the predefined constants for consistency:

```typescript
import { 
  PERMISSION_ACTIONS, 
  PERMISSION_RESOURCES, 
  ROLES,
  COMMON_PERMISSIONS 
} from '../constants/permissions';

// Check permissions
hasPermission(PERMISSION_ACTIONS.CREATE, PERMISSION_RESOURCES.USER)

// Check roles
hasRole(ROLES.ADMIN)

// Use common permissions
<ProtectedComponent requiredPermission={COMMON_PERMISSIONS.CREATE_USER}>
```

## Permission Hierarchy

The system supports a permission hierarchy:

1. **Exact Match**: `action: 'create', resource: 'user'` matches exactly
2. **Manage Permission**: `action: 'manage', resource: 'user'` grants all actions on user resource
3. **Global Manage**: `action: 'manage', resource: 'all'` grants all actions on all resources

## Best Practices

### Backend

1. Always use `@UseGuards(AuthGuard('jwt'), PermissionsGuard)` for protected endpoints
2. Use specific permissions rather than broad ones when possible
3. Group related permissions into roles
4. Use the `RequirePermissions` decorator for clear permission requirements

### Frontend

1. Use `ProtectedComponent` for conditional rendering
2. Use the HOC pattern for entire components that require authorization
3. Always provide meaningful fallback content
4. Use permission constants for consistency
5. Check permissions at the component level, not just route level

## Example Workflow

1. **Create Permissions**: Define what actions can be performed
   ```typescript
   const permissions = [
     { action: 'create', resource: 'user', name: 'Create User' },
     { action: 'read', resource: 'user', name: 'Read User' },
     { action: 'update', resource: 'user', name: 'Update User' },
     { action: 'delete', resource: 'user', name: 'Delete User' }
   ];
   ```

2. **Create Roles**: Group permissions logically
   ```typescript
   const adminRole = {
     name: 'admin',
     permissions: [/* all permissions */]
   };
   
   const userRole = {
     name: 'user', 
     permissions: [/* read permissions only */]
   };
   ```

3. **Assign to Users**: Give users roles and/or direct permissions
   ```typescript
   const user = {
     email: 'admin@example.com',
     roles: [adminRole],
     permissions: [] // additional direct permissions if needed
   };
   ```

4. **Protect Backend Endpoints**: Use guards and decorators
   ```typescript
   @RequirePermissions({ action: 'create', resource: 'user' })
   @UseGuards(AuthGuard('jwt'), PermissionsGuard)
   @Post()
   createUser() {}
   ```

5. **Protect Frontend Components**: Use protection components
   ```jsx
   <ProtectedComponent requiredPermission={{ action: 'create', resource: 'user' }}>
     <CreateUserButton />
   </ProtectedComponent>
   ```

## Database Setup

When setting up the system, you'll need to:
1. Create initial permissions for all actions and resources
2. Create default roles (admin, user, moderator)
3. Assign permissions to roles
4. Create initial admin user with admin role

This ensures the system has the necessary data to function properly.
