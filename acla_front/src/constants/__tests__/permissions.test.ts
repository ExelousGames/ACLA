import {
    PERMISSION_ACTIONS,
    PERMISSION_RESOURCES,
    ROLES,
    createPermission,
    COMMON_PERMISSIONS,
} from '../permissions';

describe('permissions constants', () => {
    describe('PERMISSION_ACTIONS', () => {
        it('should define all expected actions', () => {
            expect(PERMISSION_ACTIONS.CREATE).toBe('create');
            expect(PERMISSION_ACTIONS.READ).toBe('read');
            expect(PERMISSION_ACTIONS.UPDATE).toBe('update');
            expect(PERMISSION_ACTIONS.DELETE).toBe('delete');
            expect(PERMISSION_ACTIONS.MANAGE).toBe('manage');
        });
    });

    describe('PERMISSION_RESOURCES', () => {
        it('should define all expected resources', () => {
            expect(PERMISSION_RESOURCES.USER).toBe('user');
            expect(PERMISSION_RESOURCES.RACING_SESSION).toBe('racing_session');
            expect(PERMISSION_RESOURCES.RACING_MAP).toBe('racing_map');
            expect(PERMISSION_RESOURCES.ALL).toBe('all');
        });
    });

    describe('ROLES', () => {
        it('should define all expected roles', () => {
            expect(ROLES.ADMIN).toBe('admin');
            expect(ROLES.USER).toBe('user');
            expect(ROLES.MODERATOR).toBe('moderator');
        });
    });

    describe('createPermission', () => {
        it('should create a permission object with action and resource', () => {
            const permission = createPermission('read', 'user');
            expect(permission).toEqual({ action: 'read', resource: 'user' });
        });
    });

    describe('COMMON_PERMISSIONS', () => {
        it('should define correct permission combinations', () => {
            expect(COMMON_PERMISSIONS.CREATE_USER).toEqual({ action: 'create', resource: 'user' });
            expect(COMMON_PERMISSIONS.READ_USER).toEqual({ action: 'read', resource: 'user' });
            expect(COMMON_PERMISSIONS.UPDATE_USER).toEqual({ action: 'update', resource: 'user' });
            expect(COMMON_PERMISSIONS.DELETE_USER).toEqual({ action: 'delete', resource: 'user' });
            expect(COMMON_PERMISSIONS.MANAGE_ALL).toEqual({ action: 'manage', resource: 'all' });
        });
    });
});
