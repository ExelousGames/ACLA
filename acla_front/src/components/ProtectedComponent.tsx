import React from 'react';
import { useAuth } from '../hooks/AuthProvider';
import { useNavigate } from 'react-router-dom';

interface ProtectedComponentProps {
    children: React.ReactNode;
    requiredPermission?: {
        action: string;
        resource: string;
    };
    requiredRole?: string;
    fallback?: string;
    redirectTo?: string;
}


/**
 * this component checks for user permissions and roles, and renders the children if the user has the required permissions/roles
 * @param requiredPermission required permission in order to access the children
 * @param requiredRole required role in order to access the children
 * @param fallback fallback content to display if access is denied
 * @param redirectTo optional redirect path if access is denied
 * @returns
 */
const ProtectedComponent: React.FC<ProtectedComponentProps> = ({
    children,
    requiredPermission,
    requiredRole,
    fallback = "",
    redirectTo
}) => {
    const { hasPermission, hasRole } = useAuth();
    const navigate = useNavigate();

    // Check role if required
    if (requiredRole && !hasRole(requiredRole)) {
        if (redirectTo) {
            navigate(redirectTo);
            return null;
        }
        return <div>{fallback}</div>;
    }

    // Check permission if required
    if (requiredPermission && !hasPermission(requiredPermission.action, requiredPermission.resource)) {
        if (redirectTo) {
            navigate(redirectTo);
            return null;
        }
        return <div>{fallback}</div>;
    }

    return <>{children}</>;
};

export default ProtectedComponent;
