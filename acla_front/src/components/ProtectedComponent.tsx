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
    fallbackNavigation?: string;
}

//this component checks for user permissions and roles, and renders the children if the user has the required permissions/roles
const ProtectedComponent: React.FC<ProtectedComponentProps> = ({
    children,
    requiredPermission,
    requiredRole,
    fallbackNavigation = "/login"
}) => {
    const { hasPermission, hasRole } = useAuth();
    const navigate = useNavigate();

    // Check role if required
    if (requiredRole && !hasRole(requiredRole)) {
        navigate(fallbackNavigation);
        return <div>You don't have role to view this menu</div>;
    }

    // Check permission if required
    if (requiredPermission && !hasPermission(requiredPermission.action, requiredPermission.resource)) {
        navigate(fallbackNavigation);
        return <div>You don't have permission to view this menu</div>;
    }

    return <>{children}</>;
};

export default ProtectedComponent;
