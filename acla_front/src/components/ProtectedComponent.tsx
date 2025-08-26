import React from 'react';
import { useAuth } from '../hooks/AuthProvider';

interface ProtectedComponentProps {
    children: React.ReactNode;
    requiredPermission?: {
        action: string;
        resource: string;
    };
    requiredRole?: string;
    fallback?: React.ReactNode;
}

const ProtectedComponent: React.FC<ProtectedComponentProps> = ({
    children,
    requiredPermission,
    requiredRole,
    fallback = null
}) => {
    const { hasPermission, hasRole } = useAuth();

    // Check role if required
    if (requiredRole && !hasRole(requiredRole)) {
        return <>{fallback}</>;
    }

    // Check permission if required
    if (requiredPermission && !hasPermission(requiredPermission.action, requiredPermission.resource)) {
        return <>{fallback}</>;
    }

    return <>{children}</>;
};

export default ProtectedComponent;
