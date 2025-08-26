import React from 'react';
import { useAuth } from '../hooks/AuthProvider';

interface AuthorizationOptions {
    requiredPermission?: {
        action: string;
        resource: string;
    };
    requiredRole?: string;
    fallback?: React.ComponentType;
}

// Higher-Order Component for authorization
const withAuthorization = <P extends object>(
    WrappedComponent: React.ComponentType<P>,
    options: AuthorizationOptions
) => {
    return (props: P) => {
        const { hasPermission, hasRole } = useAuth();

        // Check role if required
        if (options.requiredRole && !hasRole(options.requiredRole)) {
            return options.fallback ? <options.fallback /> : <div>Access Denied</div>;
        }

        // Check permission if required
        if (options.requiredPermission && !hasPermission(options.requiredPermission.action, options.requiredPermission.resource)) {
            return options.fallback ? <options.fallback /> : <div>Access Denied</div>;
        }

        return <WrappedComponent {...props} />;
    };
};

export default withAuthorization;
