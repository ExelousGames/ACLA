import { useContext, createContext, useState, ReactNode, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import apiService from 'services/api.service';

interface AuthContextType {
    user: string;
    token: string | null;
    userProfile: any | null;
    login: (data: { email: string; password: string }) => Promise<void>;
    logout: () => void;
    hasPermission: (action: string, resource: string) => boolean;
    hasRole: (roleName: string) => boolean;
}

//now this AuthContext component can be accessed globally. name of the context doesn't have to match the name of the file
const AuthContext = createContext<AuthContextType | undefined>(undefined);

/*Manages the user authentication state
* Whoever accesses to the value of 'AuthContext' needs to be wrapped inside the 'Provider',
*   this should be held by the highest parent component in the stack that requires access to the state
*/
const AuthProvider = ({ children }: { children: ReactNode }) => {

    const [userEmail, setUserEmail] = useState('');
    const [token, setToken] = useState(localStorage.getItem("token") || "");

    const [userProfile, setUserProfile] = useState<any>(null);
    const navigate = useNavigate();

    // Fetch user profile with permissions and roles
    const fetchUserProfile = async () => {
        try {
            const response = await apiService.get<any>('/userinfo/profile');
            setUserProfile(response.data);
        } catch (error) {
            console.error('Error fetching user profile:', error);
        }
    };

    useEffect(() => {
        const token = localStorage.getItem("token");
        const username = localStorage.getItem("username");
        if (!token || !username) {
            logout();
            return
        }
        setToken(token);
        setUserEmail(username);

        // Fetch user profile with permissions
        fetchUserProfile();
    }, []);

    //handles user login by sending a POST request to an authentication endpoint, 
    // updating the user and token state upon a successful response, and storing the token in local storage.
    const login = async (data: any) => {
        try {
            const response = await apiService.post<{ access_token: string }>('/userinfo/auth/login', data);

            if (response) {
                let tokentemp: string = response.data.access_token;
                setUserEmail(data.email);
                setToken(tokentemp);
                localStorage.setItem("token", tokentemp);
                localStorage.setItem("username", data.email);

                // Fetch user profile after successful login
                await fetchUserProfile();

                navigate("/dashboard");
            }
        } catch (error) {
            console.error('Error during login:', error);
        }
    };

    //clears user and token data, removing the token from local storage.
    const logout = () => {
        setUserEmail('');
        setToken("");
        setUserProfile(null);
        localStorage.removeItem("token");
        localStorage.removeItem("username");
        navigate("/login");
    };

    // Permission checking function
    const hasPermission = (action: string, resource: string): boolean => {
        if (!userProfile) return false;

        // Check direct permissions
        const directPermissions = userProfile.permissions || [];
        if (directPermissions.some((p: any) => p.action === action && (p.resource === resource || p.resource === 'all'))) {
            return true;
        }

        // Check for manage permission
        if (directPermissions.some((p: any) => p.action === 'manage' && (p.resource === resource || p.resource === 'all'))) {
            return true;
        }

        // Check permissions from roles
        const roles = userProfile.roles || [];
        return roles.some((role: any) => {
            const rolePermissions = role.permissions || [];
            return rolePermissions.some((p: any) =>
                (p.action === action && (p.resource === resource || p.resource === 'all')) ||
                (p.action === 'manage' && (p.resource === resource || p.resource === 'all'))
            );
        });
    };

    // Role checking function
    const hasRole = (roleName: string): boolean => {
        if (!userProfile || !userProfile.roles) return false;
        return userProfile.roles.some((role: any) => role.name === roleName);
    };

    //use Context Provider to wrap the tree of components that need this context
    return <AuthContext.Provider value={{
        token,
        user: userEmail,
        userProfile,
        login,
        logout,
        hasPermission,
        hasRole
    }}>{children}</AuthContext.Provider>;
};

//makes the authentication state and related functions available to its child components, accessible via the useAuth hook, 
// enabling components to consume authentication data and actions within the application.
export default AuthProvider;

//this is a custom hook, call 'useAuth' to use the Auth Context. access the authentication context from within different components,
// allowing them to consume the authentication state and related functions stored in the context
export const useAuth = (): AuthContextType => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};