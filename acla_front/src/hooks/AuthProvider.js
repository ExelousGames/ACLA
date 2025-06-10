import { useContext, createContext, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from 'axios';

//now this AuthContext component can be accessed globally. name of the context doesn't have to match the name of the file
const AuthContext = createContext();

/*Manages the user authentication state
* Whoever accesses to the value of 'AuthContext' needs to be wrapped inside the 'Provider',
*   this should be held by the highest parent component in the stack that requires access to the state
*/
const AuthProvider = ({ children }) => {

    const [user, setUser] = useState(null);
    const [token, setToken] = useState(localStorage.getItem("token") || "");
    const navigate = useNavigate();

    //backend server api
    const serverIPandPort = process.env.REACT_APP_BACKEND_SERVER_IP + ":" + process.env.REACT_APP_BACKEND_PROXY_PORT
    const server_url_header = 'http://' + serverIPandPort

    //handles user login by sending a POST request to an authentication endpoint, 
    // updating the user and token state upon a successful response, and storing the token in local storage.
    const loginAction = async (data) => {
        axios.post(server_url_header + '/userinfo/auth/login', data)
            .then(response => {
                alert(response.server_url_header);
                if (response) {
                    setUser(response.data.user);
                    setToken(response.data.access_token);
                    localStorage.setItem("token", response.data.token);
                    navigate("/dashboard");
                    return;
                }
            })
            .catch(error => console.error('Error creating task:', error));
    };

    //clears user and token data, removing the token from local storage.
    const logOut = () => {
        console.log("log out");
        setUser(null);
        setToken("");
        localStorage.removeItem("token");
        navigate("/login");
    };

    //use Context Provider to wrap the tree of components that need this context
    return <AuthContext.Provider value={{ token, user, loginAction, logOut }}>{children}</AuthContext.Provider>;
};

//makes the authentication state and related functions available to its child components, accessible via the useAuth hook, 
// enabling components to consume authentication data and actions within the application.
export default AuthProvider;

//this is a custom hook, call 'useAuth' to use the Auth Context. access the authentication context from within different components,
// allowing them to consume the authentication state and related functions stored in the context
export const useAuth = () => {
    return useContext(AuthContext);
};