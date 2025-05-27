import React from "react";
import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "hooks/AuthProvider";

//handling authentication
const PrivateRoute = () => {

    //use 'useAuth' hook from the AuthProvider to access user authentication data
    const user = useAuth();

    //If the user does not possess a token, indicating they are not logged in
    if (!user.token) {
        //the code triggers a redirect to the /login route using the <Navigate> component.
        return <Navigate to="/login" />;
    }

    //Otherwise, it renders the child components nested within the PrivateRoute component accessed via <Outlet />
    return <Outlet />;
};

export default PrivateRoute;