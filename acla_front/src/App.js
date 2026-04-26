import './App.css';
import React, { useState } from 'react';
import LoginUser from 'views/login-user/login-user';
import RegisterUser from 'views/register-user/register-user';
import AuthProvider from "hooks/AuthProvider";
import { HashRouter as Router, Route, Routes, Navigate } from "react-router-dom"
import PrivateRoute from "views/routers/PrivateRoute";
import MainDashboard from 'views/dashboard/MainDashboard'
import UserProfile from 'views/user-profile/user-profile'
import EnvironmentProvider from 'contexts/EnvironmentContext'
import LandingPage from 'views/landing-page/LandingPage'
import { useAuth } from 'hooks/AuthProvider'

/* Redirects authenticated users away from public pages to dashboard */
const PublicRoute = ({ children }) => {
  const { token } = useAuth();
  if (token) return <Navigate to="/dashboard" />;
  return children;
};

function App() {
  return (
    <div className="App">
      <Router>
        <EnvironmentProvider>
          <AuthProvider>
            <Routes>
              {/* Landing page — only shown when not logged in */}
              <Route path="/" element={<PublicRoute><LandingPage /></PublicRoute>} />
              {/*the '/login' path is mapped to the Login component, rendering it when the URL matches*/}
              <Route path="/login" element={<PublicRoute><LoginUser /></PublicRoute>} />
              <Route path="/register" element={<PublicRoute><RegisterUser /></PublicRoute>} />
              {/*The <PrivateRoute /> component serves as a guard for protecting  */}
              <Route element={<PrivateRoute />}>
                <Route path="/dashboard" element={<MainDashboard />} />
                <Route path="/profile" element={<UserProfile />} />
              </Route>
            </Routes>
          </AuthProvider>
        </EnvironmentProvider>
      </Router>
    </div>
  );
}

export default App;
