import './App.css';
import React, { useState } from 'react';
import LoginUser from 'views/login-user/login-user';
import AuthProvider from "hooks/AuthProvider";
import { HashRouter as Router, Route, Routes } from "react-router-dom"
import PrivateRoute from "views/routers/PrivateRoute";
import MainDashboard from 'views/dashboard/MainDashboard'


function App() {
  return (
    <div className="App">
      <Router>
        <AuthProvider>
          <Routes>
            {/*the '/login' path is mapped to the Login component, rendering it when the URL matches*/}
            <Route path="/login" element={<LoginUser />} />
            {/*The <PrivateRoute /> component serves as a guard for protecting  */}
            <Route element={<PrivateRoute />}>
              <Route path="/" element={<MainDashboard />} />
              <Route path="/dashboard" element={<MainDashboard />} />
            </Route>
          </Routes>
        </AuthProvider>
      </Router>
    </div>
  );
}

export default App;
