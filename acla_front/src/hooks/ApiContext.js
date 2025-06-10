// src/contexts/ApiContext.js
import React, { createContext, useContext } from 'react';
import * as ApiService from '../services/apiService';

const ApiContext = createContext();

export const ApiProvider = ({ children }) => {
    return (
        <ApiContext.Provider value={ApiService}>
            {children}
        </ApiContext.Provider>
    );
};

export const useApi = () => {
    const context = useContext(ApiContext);
    if (!context) {
        throw new Error('useApi must be used within an ApiProvider');
    }
    return context;
};