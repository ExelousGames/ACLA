import React, { createContext, useContext } from 'react';
import { Environment, detectEnvironment } from '../utils/environment';

const EnvironmentContext = createContext<Environment>(detectEnvironment());

export const EnvironmentProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const environment = detectEnvironment();
    return (
        <EnvironmentContext.Provider value={environment}>
            {children}
        </EnvironmentContext.Provider>
    );
};

export const useEnvironment = (): Environment => {
    return useContext(EnvironmentContext);
};