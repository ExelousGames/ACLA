import React, { createContext, ReactNode, useContext } from 'react';
import { Environment, detectEnvironment } from '../utils/environment';

const EnvironmentContext = createContext<Environment>(detectEnvironment());

const EnvironmentProvider = ({ children }: { children: ReactNode }) => {
    const environment = detectEnvironment();
    return (
        <EnvironmentContext.Provider value={environment}>
            {children}
        </EnvironmentContext.Provider>
    );
};

export default EnvironmentProvider;

export const useEnvironment = (): Environment => {
    return useContext(EnvironmentContext);
};