import React, { createContext, ReactNode, useCallback, useContext, useEffect, useState } from 'react';
import apiService from 'services/api.service';

type UserSummaryResponse = {
    summary: Record<string, any>;
};

interface UserSummaryContextType {
    userSummary: Record<string, any>;
    userSummaryLoading: boolean;
    userSummaryError: string;
    loadUserSummary: () => Promise<void>;
}

const UserSummaryContext = createContext<UserSummaryContextType | undefined>(undefined);

const UserSummaryProvider = ({ children }: { children: ReactNode }) => {
    const [userSummary, setUserSummary] = useState<Record<string, any>>({});
    const [userSummaryLoading, setUserSummaryLoading] = useState(false);
    const [userSummaryError, setUserSummaryError] = useState('');

    const loadUserSummary = useCallback(async () => {
        setUserSummaryLoading(true);
        setUserSummaryError('');

        try {
            const response = await apiService.get<UserSummaryResponse>('/userinfo/summary');
            setUserSummary(response.data.summary || {});
        } catch (error) {
            setUserSummaryError('Unable to load user summary');
            console.error('Error fetching user summary:', error);
        } finally {
            setUserSummaryLoading(false);
        }
    }, []);

    useEffect(() => {
        void loadUserSummary();
    }, [loadUserSummary]);

    return (
        <UserSummaryContext.Provider value={{
            userSummary,
            userSummaryLoading,
            userSummaryError,
            loadUserSummary,
        }}>
            {children}
        </UserSummaryContext.Provider>
    );
};

export default UserSummaryProvider;

export const useUserSummary = (): UserSummaryContextType => {
    const context = useContext(UserSummaryContext);
    if (!context) {
        throw new Error('useUserSummary must be used within a UserSummaryProvider');
    }
    return context;
};
