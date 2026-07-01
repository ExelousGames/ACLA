import React, { createContext, ReactNode, useCallback, useContext, useEffect, useState } from 'react';
import { AiLabelsResponse, getAiLabels } from 'services/aiLabelsService';

interface AiLabelsContextType {
    labels: AiLabelsResponse | null;
    loading: boolean;
    error: string | null;
    refreshLabels: () => Promise<AiLabelsResponse | null>;
    getLabelName: (labelId: string) => string | undefined;
    getLabelId: (labelName: string) => string | undefined;
    getCategoryLabels: (category: string) => string[];
}

const AiLabelsContext = createContext<AiLabelsContextType | undefined>(undefined);

const getErrorMessage = (error: unknown): string => {
    if (error instanceof Error) {
        return error.message;
    }
    return 'Failed to retrieve AI labels';
};

const AiLabelsProvider = ({ children }: { children: ReactNode }) => {
    const [labels, setLabels] = useState<AiLabelsResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const refreshLabels = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const nextLabels = await getAiLabels();
            setLabels(nextLabels);
            return nextLabels;
        } catch (err) {
            setError(getErrorMessage(err));
            return null;
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        void refreshLabels();
    }, [refreshLabels]);

    const getLabelName = useCallback(
        (labelId: string) => labels?.label_mapping[labelId],
        [labels],
    );

    const getLabelId = useCallback(
        (labelName: string) => labels?.label_name_to_id[labelName],
        [labels],
    );

    const getCategoryLabels = useCallback(
        (category: string) => labels?.label_categories[category] ?? [],
        [labels],
    );

    return (
        <AiLabelsContext.Provider value={{
            labels,
            loading,
            error,
            refreshLabels,
            getLabelName,
            getLabelId,
            getCategoryLabels,
        }}>
            {children}
        </AiLabelsContext.Provider>
    );
};

export default AiLabelsProvider;

export const useAiLabels = (): AiLabelsContextType => {
    const context = useContext(AiLabelsContext);
    if (!context) {
        throw new Error('useAiLabels must be used within an AiLabelsProvider');
    }
    return context;
};
