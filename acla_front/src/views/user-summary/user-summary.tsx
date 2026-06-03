import { useCallback, useEffect, useState } from 'react';
import {
    Box,
    Button,
    Card,
    Flex,
    Heading,
    Separator,
    Text,
    TextArea
} from '@radix-ui/themes';
import apiService from 'services/api.service';
import AnalyzeAllSessionsControl from './AnalyzeAllSessionsControl';
import './user-summary.css';

type UserSummaryResponse = {
    summary: Record<string, any>;
};

const UserSummary = () => {
    const [summaryText, setSummaryText] = useState('{}');
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');

    const loadSummary = useCallback(async () => {
        try {
            const response = await apiService.get<UserSummaryResponse>('/userinfo/summary');
            setSummaryText(JSON.stringify(response.data.summary || {}, null, 2));
        } catch (error) {
            setError('Unable to load user summary');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        void loadSummary();
    }, [loadSummary]);

    const handleSave = async () => {
        setError('');
        setMessage('');

        let summary: Record<string, any>;
        try {
            summary = JSON.parse(summaryText);
        } catch (error) {
            setError('Summary must be valid JSON');
            return;
        }

        setIsSaving(true);
        try {
            const response = await apiService.put<UserSummaryResponse>('/userinfo/summary', { summary });
            setSummaryText(JSON.stringify(response.data.summary || {}, null, 2));
            setMessage('Saved');
        } catch (error) {
            setError('Unable to save user summary');
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <Box className="user-summary-container">
            <Card className="user-summary-card">
                <Flex direction="column" gap="4">
                    <Flex justify="between" align="center" gap="3" wrap="wrap">
                        <Box>
                            <Heading size="6">User Summary</Heading>
                            <Text size="2" color="gray">Driving skills JSON</Text>
                        </Box>
                        <Flex gap="3" align="start" wrap="wrap">
                            <AnalyzeAllSessionsControl onCompleted={loadSummary} />
                            <Button onClick={handleSave} disabled={isLoading || isSaving}>
                                {isSaving ? 'Saving' : 'Save'}
                            </Button>
                        </Flex>
                    </Flex>

                    <Separator />

                    <TextArea
                        className="user-summary-editor"
                        value={summaryText}
                        onChange={(event) => setSummaryText(event.target.value)}
                        disabled={isLoading}
                        spellCheck={false}
                    />

                    {(message || error) && (
                        <Text size="2" color={error ? 'red' : 'green'}>
                            {error || message}
                        </Text>
                    )}
                </Flex>
            </Card>
        </Box>
    );
};

export default UserSummary;
