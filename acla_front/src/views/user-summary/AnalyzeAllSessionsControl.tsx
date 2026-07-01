import { useCallback, useEffect, useRef, useState } from 'react';
import { Button, Flex, Spinner, Text } from '@radix-ui/themes';
import { UpdateIcon } from '@radix-ui/react-icons';
import apiService from 'services/api.service';

type AnalysisJob = {
    id: string;
    status: 'queued' | 'running' | 'completed' | 'failed';
    progress?: Record<string, any>;
    error?: string | null;
};

type AnalyzeAllSessionsControlProps = {
    onCompleted: () => Promise<void>;
};

const ACTIVE_STATUSES = new Set(['queued', 'running']);
const USER_SUMMARY_SESSION_LIMIT = 10;

const AnalyzeAllSessionsControl = ({ onCompleted }: AnalyzeAllSessionsControlProps) => {
    const [job, setJob] = useState<AnalysisJob | null>(null);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const completedJobRef = useRef<string | null>(null);

    const isActive = !!job && ACTIVE_STATUSES.has(job.status);

    const loadStatus = useCallback(async () => {
        const response = await apiService.get<AnalysisJob | null>('/userinfo/summary/analyze-all/status');
        setJob(response.data);
        return response.data;
    }, []);

    useEffect(() => {
        void loadStatus().catch(() => {
            setError('Unable to load analysis status');
        });
    }, [loadStatus]);

    useEffect(() => {
        if (!isActive) {
            return;
        }

        const interval = window.setInterval(() => {
            void loadStatus().catch(() => {
                setError('Unable to refresh analysis status');
            });
        }, 4000);

        return () => window.clearInterval(interval);
    }, [isActive, loadStatus]);

    useEffect(() => {
        if (job?.status !== 'completed' || completedJobRef.current === job.id) {
            return;
        }

        completedJobRef.current = job.id;
        setMessage('Analysis completed and saved');
        void onCompleted().catch(() => {
            setError('Analysis completed, but summary reload failed');
        });
    }, [job, onCompleted]);

    const handleAnalyze = async () => {
        setMessage('');
        setError('');
        try {
            const response = await apiService.post<AnalysisJob>('/userinfo/summary/analyze-all', {
                sessionLimit: USER_SUMMARY_SESSION_LIMIT,
            });
            setJob(response.data);
            setMessage('Analysis queued');
        } catch (requestError: any) {
            if (requestError?.status === 409) {
                setError('Analysis is already queued or running for this user');
                await loadStatus();
                return;
            }
            setError('Unable to queue analysis');
        }
    };

    const statusText = job
        ? `${job.status}${job.progress?.message ? `: ${job.progress.message}` : ''}`
        : 'No analysis queued';

    return (
        <Flex direction="column" gap="2" className="user-summary-analysis-control">
            <Button onClick={handleAnalyze} disabled={isActive}>
                {isActive ? <Spinner size="1" /> : <UpdateIcon />}
                {isActive ? 'Analyzing' : 'Analyze Recent Sessions'}
            </Button>
            <Text size="2" color={error ? 'red' : 'gray'}>
                {error || message || statusText}
            </Text>
        </Flex>
    );
};

export default AnalyzeAllSessionsControl;
