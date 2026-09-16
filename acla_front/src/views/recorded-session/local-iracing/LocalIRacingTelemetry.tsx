import { useContext, useEffect, useRef, useState } from 'react';
import { Box, Button, Flex, Heading, Text } from '@radix-ui/themes';
import { useEnvironment } from 'contexts/EnvironmentContext';
import { AnalysisContext } from '../analysis-context';
import SessionAnalysisSplit from '../sessionAnalysis/session-analysis-split';
import { readLocalTelemetry } from './read-local-telemetry';

export default function LocalIRacingTelemetry() {
    const environment = useEnvironment();
    const { sessionSelected, setSession, setMap } = useContext(AnalysisContext);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState('');
    const [error, setError] = useState<string | null>(null);
    const importRef = useRef<AbortController | null>(null);
    useEffect(() => () => { importRef.current?.abort(); }, []);

    const openFile = async () => {
        if (importRef.current) return;
        const controller = new AbortController();
        importRef.current = controller;
        setLoading(true);
        setError(null);
        setProgress('Opening iRacing telemetry...');
        let convertedPath: string | undefined;
        try {
            if (!window.electronAPI?.importLocalIRacingTelemetry) {
                throw new Error('Restart the updated desktop app to open .ibt files.');
            }
            const imported = await window.electronAPI.importLocalIRacingTelemetry();
            if (!imported) return;
            convertedPath = imported.filePath;
            if (controller.signal.aborted) return;
            setProgress('Loading telemetry samples...');
            const rows = await readLocalTelemetry(imported.filePath, controller.signal, (count) => {
                setProgress(`Loading telemetry samples: ${count.toLocaleString()} / ${imported.rowCount.toLocaleString()}`);
            });
            if (controller.signal.aborted) return;
            if (!rows.length) throw new Error('This .ibt file contains no telemetry samples.');
            setMap(imported.track || null);
            setSession({
                SessionId: `local-ibt:${imported.filePath}`,
                session_name: imported.fileName,
                storage: 'local',
                game_recorded_from: 'iracing_recorded',
                map: imported.track,
                car: imported.car,
                user_id: '',
                points: [],
                data: rows,
            });
        } catch (cause) {
            if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Could not open the .ibt file.');
        } finally {
            if (convertedPath) await window.electronAPI.deleteTempFile(convertedPath).catch(() => undefined);
            if (!controller.signal.aborted) {
                importRef.current = null;
                setLoading(false);
            }
        }
    };

    return (
        <div className="LiveAnalysisTabsRoot">
            <Box p="4">
                <Flex direction="column" gap="2">
                    <Heading size="4">iRacing recorded telemetry</Heading>
                    <Text size="2" color="gray">Open a locally saved .ibt file to review its telemetry and playback. Files stay on this computer.</Text>
                    {environment === 'electron' ? (
                        <>
                            <Text size="1" color="gray">Exit the car in iRacing before opening the file so recording has finished.</Text>
                            <Flex align="center" gap="3" wrap="wrap">
                                <Button onClick={() => { void openFile(); }} disabled={loading}>
                                    {loading ? 'Loading...' : 'Open .ibt file'}
                                </Button>
                                {loading && <Text role="status" size="2">{progress}</Text>}
                            </Flex>
                        </>
                    ) : <Text size="2">Open the desktop app to analyze local iRacing .ibt files.</Text>}
                    {error && <Text role="alert" color="red" size="2">{error}</Text>}
                    {sessionSelected && <Text size="2">{sessionSelected.session_name} · {sessionSelected.map || 'Unknown track'} · {sessionSelected.car || 'Unknown car'} · {sessionSelected.data.length.toLocaleString()} samples</Text>}
                </Flex>
            </Box>
            {sessionSelected && <Box className="live-analysis-container"><SessionAnalysisSplit /></Box>}
        </div>
    );
}
