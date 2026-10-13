import { useContext, useEffect, useState } from 'react';
import './session-list.css';
import {
    AlertDialog,
    Box,
    Button,
    Card,
    Flex,
    IconButton,
    Text,
} from "@radix-ui/themes";
import { TrashIcon } from '@radix-ui/react-icons';
import { ScrollArea } from "radix-ui";
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, SessionOption } from 'data/live-analysis/live-analysis-type';
import { AnalysisContext } from '../analysis-context';
import apiService from 'services/api.service';
import { useAuth } from 'hooks/AuthProvider';

const SessionList = () => {
    const [sessionList, setSessionList] = useState<SessionOption[]>([]);
    const [loaded, setLoaded] = useState(false);

    const analysisContext = useContext(AnalysisContext);
    const auth = useAuth();
    useEffect(() => {
        console.log('Fetching sessions for userId:', auth?.userProfile.id, 'and map:', analysisContext.mapSelected);
        const mapName = analysisContext.mapSelected;
        const userId = auth?.userProfile.id;
        let cancelled = false;
        setSessionList([]);
        setLoaded(false);

        if (!userId || !mapName) {
            setLoaded(true);
            return;
        }

        apiService.post('racing-session/sessionbasiclist', { map_name: mapName, user_id: userId })
            .then((result) => {
                if (cancelled) return;
                const data = result.data as SessionBasicInfoListDto;
                setSessionList(
                    data.list.map((session, index) => ({
                        dataKey: index,
                        name: session.name,
                        SessionId: session.sessionId,
                    }))
                );
            })
            .catch(() => {
                if (!cancelled) setSessionList([]);
            })
            .finally(() => {
                if (!cancelled) setLoaded(true);
            });
        return () => { cancelled = true; };
    }, [analysisContext.mapSelected, auth?.userProfile.id]);

    const handleDeleted = (sessionId: string) => {
        setSessionList((sessions) => sessions.filter((session) => session.SessionId !== sessionId));
        analysisContext.setSession((session) => session?.SessionId === sessionId ? null : session);
    };
    return (
        <div className='SessionList'>
            <ScrollArea.Root className="SessionListScrollAreaRoot">
                <ScrollArea.Viewport className="ScrollAreaViewport">
                    <Flex flexShrink="0" direction="column" gap="3">
                        {loaded && sessionList.length === 0 && (
                            <Text size="2" color="gray">No recorded sessions for this track.</Text>
                        )}
                        {sessionList.map((option: SessionOption) => (
                            <SessionCard
                                key={option.SessionId}
                                dataKey={option.dataKey}
                                name={option.name}
                                total_time={option.total_time}
                                SessionId={option.SessionId}
                                onDeleted={handleDeleted}
                            />
                        ))}
                    </Flex>
                </ScrollArea.Viewport>
                <ScrollArea.Scrollbar
                    className="ScrollAreaScrollbar"
                    orientation="vertical"
                >
                    <ScrollArea.Thumb className="ScrollAreaThumb" />
                </ScrollArea.Scrollbar>
                <ScrollArea.Scrollbar
                    className="ScrollAreaScrollbar"
                    orientation="horizontal"
                >
                    <ScrollArea.Thumb className="ScrollAreaThumb" />
                </ScrollArea.Scrollbar>
                <ScrollArea.Corner className="ScrollAreaCorner" />
            </ScrollArea.Root>

        </div>
    )
};

const formatLapTime = (totalTime?: number) => {
    if (totalTime === undefined || totalTime === null || Number.isNaN(totalTime)) {
        return null;
    }

    const minutes = Math.floor(totalTime / 60);
    const seconds = totalTime % 60;
    const wholeSeconds = Math.floor(seconds)
        .toString()
        .padStart(2, '0');
    const milliseconds = Math.round((seconds - Math.floor(seconds)) * 1000)
        .toString()
        .padStart(3, '0');

    return `${minutes}:${wholeSeconds}.${milliseconds}`;
};

const SessionCard = ({ name, total_time, SessionId: id, onDeleted }: SessionOption & {
    onDeleted: (sessionId: string) => void;
}) => {
    const analysisContext = useContext(AnalysisContext);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const [deleteError, setDeleteError] = useState<string | null>(null);

    const deleteSession = async () => {
        if (isDeleting) return;
        setIsDeleting(true);
        setDeleteError(null);
        try {
            await apiService.delete(`racing-session/${encodeURIComponent(id)}`);
            onDeleted(id);
        } catch (error: any) {
            if (error?.status === 404 && error?.data?.message === 'Session not found') {
                onDeleted(id);
            } else {
                setDeleteError('Could not delete this session. Please try again.');
            }
        } finally {
            setIsDeleting(false);
        }
    };
    function mapSelected() {
        //if no previous session, create a new one.
        const newSession: RacingSessionDetailedInfoDto = {
            session_name: name,
            SessionId: id,
            map: '',
            car: '',
            user_id: '',
            points: [],
            data: []
        };
        analysisContext.setSession(newSession);
    }

    const isSelected = analysisContext.sessionSelected?.SessionId === id;
    const lapTimeDisplay = formatLapTime(total_time);

    return (
        <AlertDialog.Root open={deleteDialogOpen} onOpenChange={(open) => {
            if (isDeleting) return;
            setDeleteDialogOpen(open);
            setDeleteError(null);
        }}>
            <Card className="SessionListCard" size="2" data-active={isSelected}>
                <Flex align="center" gap="3">
                    <button
                        type="button"
                        className="SessionListCardButton"
                        onClick={mapSelected}
                        disabled={isDeleting}
                    >
                        <Text as="div" size="2" truncate>
                            {name}
                        </Text>
                        <Text as="div" size="1" color="gray" truncate>
                            {lapTimeDisplay ?? 'No lap time recorded'}
                        </Text>
                    </button>
                    <AlertDialog.Trigger>
                        <IconButton
                            variant="ghost"
                            color="red"
                            aria-label={`Delete session ${name}`}
                            title="Delete session"
                        >
                            <TrashIcon />
                        </IconButton>
                    </AlertDialog.Trigger>
                </Flex>
            </Card>
            <AlertDialog.Content maxWidth="450px" onEscapeKeyDown={(event) => {
                if (isDeleting) event.preventDefault();
            }}>
                <AlertDialog.Title>Delete recorded session?</AlertDialog.Title>
                <AlertDialog.Description size="2">
                    Delete “{name}” and its recorded telemetry? This cannot be undone.
                </AlertDialog.Description>
                {deleteError && (
                    <Box mt="3">
                        <Text role="alert" size="2" color="red">{deleteError}</Text>
                    </Box>
                )}
                <Flex gap="3" mt="4" justify="end">
                    <AlertDialog.Cancel>
                        <Button variant="soft" color="gray" disabled={isDeleting}>Cancel</Button>
                    </AlertDialog.Cancel>
                    <Button color="red" disabled={isDeleting} onClick={() => { void deleteSession(); }}>
                        {isDeleting ? 'Deleting...' : 'Delete session'}
                    </Button>
                </Flex>
            </AlertDialog.Content>
        </AlertDialog.Root>

    )
}
export default SessionList;
