import { useContext, useEffect, useState } from 'react';
import './session-list.css';
import {
    Box,
    Card,
    Flex,
    Text,
} from "@radix-ui/themes";
import { ScrollArea } from "radix-ui";
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, SessionOption } from 'data/live-analysis/live-analysis-type';
import { AnalysisContext } from '../analysis-context';
import apiService from 'services/api.service';
import { useAuth } from 'hooks/AuthProvider';

const SessionList = () => {
    const [sessionList, setSessionList] = useState<SessionOption[]>([]);

    const analysisContext = useContext(AnalysisContext);
    const auth = useAuth();
    useEffect(() => {
        console.log('Fetching sessions for userId:', auth?.userProfile.id, 'and map:', analysisContext.mapSelected);
        const mapName = analysisContext.mapSelected;
        const userId = auth?.userProfile.id;

        if (!userId || !mapName) {
            setSessionList([]);
            return;
        }

        apiService.post('racing-session/sessionbasiclist', { map_name: mapName, user_id: userId })
            .then((result) => {
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
                setSessionList([]);
            });
    }, [analysisContext.mapSelected, auth?.userProfile.id]);
    return (
        <div className='SessionList'>
            <ScrollArea.Root className="SessionListScrollAreaRoot">
                <ScrollArea.Viewport className="ScrollAreaViewport">
                    <Flex flexShrink="0" direction="column" gap="3">
                        {sessionList.map((option: SessionOption) => (
                            <MapCard
                                key={option.dataKey}
                                dataKey={option.dataKey}
                                name={option.name}
                                total_time={option.total_time}
                                SessionId={option.SessionId}
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

const MapCard = ({ name, total_time, SessionId: id }: SessionOption) => {
    const analysisContext = useContext(AnalysisContext);
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
        <button
            type="button"
            className="SessionListCardButton"
            data-active={isSelected}
            onClick={mapSelected}
        >
            <Card className="SessionListCard" size="2">
                <Flex align="center" gap="3">
                    <Box flexGrow="1" width="0">
                        <Text as="div" size="2" truncate>
                            {name}
                        </Text>
                        <Text as="div" size="1" color="gray" truncate>
                            {lapTimeDisplay ?? 'No lap time recorded'}
                        </Text>
                    </Box>
                </Flex>
            </Card>
        </button>

    )
}
export default SessionList;