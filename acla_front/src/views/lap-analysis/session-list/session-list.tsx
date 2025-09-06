import { useContext, useEffect, useState } from 'react';
import './session-list.css';
import {
    Box,
    Button,
    Card,
    Flex,
    Text,
} from "@radix-ui/themes";
import { ScrollArea } from "radix-ui";
import { MapOption, RacingSessionDetailedInfoDto, SessionBasicInfoListDto, SessionOption } from 'data/live-analysis/live-analysis-type';
import { AnalysisContext } from '../session-analysis';
import { useEnvironment } from 'contexts/EnvironmentContext';
import { Environment } from 'utils/environment';
import apiService from 'services/api.service';
import { useAuth } from 'hooks/AuthProvider';

const SessionList = () => {
    const [seesionList, setSessionList] = useState([] as SessionOption[]);

    const analysisContext = useContext(AnalysisContext);
    const auth = useAuth();
    useEffect(() => {
        console.log('Fetching sessions for userId:', auth?.userProfile.id, 'and map:', analysisContext.mapSelected);
        if (!auth?.userProfile.id || !analysisContext.mapSelected) {
            setSessionList([]);
            return;
        }
        apiService.post('racing-session/sessionbasiclist', { map_name: analysisContext.mapSelected, user_id: auth?.userProfile.id })
            .then((result) => {
                const data = result.data as SessionBasicInfoListDto;
                let count = 0;
                setSessionList(data.list.map((seesion): SessionOption => {
                    count++;
                    return {
                        dataKey: count,
                        name: seesion.name,
                        SessionId: seesion.sessionId
                    } as SessionOption;
                }))
            }).catch((e) => {
            });
    }, [analysisContext.mapSelected]);
    return (
        <div className='SessionList'>
            <ScrollArea.Root className="SessionListScrollAreaRoot">
                <ScrollArea.Viewport className="ScrollAreaViewport">
                    <Flex flexShrink="0" direction="column" gap="9">
                        {seesionList.map((option: SessionOption) => (
                            <MapCard key={option.dataKey} dataKey={option.dataKey} name={option.name} total_time={option.total_time} SessionId={option.SessionId} />
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

const MapCard = ({ dataKey, name, total_time, SessionId: id }: SessionOption) => {
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

    return (
        <button className="Button" onClick={mapSelected}>
            <Card >
                <Flex align="center" gap="3">
                    <Box asChild width="60px" height="60px">
                        <img />
                    </Box>
                    <Box flexGrow="1" width="0">
                        <Text as="div" size="2" truncate>
                            {name}
                        </Text>
                        <Text as="div" size="1" color="gray" truncate>
                            {total_time}
                        </Text>
                    </Box>
                </Flex>
            </Card>
        </button >

    )
}
export default SessionList;