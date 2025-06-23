import { useContext, useEffect } from 'react';
import './session-list.css';
import {
    Box,
    Button,
    Card,
    Flex,
    Text,
} from "@radix-ui/themes";
import { ScrollArea } from "radix-ui";
import { SessionOption } from 'data/live-analysis/live-analysis-type';
import { AnalysisContext } from '../session-analysis';
import { useEnvironment } from 'contexts/EnvironmentContext';
import { Environment } from 'utils/environment';

const SessionList = () => {
    const options: SessionOption[] = [{
        dataKey: 1,
        name: "2025-30-20 10:30:20",
        total_time: 2,
    },
    {
        dataKey: 2,
        name: "2022-30-20 10:30:20",
        total_time: 2
    }];
    const environment = useEnvironment();
    const analysisContext = useContext(AnalysisContext);
    function HandleStartNewSessionClick() {

        const now = new Date();
        analysisContext.setSession(`${now.getFullYear()}-${now.getMonth()}-${now.getDate()} ${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}`);
    }

    return (
        <div className='SessionList'>
            <ScrollArea.Root className="SessionListScrollAreaRoot">
                <ScrollArea.Viewport className="ScrollAreaViewport">
                    <Flex flexShrink="0" direction="column" gap="9">
                        {options.map((option: SessionOption) => (
                            <MapCard key={option.dataKey} dataKey={option.dataKey} name={option.name} total_time={option.total_time} />
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
            <Flex className='SessionListOptions' flexShrink='1' justify='end'>
                {environment == 'electron' ? <Button onClick={HandleStartNewSessionClick}> Start A New Session </Button> : ''}
            </Flex>
        </div>
    )
};

const MapCard = ({ dataKey, name, total_time }: SessionOption) => {
    const analysisContext = useContext(AnalysisContext);
    function mapSelected() {
        analysisContext.setSession(name);
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