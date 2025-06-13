import { createContext, useContext } from 'react';
import './session-list.css';
import {
    Avatar,
    Badge,
    Box,
    Button,
    Card,
    Checkbox,
    DropdownMenu,
    Flex,
    Grid,
    Heading,
    IconButton,
    Link,
    Separator,
    Strong,
    Switch,
    Text,
    TextField,
    Theme,
} from "@radix-ui/themes";
import { ScrollArea } from "radix-ui";
import { SessionOption } from 'data/live-analysis/live-analysis-data';
import { AnalysisContext } from '../live-analysis';

const SessionList = () => {
    const options: SessionOption[] = [{
        key: 1,
        name: "2025-30-20 10:30:20",
        total_time: 2,

    },
    {
        key: 2,
        name: "2022-30-20 10:30:20",
        total_time: 2
    }];
    return (
        <ScrollArea.Root className="MapListScrollAreaRoot">
            <ScrollArea.Viewport className="ScrollAreaViewport">
                <Flex flexShrink="0" direction="column" gap="9">
                    {options.map((option: SessionOption) => (
                        <MapCard key={option.key} name={option.name} total_time={option.total_time} />
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


    )
};

const MapCard = ({ key, name, total_time }: SessionOption) => {
    const { sessionContext } = useContext(AnalysisContext);
    function mapSelected() {
        sessionContext.setSession({ key, name, total_time });
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