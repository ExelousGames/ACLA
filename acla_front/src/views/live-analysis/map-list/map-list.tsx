import { createContext, useContext } from 'react';
import './map-list.css';

import {
    AspectRatio,
    Badge,
    Avatar,
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
import { MapOption } from 'data/live-analysis/live-analysis-data';
import { AnalysisContext } from '../live-analysis';

const MapList = (setMapState: any) => {
    const options: MapOption[] = [{
        key: 1,
        name: "Track 1",
        session_count: 2
    },
    {
        key: 2,
        name: "Track 2",
        session_count: 2
    }];

    return (
        <ScrollArea.Root className="MapListScrollAreaRoot">
            <ScrollArea.Viewport className="ScrollAreaViewport">
                <Flex flexShrink="0" direction="column" gap="9">
                    {options.map((option: MapOption) => (

                        <MapCard {...option} />

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

function MapCard(mapinfo: MapOption) {
    const { mapContext } = useContext(AnalysisContext);
    function mapSelected() {
        mapContext.setMap(mapinfo);
    }

    return (
        <button className="Button" onClick={mapSelected}>
            <Card >
                <Flex align="center" gap="3" key={mapinfo.key}>
                    <Box asChild width="60px" height="60px">
                        <img />
                    </Box>
                    <Box flexGrow="1" width="0">
                        <Text as="div" size="2" truncate>
                            {mapinfo.name}
                        </Text>
                        <Text as="div" size="1" color="gray" truncate>
                            {mapinfo.session_count}
                        </Text>
                    </Box>
                </Flex>
            </Card>
        </button >

    )
};

export default MapList;

