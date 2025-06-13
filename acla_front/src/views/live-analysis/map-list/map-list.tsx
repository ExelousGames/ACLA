import { createContext, useContext, useEffect, useState } from 'react';
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
import { AllMapsBasicInfoListDto, MapOption } from 'data/live-analysis/live-analysis-data';
import { AnalysisContext } from '../live-analysis';
import apiService from 'services/api.service';

const MapList = (setMapState: any) => {

    const [options, setOptions] = useState([{
        key: 1,
        name: "Track 1",
        session_count: 0,

    },
    {
        key: 2,
        name: "Track 2",
        session_count: 0
    }] as MapOption[]);

    useEffect(() => {

        apiService.get('/racingmap/map/infolists')
            .then((result) => {

                const data = result.data as AllMapsBasicInfoListDto;
                let count = 0;

                setOptions(data.list.map((option): MapOption => {
                    count++;
                    return {
                        key: count,
                        name: option.name,
                        session_count: 0,
                    } as MapOption;
                }))
                console.log(options);
                return result.data;
            }).catch((e) => {

            });

    }, []);

    return (
        <ScrollArea.Root className="MapListScrollAreaRoot">
            <ScrollArea.Viewport className="ScrollAreaViewport">
                <Flex flexShrink="0" direction="column" gap="9">
                    {options.map((option: MapOption) => (
                        <MapCard key={option.key} name={option.name} session_count={option.session_count} />
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

function MapCard({ key, name, session_count }: MapOption) {
    const { mapContext } = useContext(AnalysisContext);
    function mapSelected() {
        mapContext.setMap({ key, name, session_count });
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
                            {session_count}
                        </Text>
                    </Box>
                </Flex>
            </Card>
        </button >

    )
};

export default MapList;

