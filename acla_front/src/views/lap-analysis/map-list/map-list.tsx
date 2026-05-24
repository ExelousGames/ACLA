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
import { AllMapsBasicInfoListDto, MapOption } from 'data/live-analysis/live-analysis-type';
import { AnalysisContext } from '../analysis-context';
import apiService from 'services/api.service';

const MapList = () => {

    const [options, setOptions] = useState<MapOption[]>([]);
    const [loaded, setLoaded] = useState(false);

    useEffect(() => {

        apiService.get('/racingmap/map/infolists')
            .then((result) => {
                const data = result.data as AllMapsBasicInfoListDto;
                let count = 0;

                setOptions(data.list.map((option): MapOption => {
                    count++;
                    return {
                        dataKey: count,
                        name: option.name,
                        session_count: 0,
                    } as MapOption;
                }))

            }).catch((e) => {
            }).finally(() => setLoaded(true));
    }, []);

    return (
        <ScrollArea.Root className="MapListScrollAreaRoot">
            <ScrollArea.Viewport className="ScrollAreaViewport">
                {loaded && options.length === 0 ? (
                    <div className="MapListEmptyState">
                        <div className="MapListEmptyState__eyebrow">
                            <span className="MapListEmptyState__dot" />
                            NO MAPS YET
                        </div>
                        <h3 className="MapListEmptyState__title">Record your first session</h3>
                        <p className="MapListEmptyState__sub">
                            Launch Assetto Corsa Competizione and hit <strong>Start Recording</strong> below.
                            Maps appear here once telemetry is captured.
                        </p>
                    </div>
                ) : (
                    <Flex flexShrink="0" direction="column" gap="3">
                        {options.map((option: MapOption) => (
                            //each child is a list should have a unique "key" prop
                            <MapCard key={option.dataKey} dataKey={option.dataKey} name={option.name} session_count={option.session_count} />
                        ))}
                    </Flex>
                )}
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

function MapCard({ dataKey, name, session_count }: MapOption) {
    const analysisContext = useContext(AnalysisContext);
    function mapSelected() {
        analysisContext.setMap(name);
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

