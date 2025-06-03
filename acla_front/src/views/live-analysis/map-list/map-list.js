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
const MapContext = createContext();

const MapList = (setMapState) => {
    const albumsFavorites = [1, 2, 3, 4, 5]

    return (

        <ScrollArea.Root className="MapListScrollAreaRoot">
            <ScrollArea.Viewport className="ScrollAreaViewport">
                <Grid row="5" gap="5" mb="7">
                    {albumsFavorites.map((album) => (
                        <MapContext.Provider>
                            <MapCard />
                        </MapContext.Provider>
                    ))}
                </Grid>
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

const MapCard = (setMapState) => {
    const theMap = useContext(MapContext);
    return (

        <Button >
            <Flex gap="3" align="center">
                <Avatar variant="solid" fallback="A" />
                <Box>
                    <Text as="div" size="2" weight="bold">
                        Teodros Girmay
                    </Text>
                    <Text as="div" size="2" color="gray">
                        Engineering
                    </Text>
                </Box>
            </Flex>
        </Button>


    )
};

export default MapList;