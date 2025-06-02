import { createContext, useContext } from 'react';
import './map-list.css';

import {
    AspectRatio,
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
import { ScrollArea, color, Inset, } from "radix-ui";
const MapContext = createContext();

const MapList = () => {
    const albumsFavorites = []
    return (
        <ScrollArea type="always" scrollbars="vertical" style={{ height: 180 }}>

            <Grid row="5" gap="5" mb="7">
                {albumsFavorites.map((album) => (
                    <MapContext.Provider>
                        <MapCard />
                    </MapContext.Provider>
                ))}
            </Grid>

        </ScrollArea>
    )
};

const MapCard = () => {
    const theMap = useContext(MapContext);
    return (
        <Box p="3" m="-3">
        </Box>

    )
};

export default MapList;