import { createContext, useContext } from 'react';
import './session-list.css';

import {
    ScrollArea,
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

const SessionContext = createContext();

const SessionList = () => {
    const albumsFavorites = []
    return (
        <ScrollArea type="always" scrollbars="vertical" style={{ height: 180 }}>

            <Grid row="5" gap="5" mb="7">
                {albumsFavorites.map((album) => (
                    <SessionContext.Provider >
                        <MapCard />
                    </SessionContext.Provider>
                ))}
            </Grid>

        </ScrollArea>

    )
};

const MapCard = () => {
    const theSession = useContext(SessionContext);
    return ({

    })
}
export default SessionList;