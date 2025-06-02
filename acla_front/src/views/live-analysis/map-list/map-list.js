import { createContext, useContext } from 'react';
import './map-list.css';

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

const MapContext = createContext();

const MapList = () => {

    return (
        <ScrollArea type="always" scrollbars="vertical" style={{ height: 180 }}>

            <Grid row="5" gap="5" mb="7">
                {albumsFavorites.map((album) => (
                    <MapContext.Provider>
                        <MapCard
                            focusable={focusable}
                            title={album.name}
                            caption={album.artist}
                            cover={album.cover}
                            color={album.color}
                            key={album.name}
                        />
                    </MapContext.Provider>
                ))}
            </Grid>

        </ScrollArea>
    )
};

const MapCard = () => {
    const theMap = useContext(MapContext);
    return (
        <Hover.Root>
            <Box p="3" m="-3">
                <Box mb="2" position="relative">
                    <Card
                        style={{
                            boxShadow: `0 8px 48px -16px ${color.replace("%)", "%, 0.6)")}`,
                        }}
                    >
                        <Inset>
                            <AspectRatio ratio={1}>
                                <img
                                    src={cover}
                                    style={{
                                        width: "100%",
                                        height: "100%",
                                        objectFit: "cover",
                                    }}
                                />
                            </AspectRatio>
                        </Inset>
                    </Card>

                    <Hover.Show>
                        <Flex gap="2" position="absolute" bottom="0" right="0" m="2">
                            <IconButton tabIndex={-1} radius="full" size="3">
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="currentcolor"
                                    viewBox="0 0 30 30"
                                    width="20"
                                    height="20"
                                    style={{ marginRight: -2 }}
                                >
                                    <path d="M 6 3 A 1 1 0 0 0 5 4 A 1 1 0 0 0 5 4.0039062 L 5 15 L 5 25.996094 A 1 1 0 0 0 5 26 A 1 1 0 0 0 6 27 A 1 1 0 0 0 6.5800781 26.8125 L 6.5820312 26.814453 L 26.416016 15.908203 A 1 1 0 0 0 27 15 A 1 1 0 0 0 26.388672 14.078125 L 6.5820312 3.1855469 L 6.5800781 3.1855469 A 1 1 0 0 0 6 3 z" />
                                </svg>
                            </IconButton>
                        </Flex>
                    </Hover.Show>
                </Box>

                <Flex direction="column" position="relative" align="start">
                    <Link
                        href="#"
                        onClick={(e) => e.preventDefault()}
                        tabIndex={focusable ? undefined : -1}
                        size="2"
                        weight="medium"
                        color="gray"
                        highContrast
                        style={{ textDecoration: "none" }}
                    >
                        {title}
                    </Link>
                    <Text size="2" color="gray">
                        {caption}
                    </Text>
                </Flex>
            </Box>
        </Hover.Root>
    )
};

export default MapList;