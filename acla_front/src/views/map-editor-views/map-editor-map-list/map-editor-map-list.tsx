import { createContext, useContext, useEffect, useState } from 'react';
import './map-editor-map-list.css';

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
import apiService from 'services/api.service';
import { MapEditorContext } from '../map-editor-view';

// Type for the create map response
type CreateMapResponse = {
    success: boolean;
    message: string;
    map?: any;
};

const MapEditorMapList = (props: any) => {

    const [options, setOptions] = useState([] as MapOption[]);
    const [isCreating, setIsCreating] = useState(false);
    const [newMapName, setNewMapName] = useState('');

    const loadMaps = () => {
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
                console.error('Error loading maps:', e);

                // Handle specific error cases
                if (e.status === 403) {
                    console.error('Access forbidden: User may not have sufficient permissions or may not be logged in');
                    // You might want to redirect to login or show a permission error
                } else if (e.status === 401) {
                    console.error('Authentication required: Please log in');
                    // Redirect to login page
                } else {
                    console.error('Failed to load maps:', e.message);
                }
            });
    };

    const handleCreateMap = async () => {
        if (!newMapName.trim()) {
            alert('Please enter a map name');
            return;
        }

        setIsCreating(true);
        try {
            const response = await apiService.post<CreateMapResponse>('/racingmap/map/create', { name: newMapName.trim() });

            if (response.data.success) {
                // Clear the input and refresh the list
                setNewMapName('');
                loadMaps();
                alert('Map created successfully!');
            } else {
                alert(response.data.message || 'Failed to create map');
            }
        } catch (error: any) {
            console.error('Error creating map:', error);
            alert(error.message || 'Failed to create map');
        } finally {
            setIsCreating(false);
        }
    };

    useEffect(() => {
        loadMaps();
    }, []);

    return (
        <Box className="map-editor-map-list-container" p="6" style={{ height: '100vh', background: 'var(--gray-1)' }}>
            {/* Add New Map Section */}
            <Card size="3" style={{ marginBottom: '24px' }}>
                <Flex direction="column" gap="4">
                    <Flex align="center" gap="2">
                        <Heading size="5" weight="bold" color="gray" highContrast>
                            Create New Map
                        </Heading>
                    </Flex>

                    <Flex gap="3" align="end">
                        <Box flexGrow="1">
                            <Text size="2" weight="medium" color="gray" mb="2" as="label">
                                Map Name
                            </Text>
                            <TextField.Root
                                size="3"
                                placeholder="Enter map name..."
                                value={newMapName}
                                onChange={(e) => setNewMapName(e.target.value)}
                                onKeyPress={(e) => {
                                    if (e.key === 'Enter') {
                                        handleCreateMap();
                                    }
                                }}
                                style={{ width: '100%' }}
                            />
                        </Box>
                        <Button
                            size="3"
                            onClick={handleCreateMap}
                            disabled={isCreating || !newMapName.trim()}
                            style={{
                                background: isCreating ? 'var(--gray-6)' : 'var(--blue-9)',
                                cursor: isCreating || !newMapName.trim() ? 'not-allowed' : 'pointer'
                            }}
                        >
                            {isCreating ? 'Creating...' : 'Create Map'}
                        </Button>
                    </Flex>
                </Flex>
            </Card>

            {/* Maps List */}
            <Card size="3" style={{ flex: 1, minHeight: 0 }}>
                <Flex direction="column" gap="3" style={{ height: '100%' }}>
                    <Flex align="center" gap="2" mb="3">
                        <Heading size="4" weight="bold" color="gray" highContrast>
                            Map Collection
                        </Heading>
                        <Badge variant="soft" color="blue" ml="auto">
                            {options.length} maps
                        </Badge>
                    </Flex>

                    <ScrollArea.Root style={{ flex: 1 }}>
                        <ScrollArea.Viewport style={{ height: '100%', padding: '8px' }}>
                            <Flex direction="column" gap="3">
                                {options.length === 0 ? (
                                    <Box
                                        p="6"
                                        style={{
                                            textAlign: 'center',
                                            border: '2px dashed var(--gray-6)',
                                            borderRadius: '12px',
                                            background: 'var(--gray-2)'
                                        }}
                                    >
                                        <Text size="3" color="gray">
                                            No maps yet. Create your first map above! 🚀
                                        </Text>
                                    </Box>
                                ) : (
                                    options.map((option: MapOption) => (
                                        <MapCard key={option.dataKey} dataKey={option.dataKey} name={option.name} />
                                    ))
                                )}
                            </Flex>
                        </ScrollArea.Viewport>
                        <ScrollArea.Scrollbar orientation="vertical">
                            <ScrollArea.Thumb />
                        </ScrollArea.Scrollbar>
                    </ScrollArea.Root>
                </Flex>
            </Card>
        </Box>
    )
};

function MapCard({ dataKey, name }: MapOption) {
    const mapEditorContext = useContext(MapEditorContext);

    function mapSelected() {
        mapEditorContext.setMap(name);
    }

    return (
        <Card
            asChild
            size="3"
            style={{
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                border: '1px solid var(--gray-6)',
                background: 'var(--gray-1)'
            }}
        >
            <button onClick={mapSelected} style={{ all: 'unset', width: '100%' }}>
                <Flex align="center" justify="between" p="4">
                    <Flex align="center" gap="4" flexGrow="1">
                        {/* Map Icon */}
                        <Box
                            style={{
                                width: '48px',
                                height: '48px',
                                borderRadius: '12px',
                                background: 'var(--blue-3)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                border: '1px solid var(--blue-6)'
                            }}
                        >
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path
                                    d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 01.553-.894L9 2l6 3 5.447-2.724A1 1 0 0121 3.118v10.764a1 1 0 01-.553.894L15 17l-6-3z"
                                    stroke="var(--blue-9)"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                />
                            </svg>
                        </Box>

                        {/* Map Info */}
                        <Flex direction="column" align="start" gap="1" flexGrow="1">
                            <Heading size="4" weight="bold" color="gray" highContrast>
                                {name}
                            </Heading>
                            <Flex align="center" gap="2">
                                <Badge variant="soft" color="green" size="1">
                                    <Box
                                        style={{
                                            width: '6px',
                                            height: '6px',
                                            borderRadius: '50%',
                                            background: 'var(--green-9)',
                                            marginRight: '4px'
                                        }}
                                    />
                                    Ready to edit
                                </Badge>
                                <Text size="2" color="gray">
                                    Last updated today
                                </Text>
                            </Flex>
                        </Flex>
                    </Flex>

                    {/* Action Indicator */}
                    <Flex align="center" gap="2">
                        <Badge variant="outline" color="blue" size="1">
                            Edit
                        </Badge>
                        <Box style={{ color: 'var(--gray-9)' }}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path
                                    d="M5 12h14M12 5l7 7-7 7"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                />
                            </svg>
                        </Box>
                    </Flex>
                </Flex>
            </button>
        </Card>
    );
};

export default MapEditorMapList;

