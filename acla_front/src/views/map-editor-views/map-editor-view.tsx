import { createContext, useState } from "react";
import MapEditorMapList from "./map-editor-map-list/map-editor-map-list";
import { Box, Tabs } from "@radix-ui/themes";
import MapEditor from "./map-editor/map-editor";


interface MapEditorViewSharedData {
    mapSelected: string | null,
    setMap: (map: string | null) => void;
}

export const MapEditorContext = createContext<MapEditorViewSharedData>({
    mapSelected: '',
    setMap: (map: string | null) => { },
});

const MapEditorView = () => {

    const [mapSelected, setMap] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState('mapLists');
    return (
        <MapEditorContext.Provider value={{ mapSelected: mapSelected, setMap: setMap }}>
            <Tabs.Root className="MapEditorViewTabsRoot" defaultValue="mapLists" value={activeTab} onValueChange={setActiveTab}>
                <Tabs.List className="map-editor-tablists" justify="start">
                    <Tabs.Trigger value="mapLists">Maps</Tabs.Trigger>
                    {mapSelected == null ? "" : <Tabs.Trigger value="mapEditor">{mapSelected}</Tabs.Trigger>}
                </Tabs.List>

                <Box className="live-analysis-container" >
                    <Tabs.Content className="TabContent" value="mapLists">
                        <MapEditorMapList>
                        </MapEditorMapList>
                    </Tabs.Content>

                    <Tabs.Content className="TabContent" value="mapEditor">
                        <MapEditor></MapEditor>
                    </Tabs.Content>
                </Box >
            </Tabs.Root>
        </MapEditorContext.Provider>
    );


}

export default MapEditorView;