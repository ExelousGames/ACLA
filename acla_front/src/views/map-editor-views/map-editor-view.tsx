import { createContext, useState } from "react";
import MapEditorMapList from "./map-editor-map-list/map-editor-map-list";


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

    return (
        <MapEditorContext.Provider value={{ mapSelected: mapSelected, setMap: setMap }}>
            <div>
                <MapEditorMapList>
                </MapEditorMapList>
                <button> Add new Map</button>
            </div>

        </MapEditorContext.Provider>
    );


}

export default MapEditorView;