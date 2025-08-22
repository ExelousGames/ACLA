import { createContext } from "react";

interface MapEditorViewSharedData {
    mapSelected: string | null,
    setMap: (map: string | null) => void;
}

export const mapEditorContext = createContext<MapEditorViewSharedData>({
    mapSelected: '',
    setMap: (map: string | null) => { },
});