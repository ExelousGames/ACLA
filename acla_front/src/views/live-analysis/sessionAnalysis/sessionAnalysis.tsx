import { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Arc, Circle, Rect } from 'react-konva';

const SessionAnalysis = () => {

    // Define virtual size for our scene
    const sceneWidth = 0;
    const sceneHeight = 0;

    // State to track current scale and dimensions
    const [stageSize, setStageSize] = useState({
        width: sceneWidth,
        height: sceneHeight,
        scale: 1
    });

    // Reference to parent container
    const containerRef = useRef<HTMLInputElement>(null);

    // Function to handle resize
    const updateSize = () => {
        if (!containerRef.current) return;

        // Get container width
        const containerWidth = containerRef.current.offsetWidth;
        const containerHeight = containerRef.current.offsetHeight;
        console.log(containerWidth, containerHeight);
        // Calculate scale
        const widthScale = containerWidth / sceneWidth;
        const heightScale = containerHeight / sceneWidth;
        // Update state with new dimensions
        setStageSize({
            width: containerWidth,
            height: containerHeight,
            scale: 1
        });
    };

    // Update on mount and when window resizes
    useEffect(() => {
        updateSize();

    }, []);

    return (
        <div ref={containerRef} style={{ width: '100%', height: '100%' }}>

            <Stage width={stageSize.width}
                height={stageSize.height}
                scaleX={stageSize.scale}
                scaleY={stageSize.scale}>

                <Layer>
                    <Circle
                        radius={50}
                        fill="red"
                        x={sceneWidth / 2}
                        y={sceneHeight / 2}
                    />
                    <Rect
                        fill="green"
                        x={sceneWidth - 100}
                        y={sceneHeight - 100}
                        width={100}
                        height={100}
                    />
                </Layer>
            </Stage>
        </div>
    );
};



export default SessionAnalysis;

