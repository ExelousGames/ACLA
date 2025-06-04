import { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Arc, Circle, Rect } from 'react-konva';

const SessionAnalysis = () => {

    // Define virtual size for our scene
    let containerWidth = 500;
    let containerHeight = 500;

    // State to track current scale and dimensions
    const [stageSize, setStageSize] = useState({
        width: containerWidth,
        height: containerHeight,
        scale: 1
    });

    // Reference to parent container
    const containerRef = useRef<HTMLInputElement>(null);

    // Function to handle resize
    const updateSize = () => {
        if (!containerRef.current) return;

        // Get container width
        containerWidth = containerRef.current.offsetWidth;
        containerHeight = containerRef.current.offsetHeight;

        // Update state with new dimensions
        setStageSize({
            width: containerWidth,
            height: containerWidth,
            scale: 1
        });
    };

    // Update on mount and when window resizes
    useEffect(() => {

        updateSize();


    }, []);

    return (
        <div ref={containerRef} style={{ width: '100%', height: '90%' }}>

            <Stage width={stageSize.width}
                height={stageSize.height}
                scaleX={stageSize.scale}
                scaleY={stageSize.scale}>

                <Layer>
                    <Circle
                        radius={50}
                        fill="red"
                        x={containerWidth / 2}
                        y={containerHeight / 2}
                    />
                    <Rect
                        fill="green"
                        x={containerWidth - 100}
                        y={containerHeight - 100}
                        width={100}
                        height={100}
                    />
                </Layer>
            </Stage>
        </div>
    );
};



export default SessionAnalysis;

