import { createContext, Key, useContext, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Arc, Circle, Rect, Line, Group } from 'react-konva';
import { Point } from 'utils/curve-tobezier/points-on-curve';

const SessionAnalysis = () => {

    // Define virtual size for our scene
    let containerWidth = window.innerWidth;
    let containerHeight = window.innerWidth;
    // State to track current scale and dimensions
    const [stageSize, setStageSize] = useState({
        width: containerWidth,
        height: containerHeight,
    });
    const [shapes, setShapes] = useState(createInitialShapes());
    // Reference to parent container
    const containerRef = useRef<HTMLInputElement>(null);

    ///////////////functions////////////////////

    // Function to handle resize
    const updateSize = () => {
        if (!containerRef.current) return;

        // Get container width
        containerWidth = containerRef.current.offsetWidth;
        containerHeight = containerRef.current.offsetHeight;
        console.log(containerWidth, containerHeight);
        // Update state with new dimensions
        setStageSize({
            width: containerWidth,
            height: containerHeight,
        });
    };

    // Update on mount and when window resizes
    useEffect(() => {
        updateSize();
    }, []);

    function handleDragMove(e: any, id: any) {
        const target = e.target;
        const targetRect = target.getClientRect();

        setShapes(shapes.map((shape: { id: any; }) => {
            if (shape.id === id) {
                return shape;
            }
            const shapeGroup = target.parent.parent.findOne(`#group-${shape.id}`);
            if (!shapeGroup) return shape;
        }));

        setShapes(shapes.map((shape: { id: any; }) => {
            if (shape.id !== id) return;

            let point: any[] = [e.target.x(), e.target.y()];

            console.log(e.target.x(), e.target.y());
            if (e.target.x() <= 0) {
                point[0] = 0;
            }
            if (e.target.x() as number >= stageSize.width) {
                point[0] = stageSize.width;
            }

            if (e.target.y() < 0) {
                point[1] = 0;
            }
            if (e.target.y() as number >= stageSize.height) {
                point[1] = stageSize.height;
            }

            //set posisition
            e.target.absolutePosition({
                x: point[0],
                y: point[1]
            });

            //record position
            if (shape.id === id) {
                return { ...shape, x: point[0], y: point[1] }
            }
            return shape;
        }

        ));
    }

    const handleDragEnd = (e: any, id: any) => {

    };

    function createInitialShapes(): any {
        return [
            { id: 0, x: 0, y: 0 }
        ]
    }
    return (
        <div ref={containerRef} style={{ width: '100%', height: '90%' }}>

            <Stage width={stageSize.width} height={stageSize.height} >
                <Layer>
                    {shapes.map((shape: { id: Key; x: number | undefined; y: number | undefined; }) => (
                        <Group key={shape.id} id={`group-${shape.id}`} x={shape.x} y={shape.y} draggable onDragMove={(e) => handleDragMove(e, shape.id)} onDragEnd={(e) => handleDragEnd(e, shape.id)}>
                            <Circle key={shape.id} radius={20} fill={"red"} name={shape.id.toString()} />
                        </Group>
                    ))}
                </Layer>
            </Stage>
        </div>
    );
};

function conver_1D_array_to_2d_array(points: number[]): Point[] {
    const result: Point[] = [];
    for (let i = 0; i < points.length; i += 2) {
        const point = points.slice(i, i + 2);
        result.push({ x: point[0], y: point[1] });
    }
    return result;
}


export default SessionAnalysis;


