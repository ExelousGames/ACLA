import { createContext, Key, useContext, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Arc, Circle, Rect, Line, Group } from 'react-konva';
import { PointsToBezierPoints } from 'utils/curve-tobezier/curve-to-bezier';
import { offsetPolyBezier, Point, pointsOnBezierCurves } from 'utils/curve-tobezier/points-on-curve';

type RacingTurningPoint = { id: number, point: Point };
type BezierPoints = { id: number, point: Point };
const SessionAnalysis = () => {

    // Define virtual size for our scene
    let containerWidth = window.innerWidth;
    let containerHeight = window.innerWidth;
    // State to track current scale and dimensions
    const [stageSize, setStageSize] = useState({
        width: containerWidth,
        height: containerHeight,
    });
    const [turningPoints, setTurningPoints] = useState<RacingTurningPoint[]>(createInitialShapes());
    const [bezierPoints, setBezierPoints] = useState<BezierPoints[]>([]);
    // Reference to parent container
    const containerRef = useRef<HTMLInputElement>(null);

    ///////////////functions////////////////////

    function createInitialShapes(): RacingTurningPoint[] {
        return [
            { id: 0, point: [0, 0] },
            { id: 1, point: [0, 20] },
            { id: 2, point: [0, 40] },
        ]
    }

    // Function to handle resize
    const updateSize = () => {
        if (!containerRef.current) return;

        // Get container width
        containerWidth = containerRef.current.offsetWidth;
        containerHeight = containerRef.current.offsetHeight;

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

        setTurningPoints(turningPoints.map((turningPoint: { id: number, point: Point }) => {
            if (turningPoint.id !== id) return turningPoint;

            let pointPosition: any[] = [e.target.x(), e.target.y()];

            if (e.target.x() <= 0) {
                pointPosition[0] = 0;
            }
            if (e.target.x() as number >= stageSize.width) {
                pointPosition[0] = stageSize.width;
            }

            if (e.target.y() < 0) {
                pointPosition[1] = 0;
            }
            if (e.target.y() as number >= stageSize.height) {
                pointPosition[1] = stageSize.height;
            }

            //set posisition. for some reason, point wont set again at boundary, this fix it temporaryily
            e.target.absolutePosition({
                x: pointPosition[0],
                y: pointPosition[1]
            });

            return { ...turningPoint, point: [pointPosition[0], pointPosition[1]] }

        }));

        //recalculate controlling point for bezier curve since the turning point moved
        AddBezierControllingPoints(turningPoints);
    }

    const handleDragEnd = (e: any, id: any) => {

    };


    /**
     * Add controlling points to input points, also cached it for later use
     * @param turningPoints 
     */
    function AddBezierControllingPoints(turningPoints: RacingTurningPoint[]): Point[] | undefined {
        const copyPoints = [...turningPoints] as RacingTurningPoint[];
        const points = PointsToBezierPoints(extractRacingTurningPointToPoint(copyPoints))
        let index = 0;
        let result: BezierPoints[] = [];
        points.forEach((point) => {
            index++;
            result.push({ id: index, point: point });
        })
        setBezierPoints(result);
        return points;
    }

    return (
        <div ref={containerRef} style={{ width: '100%', height: '90%' }}>

            <Stage width={stageSize.width} height={stageSize.height} >
                <Layer>

                    <Line
                        points={exportPointsForDrawing(exportCurbBezierPoints(bezierPoints, 'left'))}
                        stroke="red" strokeWidth={4}
                    />

                    <Line
                        points={exportPointsForDrawing(extractBezierPointToPoint(bezierPoints))}
                        stroke="red" strokeWidth={4}
                    />

                    <Line
                        points={exportPointsForDrawing(exportCurbBezierPoints(bezierPoints, 'right'))}
                        stroke="red" strokeWidth={4}
                    />
                    {turningPoints.map((turningPoint: { id: Key, point: Point }) => (
                        <Group key={turningPoint.id} id={`group-${turningPoint.id}`} x={turningPoint.point[0]} y={turningPoint.point[1]} draggable onDragMove={(e) => handleDragMove(e, turningPoint.id)} onDragEnd={(e) => handleDragEnd(e, turningPoint.id)}>
                            <Circle key={turningPoint.id} radius={20} fill={"red"} name={turningPoint.id.toString()} />
                        </Group>
                    ))}

                    {bezierPoints.map((point: { id: Key, point: Point }) => (

                        <Circle key={point.id} x={point.point[0]} y={point.point[1]} radius={10} fill={"blue"} name={point.id.toString()} />
                    ))}
                </Layer>
            </Stage>
        </div>
    );


};

function convert_1D_array_to_2d_array(points: number[]): Point[] {
    const result: Point[] = [];
    for (let i = 0; i < points.length; i += 2) {
        const point = points.slice(i, i + 2);
        result.push([point[0], point[1]]);
    }
    return result;
}

function extractRacingTurningPointToPoint(points: RacingTurningPoint[]): Point[] {
    return points.reduce((acc, curr): Point[] => {
        return [...acc, curr.point];
    }, [] as Point[]);
}

function extractBezierPointToPoint(points: BezierPoints[]): Point[] {
    if (!points) return [];
    return points.reduce((acc, curr): Point[] => {
        return [...acc, curr.point];
    }, [] as Point[]);
}

/**
 * flat point[] to 1d number[]
 * @param points 
 * @returns 
 */
function convert_Points_to_1d_array(points: Point[]): number[] {
    if (points.length === 0) return [];

    // let result: number[] = [];
    // for (let i = 0; i < points.length; i += 1) {

    //     result = result.concat(points[i]);

    // }

    return points.flat();;
}

/**
 * give bezier points of turning points, return curbs point of these turning points
 * @param points 
 * @param direction 
 * @returns 
 */
function exportCurbBezierPoints(points?: BezierPoints[], direction: 'left' | 'right' = 'left'): Point[] {
    if (!points || points.length === 0) return [];
    return PointsToBezierPoints(offsetPolyBezier(extractBezierPointToPoint(points), 50, direction));
}
/**
 * input points which should contains controlling points, and convert them into Curves and convert the result into 1d array readable by the Line component
 * @param bezierPoints 
 * @returns 1d array for Konva Line component
 */
function exportPointsForDrawing(points?: Point[]): number[] {
    if (!points) return [];
    //-> smooth the points to more points -> convert into 1d array
    return convert_Points_to_1d_array(pointsOnBezierCurves(points));
}


export default SessionAnalysis;

