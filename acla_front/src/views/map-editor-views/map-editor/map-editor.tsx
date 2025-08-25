import { useContext, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Circle, Line, Group, Image } from 'react-konva';
import { AddControlPoints } from 'utils/curve-tobezier/curve-to-bezier';
import { offsetBezierPoints, Point, getBezierTangent, getEndDirection, pointOnCubicBezierSpline, calculateSegmentLengths } from 'utils/curve-tobezier/points-on-curve';
import useImage from 'use-image';
import image from 'assets/map2.png'
import apiService from 'services/api.service';
import { MapInfo, RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { useEnvironment } from 'contexts/EnvironmentContext';
import { ContextMenu, DropdownMenu, IconButton } from '@radix-ui/themes';
import { Html } from 'react-konva-utils';
import { PlusIcon } from '@radix-ui/react-icons';
import { MapEditorContext } from '../map-editor-view';

type RacingTurningPoint = {
    position: Point,
    type: number,
    index: number, //type and index are used together. some points are index sensitive
    description?: string,
    info?: string,

    //variables will be saved online
    variables?: [{ key: string, value: string }],

    //variables will not be saved
    isDoubleClicked: boolean
};
type CurbTurningPoint = { id: number, position: Point };
type BezierPoints = { id: number, position: Point };
type RacingLinePoint = { id: number, position: Point };
const MapEditor = () => {

    // Define virtual size for our scene
    let containerWidth = window.innerWidth;
    let containerHeight = window.innerWidth;
    // State to track current scale and dimensions
    const [stageSize, setStageSize] = useState({
        width: containerWidth,
        height: containerHeight,
    });

    const mapEditorContext = useContext(MapEditorContext);

    //track turning points
    const [turningPoints, setTurningPoints] = useState<RacingTurningPoint[]>([]);

    //points which construct the curve (turningPoints + auto created controlling points)
    const [bezierPoints, setBezierPoints] = useState<BezierPoints[]>([]);
    const [segmentLengths, setSegmentLengths] = useState<number[]>([]);

    const [leftCurbTurningPoints, setLeftCurbTurningPoints] = useState<CurbTurningPoint[]>([]);
    const [leftCurbBezierPoints, setLeftCrubBezierPoints] = useState<BezierPoints[]>([]);
    const [rightCurbTurningPoints, setRightCurbTurningPoints] = useState<CurbTurningPoint[]>([]);
    const [rightCurbBezierPoints, setRightCurbBezierPoints] = useState<BezierPoints[]>([]);
    const [racingLinePoints, setRacingLinePoints] = useState<RacingLinePoint[]>([]);
    const [racingLineBezierPoints, setRacingLineBezierPoints] = useState<BezierPoints[]>([]);
    const [iterations, setIterations] = useState<number>(10);
    const [mapImage] = useImage(image);
    // Reference to parent container
    const containerRef = useRef<HTMLInputElement>(null);

    ///////////////functions////////////////////

    // Update on mount and when window resizes
    useEffect(() => {
        updateSize();
        createInitialShapes();
    }, []);

    //recalculate controlling position for bezier curve since the turning position changed
    useEffect(() => {
        calculateAndDrawTrack();
    }, [turningPoints]);


    //after turning points updated, we can update the racing line in the next frame
    useEffect(() => {
        calculateAndDrawRacingLine();

    }, [bezierPoints]);

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

    function createInitialShapes() {

        apiService.post('/racingmap/map/infolists', { name: mapEditorContext.mapSelected }).then((result) => {
            const data = result.data as MapInfo;
            setTurningPoints(data.points.map((point) => {
                return {
                    type: point.type,
                    index: point.index,
                    position: [point.position[0], point.position[1]],
                    description: "",
                    info: "",
                    isDoubleClicked: false
                };
            }));
        }).catch((e) => {
        });
    }

    function handleDragMove(e: any, id: any) {

        //use setTurningPoints, it triggers ui refresh
        setTurningPoints(turningPoints.map((turningPoint: {
            position: Point,
            type: number,
            index: number, //type and index are used together. some points are index sensitive
            description?: string,
            info?: string,
            variables?: [{ key: string, value: string }],
            isDoubleClicked: boolean
        }) => {
            if (turningPoint.index !== id) return turningPoint;

            //assign the targeted position
            let pointPosition: any[] = [e.target.x(), e.target.y()];

            //check boundary
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

            //set position. for some reason, position wont set again at boundary, this fix it temporarily
            e.target.absolutePosition({
                x: pointPosition[0],
                y: pointPosition[1]
            });

            return { ...turningPoint, position: [pointPosition[0], pointPosition[1]] }

        }));
    }

    const handleDragEnd = (e: any, id: any) => {

    };

    const handleDoubleClick = (e: any, id: any) => {
        //use setTurningPoints, it triggers ui refresh
        setTurningPoints(turningPoints.map(
            (turningPoint: {
                position: Point,
                type: number,
                index: number, //type and index are used together. some points are index sensitive
                description?: string,
                info?: string,
                variables?: [{ key: string, value: string }],
                isDoubleClicked: boolean
            }) => {
                if (turningPoint.index !== id) return turningPoint;

                return { ...turningPoint, isHovered: true }
            }
        ));
    };

    function calculateAndDrawRacingLine() {

        let tempRacingLinePoints: RacingLinePoint[] = [];
        let racingLineDisplacement = [];

        // Reset racing line
        //deep copy, regular copy will have the reference on the old array. cause mutation when changing the new array
        turningPoints.map((point) => {
            tempRacingLinePoints.push({
                id: point.index,
                position: [point.position[0], point.position[1]]
            });
        });
        racingLineDisplacement = new Array(tempRacingLinePoints.length).fill(0);

        for (let iteration = 0; iteration < iterations; iteration++) {
            for (let p = 0; p < tempRacingLinePoints.length; p++) {

                // Get locations of neighbour nodes
                const pointRight = tempRacingLinePoints[(p + 1) % tempRacingLinePoints.length];
                const pointLeft = tempRacingLinePoints[(p + tempRacingLinePoints.length - 1) % tempRacingLinePoints.length];
                const pointMiddle = tempRacingLinePoints[p];

                // Create vectors to neighbours
                const vectorLeft = [pointLeft.position[0] - pointMiddle.position[0], pointLeft.position[1] - pointMiddle.position[1]];
                const vectorRight = [pointRight.position[0] - pointMiddle.position[0], pointRight.position[1] - pointMiddle.position[1]];

                // Normalise neighbours
                const lengthLeft = Math.sqrt(vectorLeft[0] * vectorLeft[0] + vectorLeft[1] * vectorLeft[1]);
                const leftn = [vectorLeft[0] / lengthLeft, vectorLeft[1] / lengthLeft];
                const lengthRight = Math.sqrt(vectorRight[0] * vectorRight[0] + vectorRight[1] * vectorRight[1]);
                const rightn = [vectorRight[0] / lengthRight, vectorRight[1] / lengthRight];

                // Add together to create bisector vector
                const vectorSum = [rightn[0] + leftn[0], rightn[1] + leftn[1]];
                const len = Math.sqrt(vectorSum[0] * vectorSum[0] + vectorSum[1] * vectorSum[1]);
                vectorSum[0] = (len === 0) ? vectorSum[0] : vectorSum[0] / len;
                vectorSum[1] = (len === 0) ? vectorSum[1] : vectorSum[1] / len;

                // Get position gradient and normalise (TODO: ADD circular)
                const P0 = bezierPoints[(3 * p)].position;
                const P1 = bezierPoints[(3 * p + 1) % bezierPoints.length].position;
                const P2 = bezierPoints[(3 * p + 2) % bezierPoints.length].position;
                const P3 = bezierPoints[(3 * p + 3) % bezierPoints.length].position;

                const g = getBezierTangent(P0, P1, P2, P3, 0);

                const glen = Math.sqrt(g[0] * g[0] + g[1] * g[1]);
                g[0] = (glen === 0) ? g[0] : g[0] / glen;
                g[1] = (glen === 0) ? g[1] : g[1] / glen;

                // Project required correction onto position tangent to give displacment (projection)
                const dp = -g[1] * vectorSum[0] + g[0] * vectorSum[1];

                // Shortest path
                //racingLineDisplacement[p] += (dp * 4);

                // Curvature
                //fDisplacement[(i + 1) % racingLine.points.size()] += dp * -0.2f;
                //fDisplacement[(i - 1 + racingLine.points.size()) % racingLine.points.size()] += dp * -0.2f;

                racingLineDisplacement[(p + 1) % tempRacingLinePoints.length] += dp * -10;
                racingLineDisplacement[(p - 1 + tempRacingLinePoints.length) % tempRacingLinePoints.length] += dp * -10;
            }

            // Clamp displaced points to track width
            for (let p = 0; p < tempRacingLinePoints.length; p++) {
                {
                    if (racingLineDisplacement[p] >= 1) racingLineDisplacement[p] = 1;
                    if (racingLineDisplacement[p] <= -1) racingLineDisplacement[p] = -1;

                    const P0 = bezierPoints[(3 * p)].position;
                    const P1 = bezierPoints[(3 * p + 1) % bezierPoints.length].position;
                    const P2 = bezierPoints[(3 * p + 2) % bezierPoints.length].position;
                    const P3 = bezierPoints[(3 * p + 3) % bezierPoints.length].position;

                    const g = getBezierTangent(P0, P1, P2, P3, 0);

                    const glen = Math.sqrt(g[0] * g[0] + g[1] * g[1]);
                    g[0] /= glen; g[1] /= glen;

                    tempRacingLinePoints[p].position[0] = tempRacingLinePoints[p].position[0] - g[1] * racingLineDisplacement[p];
                    tempRacingLinePoints[p].position[1] = tempRacingLinePoints[p].position[1] - g[0] * racingLineDisplacement[p];
                }
            }

        }

        setRacingLinePoints(tempRacingLinePoints);

        //Add controlling points to turning points, also cached them together for later use
        try {
            let points = AddControlPoints(extractRacingLinePointToPoint(tempRacingLinePoints), 0.6);
            let index = 0;
            let result: BezierPoints[] = [];
            points.forEach((position) => {
                index++;
                result.push({ id: index, position: position });
            })
            setRacingLineBezierPoints(result);
        }
        catch (e) {
            return;
        }
    }

    /**
    * Appends a new point continuing the spline's end direction
    */
    function AddPointInDirection() {
        if (bezierPoints.length < 4) {
            return;
        }

        const points = extractBezierPointToPoint(bezierPoints);
        const lastPoint = points[points.length - 1];
        const direction = getEndDirection(points);
        const distance = 50
        // Normalize direction vector
        const dirLength = Math.sqrt(direction[0] ** 2 + direction[1] ** 2);
        const normalizedDir: Point = [
            direction[0] / dirLength,
            direction[1] / dirLength
        ];

        const newP = {
            position: [
                lastPoint[0] + normalizedDir[0] * distance,
                lastPoint[1] + normalizedDir[1] * distance
            ],
            type: 0,
            index: turningPoints.length //type and index are used together. some points are index sensitive
        } as RacingTurningPoint;
        // Return new array with added points (maintaining cubic Bézier requirements)
        return setTurningPoints([...turningPoints, newP]);
    }

    /**
     * using the 'turningPoints' and calculate the left and right curbs
     * @returns 
     */
    function calculateAndDrawTrack() {

        if (turningPoints.length === 0) return;

        //Add controlling points to turning points, also cached them together for later use
        let points = AddControlPoints(extractRacingTurningPointToPoint(turningPoints), 0.6)
        let index = 0;
        let result: BezierPoints[] = [];
        points.forEach((position) => {
            index++;
            result.push({ id: index, position: position });
        })

        setBezierPoints(result);
        setSegmentLengths(calculateSegmentLengths(extractBezierPointToPoint(result)))
        if (bezierPoints.length > 3) {

            //create left curb
            points = createRacingTurningPointOffset(bezierPoints, 'left')
            index = 0;
            result = [];
            for (let i = 0; i < points.length - 1; i += 3) {
                //we only want the turning points
                const P0 = points[i];
                const P3 = points[i + 3];
                index++;
                // Only push Q0 if it's the first segment to avoid duplicates
                if (i === 0) result.push({ id: 0, position: P0 });
                result.push({ id: index, position: P3 })
            }
            setLeftCurbTurningPoints(result);

            //add controll points for left curb Bezier 
            points = AddControlPoints(extractCurbTurningPointToPoint(result), 0.4);
            index = 0;
            result = [];
            points.forEach((position) => {
                index++;
                result.push({ id: index, position: position });
            })
            setLeftCrubBezierPoints(result);

            //create right curb
            points = createRacingTurningPointOffset(bezierPoints, 'right');
            index = 0;
            result = [];
            //we only need the turning position
            for (let i = 0; i < points.length - 1; i += 3) {
                //we only want the turning points
                const P0 = points[i];
                const P3 = points[i + 3];
                index++;
                // Only push Q0 if it's the first segment to avoid duplicates
                if (i === 0) result.push({ id: 0, position: P0 });
                result.push({ id: index, position: P3 })
            }
            setRightCurbTurningPoints(result);

            //add controll points for right curb Bezier 
            points = AddControlPoints(extractCurbTurningPointToPoint(result), 0.4);
            index = 0;
            result = [];
            points.forEach((position) => {
                index++;
                result.push({ id: index, position: position });
            })
            setRightCurbBezierPoints(result);
        }
    }

    function deleteTurningPoint(index: number) {
        if (turningPoints.length <= 4) return;
        setTurningPoints(prevState => {
            return prevState.filter(task => task.index !== index)
        })
    }

    return (

        <div ref={containerRef} style={{ width: '100%', height: '90%' }}>
            <ContextMenu.Root>
                <ContextMenu.Trigger>
                    <div>
                        <Stage width={stageSize.width} height={stageSize.height} >
                            <Layer>

                                {/* 
                                <Image
                                    x={0}
                                    y={0}
                                    image={mapImage}
                                    scaleX={3}
                                    scaleY={3}
                                />
                                 */}

                                <Line
                                    points={exportPointsForDrawing(extractBezierPointToPoint(bezierPoints))}
                                    stroke="red" strokeWidth={25} bezier={true}
                                />
                                {

                                    turningPoints.map((
                                        turningPoint: {
                                            position: Point,
                                            type: number,
                                            index: number, //type and index are used together. some points are index sensitive
                                            description?: string,
                                            info?: string,
                                            variables?: [{ key: string, value: string }],
                                            isDoubleClicked: boolean
                                        }) => (

                                        <Group
                                            key={turningPoint.index} id={`group-${turningPoint.index}`}
                                            x={turningPoint.position[0]} y={turningPoint.position[1]}
                                            draggable
                                            onDragMove={(e) => handleDragMove(e, turningPoint.index)}
                                            onDragEnd={(e) => handleDragEnd(e, turningPoint.index)}
                                            onDoubleClick={(e: any) => handleDoubleClick(e, turningPoint.index)}>
                                            <Circle key={turningPoint.index} radius={10} fill={"green"} name={turningPoint.index.toString()} />
                                            <Html>
                                                <DropdownMenu.Root open={turningPoint.isDoubleClicked}>
                                                    <DropdownMenu.Content>
                                                        <DropdownMenu.Separator />
                                                        <DropdownMenu.Item color="red" onSelect={() => deleteTurningPoint(turningPoint.index)}>
                                                            Delete
                                                        </DropdownMenu.Item>
                                                    </DropdownMenu.Content>
                                                </DropdownMenu.Root>
                                            </Html>
                                        </Group>
                                    ))}
                            </Layer>
                        </Stage>
                    </div>
                </ContextMenu.Trigger>
                <ContextMenu.Content size="1">
                    <ContextMenu.Item onClick={AddPointInDirection}>Add a new turning point</ContextMenu.Item>
                </ContextMenu.Content>
            </ContextMenu.Root>
        </div>
    );


};

function convert_1D_array_to_2d_array(points: number[]): Point[] {
    const result: Point[] = [];
    for (let i = 0; i < points.length; i += 2) {
        const position = points.slice(i, i + 2);
        result.push([position[0], position[1]]);
    }
    return result;
}

function extractRacingTurningPointToPoint(points: RacingTurningPoint[]): Point[] {
    return points.reduce((acc, curr): Point[] => {
        return [...acc, curr.position];
    }, [] as Point[]);
}

function extractRacingLinePointToPoint(points: RacingLinePoint[]): Point[] {
    return points.reduce((acc, curr): Point[] => {
        return [...acc, curr.position];
    }, [] as Point[]);
}

function extractCurbTurningPointToPoint(points: CurbTurningPoint[]): Point[] {
    return points.reduce((acc, curr): Point[] => {
        return [...acc, curr.position];
    }, [] as Point[]);
}

function extractBezierPointToPoint(points: BezierPoints[]): Point[] {
    if (!points) return [];
    return points.reduce((acc, curr): Point[] => {
        return [...acc, curr.position];
    }, [] as Point[]);
}

/**
 * flat position[] to 1d number[]
 * @param points 
 * @returns 
 */
function convert_Points_to_1d_array(points: Point[]): number[] {
    if (points.length === 0) return [];
    return points.flat();;
}

/**
 * give points of turning and the control points come with it, return offseted curbs position and the control points using Bezier 
 * @param points 
 * @param direction 
 * @returns 
 */
function createRacingTurningPointOffset(points?: BezierPoints[], direction: 'left' | 'right' = 'left'): Point[] {
    if (!points || points.length === 0) return [];
    return offsetBezierPoints(extractBezierPointToPoint(points), 10, direction);
}
/**
 * input points which should contains controlling points, and convert them into Curves and convert the result into 1d array readable by the Line component
 * @param bezierPoints 
 * @returns 1d array for Konva Line component
 */
function exportPointsForDrawing(points?: Point[]): number[] {
    if (!points) return [];
    //-> smooth the points to more points -> convert into 1d array
    return convert_Points_to_1d_array(points);
}

export default MapEditor;

