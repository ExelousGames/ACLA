import { createContext, Key, useContext, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Arc, Circle, Rect, Line, Group, Image } from 'react-konva';
import { AddControlPoints } from 'utils/curve-tobezier/curve-to-bezier';
import { offsetBezierPoints, Point, ConstructAllPointsOnBezierCurves, getBezierTangent } from 'utils/curve-tobezier/points-on-curve';
import useImage from 'use-image';
import image from 'assets/map2.png'
import myData from 'data/sessionAnalysis.json';
type RacingTurningPoint = { id: number, point: Point, l_width: number, r_width: number };
type CurbTurningPoint = { id: number, point: Point };
type BezierPoints = { id: number, point: Point };
type RacingLinePoint = { id: number, point: Point };
const SessionAnalysis = () => {

    // Define virtual size for our scene
    let containerWidth = window.innerWidth;
    let containerHeight = window.innerWidth;
    // State to track current scale and dimensions
    const [stageSize, setStageSize] = useState({
        width: containerWidth,
        height: containerHeight,
    });
    const [turningPoints, setTurningPoints] = useState<RacingTurningPoint[]>([]);
    const [bezierPoints, setBezierPoints] = useState<BezierPoints[]>([]);
    const [leftCurbTurningPoints, setLeftCurbTurningPoints] = useState<CurbTurningPoint[]>([]);
    const [leftCurbBezierPoints, setLeftCrubBezierPoints] = useState<BezierPoints[]>([]);
    const [rightCurbTurningPoints, setRightCurbTurningPoints] = useState<CurbTurningPoint[]>([]);
    const [rightCurbBezierPoints, setRightCurbBezierPoints] = useState<BezierPoints[]>([]);
    const [racingLinePoints, setRacingLinePoints] = useState<RacingLinePoint[]>([]);
    const [racingLineBezierPoints, setRacingLineBezierPoints] = useState<BezierPoints[]>([]);
    const [iterations, setIterations] = useState<number>(10);
    const [yodaImage] = useImage(image);
    // Reference to parent container
    const containerRef = useRef<HTMLInputElement>(null);

    ///////////////functions////////////////////

    function createInitialShapes(): RacingTurningPoint[] {
        let points: RacingTurningPoint[] = [];
        myData.map((point) => {
            points.push({
                id: point.id,
                point: [point.point[0], point.point[1]],
                l_width: point.l_width,
                r_width: point.r_width
            })
        })

        return points;

        // return [
        //     {
        //         id: 0, point: [0, 0],
        //         l_width: 5,
        //         r_width: 5
        //     },
        //     {
        //         id: 1, point: [0, 20],
        //         l_width: 5,
        //         r_width: 5
        //     },
        //     {
        //         id: 2, point: [0, 40],
        //         l_width: 5,
        //         r_width: 5
        //     },
        //     {
        //         id: 3, point: [0, 120],
        //         l_width: 5,
        //         r_width: 5
        //     },
        //     {
        //         id: 4, point: [0, 160],
        //         l_width: 5,
        //         r_width: 5
        //     },
        //     {
        //         id: 5, point: [0, 180],
        //         l_width: 5,
        //         r_width: 5
        //     },
        // ]
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
        setTurningPoints(createInitialShapes());
    }, []);

    //recalculate controlling point for bezier curve since the turning point moved
    useEffect(() => {
        calculateTrack();
    }, [turningPoints]);


    function handleDragMove(e: any, id: any) {
        const target = e.target;

        setTurningPoints(turningPoints.map((turningPoint: { id: number, point: Point, l_width: number, r_width: number }) => {
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

            //set position. for some reason, point wont set again at boundary, this fix it temporarily
            e.target.absolutePosition({
                x: pointPosition[0],
                y: pointPosition[1]
            });

            return { ...turningPoint, point: [pointPosition[0], pointPosition[1]] }

        }));
        console.log(turningPoints);

    }


    const handleDragEnd = (e: any, id: any) => {

    };

    function calculateRacingLine() {

        let tempRacingLinePoints: RacingLinePoint[] = [];
        let racingLineDisplacement = [];

        // Reset racing line
        //deep copy, regular copy will have the reference on the old array. cause mutation when changing the new array
        turningPoints.map((point) => {
            tempRacingLinePoints.push({
                id: point.id,
                point: [point.point[0], point.point[1]]
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
                const vectorLeft = [pointLeft.point[0] - pointMiddle.point[0], pointLeft.point[1] - pointMiddle.point[1]];
                const vectorRight = [pointRight.point[0] - pointMiddle.point[0], pointRight.point[1] - pointMiddle.point[1]];

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

                // Get point gradient and normalise (TODO: ADD circular)
                const P0 = bezierPoints[(3 * p)].point;
                const P1 = bezierPoints[(3 * p + 1) % bezierPoints.length].point;
                const P2 = bezierPoints[(3 * p + 2) % bezierPoints.length].point;
                const P3 = bezierPoints[(3 * p + 3) % bezierPoints.length].point;

                const g = getBezierTangent(P0, P1, P2, P3, 0);

                const glen = Math.sqrt(g[0] * g[0] + g[1] * g[1]);
                g[0] = (glen === 0) ? g[0] : g[0] / glen;
                g[1] = (glen === 0) ? g[1] : g[1] / glen;

                // Project required correction onto point tangent to give displacment (projection)
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

                    const P0 = bezierPoints[(3 * p)].point;
                    const P1 = bezierPoints[(3 * p + 1) % bezierPoints.length].point;
                    const P2 = bezierPoints[(3 * p + 2) % bezierPoints.length].point;
                    const P3 = bezierPoints[(3 * p + 3) % bezierPoints.length].point;

                    const g = getBezierTangent(P0, P1, P2, P3, 0);

                    const glen = Math.sqrt(g[0] * g[0] + g[1] * g[1]);
                    g[0] /= glen; g[1] /= glen;

                    tempRacingLinePoints[p].point[0] = tempRacingLinePoints[p].point[0] - g[1] * racingLineDisplacement[p];
                    tempRacingLinePoints[p].point[1] = tempRacingLinePoints[p].point[1] - g[0] * racingLineDisplacement[p];
                }
            }

        }

        setRacingLinePoints(tempRacingLinePoints);

        //Add controlling points to turning points, also cached them together for later use
        try {
            let points = AddControlPoints(extractRacingLinePointToPoint(tempRacingLinePoints), 0.6);
            let index = 0;
            let result: BezierPoints[] = [];
            points.forEach((point) => {
                index++;
                result.push({ id: index, point: point });
            })
            setRacingLineBezierPoints(result);
        }
        catch (e) {
            return;
        }
    }
    function calculateTrack() {

        if (turningPoints.length === 0) return;

        //Add controlling points to turning points, also cached them together for later use
        let points = AddControlPoints(extractRacingTurningPointToPoint(turningPoints), 0.6)
        let index = 0;
        let result: BezierPoints[] = [];
        points.forEach((point) => {
            index++;
            result.push({ id: index, point: point });
        })
        setBezierPoints(result);


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
                if (i === 0) result.push({ id: 0, point: P0 });
                result.push({ id: index, point: P3 })
            }
            setLeftCurbTurningPoints(result);

            //add controll points for left curb Bezier 
            points = AddControlPoints(extractCurbTurningPointToPoint(result), 0.4);
            index = 0;
            result = [];
            points.forEach((point) => {
                index++;
                result.push({ id: index, point: point });
            })
            setLeftCrubBezierPoints(result);

            //create right curb
            points = createRacingTurningPointOffset(bezierPoints, 'right');
            index = 0;
            result = [];
            //we only need the turning point
            for (let i = 0; i < points.length - 1; i += 3) {
                //we only want the turning points
                const P0 = points[i];
                const P3 = points[i + 3];
                index++;
                // Only push Q0 if it's the first segment to avoid duplicates
                if (i === 0) result.push({ id: 0, point: P0 });
                result.push({ id: index, point: P3 })
            }
            setRightCurbTurningPoints(result);

            //add controll points for right curb Bezier 
            points = AddControlPoints(extractCurbTurningPointToPoint(result), 0.4);
            index = 0;
            result = [];
            points.forEach((point) => {
                index++;
                result.push({ id: index, point: point });
            })
            setRightCurbBezierPoints(result);

            calculateRacingLine();

        }
    }

    return (
        <div ref={containerRef} style={{ width: '100%', height: '90%' }}>

            <Stage width={stageSize.width} height={stageSize.height} >
                <Layer>
                    <Image
                        x={0}
                        y={0}
                        image={yodaImage}
                        scaleX={3}
                        scaleY={3}

                    />
                    <Line
                        points={exportPointsForDrawing(extractBezierPointToPoint(leftCurbBezierPoints))}
                        stroke="red" strokeWidth={2} bezier={true}
                    />
                    {/* 
                    <Line
                        points={exportPointsForDrawing(extractBezierPointToPoint(bezierPoints))}
                        stroke="red" strokeWidth={2} bezier={true}
                    />
                    */}
                    <Line
                        points={exportPointsForDrawing(extractBezierPointToPoint(rightCurbBezierPoints))}
                        stroke="red" strokeWidth={2} bezier={true}
                    />

                    {turningPoints.map((turningPoint: { id: Key, point: Point }) => (
                        <Group
                            key={turningPoint.id} id={`group-${turningPoint.id}`} x={turningPoint.point[0]} y={turningPoint.point[1]} draggable
                            onDragMove={(e) => handleDragMove(e, turningPoint.id)}
                            onDragEnd={(e) => handleDragEnd(e, turningPoint.id)}>
                            <Circle
                                key={turningPoint.id} radius={10} fill={"red"} name={turningPoint.id.toString()}
                            />
                        </Group>
                    ))}

                    <Line
                        points={exportPointsForDrawing(extractBezierPointToPoint(racingLineBezierPoints))}
                        stroke="green" strokeWidth={2} bezier={true}
                    />
                    {/* 
                    {racingLinePoints.map((point: { id: Key, point: Point }) => (
                        <Circle
                            key={point.id} x={point.point[0]} y={point.point[1]} radius={10} fill={"blue"} name={point.id.toString()}
                        />
                    ))}
                    */}
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

function extractRacingLinePointToPoint(points: RacingLinePoint[]): Point[] {
    return points.reduce((acc, curr): Point[] => {
        return [...acc, curr.point];
    }, [] as Point[]);
}

function extractCurbTurningPointToPoint(points: CurbTurningPoint[]): Point[] {
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
    return points.flat();;
}

/**
 * give points of turning and the control points come with it, return offseted curbs point and the control points using Bezier 
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

export default SessionAnalysis;

