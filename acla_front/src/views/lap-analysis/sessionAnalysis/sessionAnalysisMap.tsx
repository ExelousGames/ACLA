import { useContext, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Line } from 'react-konva';
import { AddControlPoints } from 'utils/curve-tobezier/curve-to-bezier';
import { offsetBezierPoints, Point, getBezierTangent, getEndDirection, pointOnCubicBezierSpline, calculateSegmentLengths } from 'utils/curve-tobezier/points-on-curve';
import apiService from 'services/api.service';
import { MapInfo, RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { AnalysisContext } from '../session-analysis';
import LiveAnalysisSessionRecording from '../liveAnalysisSessionRecording';
import { useEnvironment } from 'contexts/EnvironmentContext';
import { IconButton } from '@radix-ui/themes';
import { ZoomInIcon, ZoomOutIcon } from '@radix-ui/react-icons';

type RacingTurningPoint = {
    position: Point,
    type: number,
    index: number, //type and index are used together. some points are index sensitive
    description?: string,
    info?: string,
    variables?: [{ key: string, value: string }]
};
type CurbTurningPoint = { id: number, position: Point };
type BezierPoints = { id: number, position: Point };
type RacingLinePoint = { id: number, position: Point };
const SessionAnalysisMap = () => {

    // Define virtual size for our scene
    let containerWidth = window.innerWidth;
    let containerHeight = window.innerWidth;
    // State to track current scale and dimensions
    const [stageSize, setStageSize] = useState({
        width: containerWidth,
        height: containerHeight,
    });

    const analysisContext = useContext(AnalysisContext);

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

    // Camera/viewport control state
    const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
    const [stageScale, setStageScale] = useState(1);
    const stageRef = useRef<any>(null);

    // Reference to parent container
    const containerRef = useRef<HTMLInputElement>(null);

    ///////////////functions////////////////////

    // Update on mount and when window resizes
    useEffect(() => {
        updateSize();
        createInitialShapes();

        // Add resize listener
        const handleResize = () => {
            updateSize();
        };

        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    //recalculate controlling position for bezier curve since the turning position changed
    useEffect(() => {
        calculateAndDrawTrack();
    }, [turningPoints]);

    // Center the track when bezier points are first calculated
    useEffect(() => {
        if (bezierPoints.length > 3) {
            centerTrack();
        }
    }, [bezierPoints, stageSize]);

    // Function to center the track in the viewport
    const centerTrack = () => {
        if (bezierPoints.length === 0 || !stageRef.current) return;

        // Calculate bounding box of all track points
        const points = extractBezierPointToPoint(bezierPoints);
        if (points.length === 0) return;

        let minX = points[0][0];
        let maxX = points[0][0];
        let minY = points[0][1];
        let maxY = points[0][1];

        points.forEach(point => {
            minX = Math.min(minX, point[0]);
            maxX = Math.max(maxX, point[0]);
            minY = Math.min(minY, point[1]);
            maxY = Math.max(maxY, point[1]);
        });

        // Calculate track center and dimensions
        const trackCenterX = (minX + maxX) / 2;
        const trackCenterY = (minY + maxY) / 2;
        const trackWidth = maxX - minX;
        const trackHeight = maxY - minY;

        // Calculate scale to fit track in viewport with some padding
        const padding = 0.1; // 10% padding
        const availableWidth = stageSize.width * (1 - padding);
        const availableHeight = stageSize.height * (1 - padding);
        const scaleX = availableWidth / trackWidth;
        const scaleY = availableHeight / trackHeight;
        const scale = Math.min(scaleX, scaleY, 2); // Max scale of 2x

        // Calculate position to center the track
        const viewportCenterX = stageSize.width / 2;
        const viewportCenterY = stageSize.height / 2;
        const offsetX = viewportCenterX - (trackCenterX * scale);
        const offsetY = viewportCenterY - (trackCenterY * scale);

        setStageScale(scale);
        setStagePos({ x: offsetX, y: offsetY });
    };

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

    // Zoom and camera control functions
    const handleZoomIn = () => {
        const newScale = Math.min(stageScale * 1.2, 3); // Max zoom 3x
        setStageScale(newScale);
    };

    const handleZoomOut = () => {
        const newScale = Math.max(stageScale / 1.2, 0.1); // Min zoom 0.1x
        setStageScale(newScale);
    };

    const handleStageDrag = (e: any) => {
        setStagePos({
            x: e.target.x(),
            y: e.target.y(),
        });
    };

    const handleWheel = (e: any) => {
        e.evt.preventDefault();

        const scaleBy = 1.05;
        const stage = e.target.getStage();
        const oldScale = stage.scaleX();
        const mousePointTo = {
            x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
            y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale,
        };

        const newScale = e.evt.deltaY > 0 ? oldScale / scaleBy : oldScale * scaleBy;
        const clampedScale = Math.max(0.1, Math.min(3, newScale));

        setStageScale(clampedScale);
        setStagePos({
            x: -(mousePointTo.x - stage.getPointerPosition().x / clampedScale) * clampedScale,
            y: -(mousePointTo.y - stage.getPointerPosition().y / clampedScale) * clampedScale,
        });
    }

    function createInitialShapes() {

        apiService.post('/racingmap/map/infolists', { name: analysisContext.mapSelected }).then((result) => {
            const data = result.data as MapInfo;
            setTurningPoints(data.points.map((point) => {
                return {
                    type: point.type,
                    index: point.index,
                    position: [point.position[0], point.position[1]],
                    description: "",
                    info: "",
                };
            }));
        }).catch((e) => {
        });

        if (analysisContext.sessionSelected?.SessionId) {
            apiService.post('/racing-session/detailedSessionInfo', { id: analysisContext.sessionSelected?.SessionId }).then((result) => {
                const data = result.data as RacingSessionDetailedInfoDto;
                analysisContext.setSession(data);
            }).catch((e) => {
            });
        }


    }

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

    return (
        <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
            {/* Zoom Controls */}
            <div style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                zIndex: 1000,
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
            }}>
                <IconButton
                    onClick={handleZoomIn}
                    size="3"
                    variant="outline"
                    style={{
                        backgroundColor: 'white',
                        cursor: 'pointer'
                    }}
                    title="Zoom In"
                >
                    <ZoomInIcon />
                </IconButton>
                <IconButton
                    onClick={handleZoomOut}
                    size="3"
                    variant="outline"
                    style={{
                        backgroundColor: 'white',
                        cursor: 'pointer'
                    }}
                    title="Zoom Out"
                >
                    <ZoomOutIcon />
                </IconButton>
            </div>

            <Stage
                width={stageSize.width}
                height={stageSize.height}
                ref={stageRef}
                scaleX={stageScale}
                scaleY={stageScale}
                x={stagePos.x}
                y={stagePos.y}
                draggable={true}
                onDragEnd={handleStageDrag}
                onWheel={handleWheel}
            >
                <Layer>
                    {/* Racing Track Lines */}
                    {bezierPoints.length > 3 && (
                        <>
                            {/* Track Shadow/Border for depth */}
                            <Line
                                points={exportPointsForDrawing(extractBezierPointToPoint(bezierPoints))}
                                stroke="#1a1a1a"
                                strokeWidth={32}
                                bezier={true}
                                lineCap="round"
                                lineJoin="round"
                                shadowColor="#000000"
                                shadowBlur={8}
                                shadowOffset={{ x: 2, y: 2 }}
                                shadowOpacity={0.3}
                            />

                            {/* Main Track Surface */}
                            <Line
                                points={exportPointsForDrawing(extractBezierPointToPoint(bezierPoints))}
                                stroke="#404040"
                                strokeWidth={28}
                                bezier={true}
                                lineCap="round"
                                lineJoin="round"
                            />

                            {/* Center Line Dashes */}
                            <Line
                                points={exportPointsForDrawing(extractBezierPointToPoint(bezierPoints))}
                                stroke="#ffffff"
                                strokeWidth={2}
                                bezier={true}
                                lineCap="round"
                                lineJoin="round"
                                dash={[8, 12]}
                                opacity={0.8}
                            />
                        </>
                    )}
                </Layer>
            </Stage>
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

export default SessionAnalysisMap;

