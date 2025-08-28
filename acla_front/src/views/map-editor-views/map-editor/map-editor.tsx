import { useContext, useEffect, useRef, useState } from 'react';
import { Stage, Layer, Circle, Line, Group, Image } from 'react-konva';
import { AddControlPoints } from 'utils/curve-tobezier/curve-to-bezier';
import { offsetBezierPoints, Point, getBezierTangent, getEndDirection, pointOnCubicBezierSpline, calculateSegmentLengths } from 'utils/curve-tobezier/points-on-curve';
import useImage from 'use-image';
import image from 'assets/map2.png'
import apiService from 'services/api.service';
import { MapInfo, RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { useEnvironment } from 'contexts/EnvironmentContext';
import { ContextMenu, IconButton, Dialog, Button, TextField, Text, Flex, Select } from '@radix-ui/themes';
import { Html } from 'react-konva-utils';
import { HamburgerMenuIcon, PlusIcon, ZoomInIcon, ZoomOutIcon, Pencil1Icon, TrashIcon, DotFilledIcon, TriangleRightIcon, CursorArrowIcon, PlayIcon, StopIcon } from '@radix-ui/react-icons';
import { MapEditorContext } from '../map-editor-view';
import { DropdownMenu, HoverCard } from 'radix-ui';
import SettingsMenu from './components/SettingsMenu';
import "./map-editor.css";
type RacingTurningPoint = {
    position: Point,
    type: number,
    index: number, //type and index are used together. some points are index sensitive
    description?: string,
    info?: string,

    //variables will be saved online
    variables?: [{ key: string, value: string }],

    //variables will not be saved
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
    const [uploadedMapImage, setUploadedMapImage] = useState<HTMLImageElement | null>(null);

    // Camera/viewport control state
    const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
    const [stageScale, setStageScale] = useState(1);
    const stageRef = useRef<any>(null);
    const [isDraggingPoint, setIsDraggingPoint] = useState(false);

    // trigger useffect when mouse move, used to detect mouse movement direction
    const [mouseMovement, setMouseMovement] = useState({ x: 0, y: 0 });

    //record mouse coord, used for calculating mouse movement direction. we must store current and previous data at the same time
    const currCoords = useRef({ x: 0, y: 0 });

    // reference to previous mouse coords, useRef is used to persist value without causing re-render
    const prvCoords = useRef({ x: 0, y: 0 });

    // Reference to the menu element
    const menuRef = useRef<HTMLDivElement | null>(null);

    // Track which menu is active (by point index), null if none
    const [activeMenu, setActiveMenu] = useState<number | null>(null);

    // Timeout reference for delaying menu close
    const timeoutRef = useRef<NodeJS.Timeout | null>(null);

    // Reference to parent container
    const containerRef = useRef<HTMLInputElement>(null);

    // Point editing state
    const [editingPoint, setEditingPoint] = useState<RacingTurningPoint | null>(null);
    const [editDialogOpen, setEditDialogOpen] = useState(false);
    const [editFormData, setEditFormData] = useState({
        type: 0,
        description: '',
        info: '',
        variables: [] as { key: string, value: string }[]
    });

    ///////////////functions////////////////////

    // Update on mount and when window resizes, track mouse movement
    useEffect(() => {
        updateSize();
        createInitialShapes();
        const handleMouseMove = (e: any) => {
            prvCoords.current = { x: currCoords.current.x, y: currCoords.current.y };
            currCoords.current = { x: e.clientX, y: e.clientY };

            //trigger a re-render
            setMouseMovement({ x: e.clientX, y: e.clientY });

        };
        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, []);

    //recalculate controlling position for bezier curve since the turning position changed
    useEffect(() => {
        calculateAndDrawTrack();
    }, [turningPoints]);

    // Check if mouse is moving toward menu
    useEffect(() => {
        if (activeMenu === null || !menuRef.current) return;

        const menuRect = menuRef.current.getBoundingClientRect();
        // Calculate mouse movement vector  
        const dx = currCoords.current.x - prvCoords.current.x;
        const dy = currCoords.current.y - prvCoords.current.y;

        // Calculate vector from previous mouse position to menu center
        const menuCenter = {
            x: menuRect.x + menuRect.width / 2,
            y: menuRect.y + menuRect.height / 2,
        };
        const toMenuX = menuCenter.x - prvCoords.current.x;
        const toMenuY = menuCenter.y - prvCoords.current.y;

        // Calculate dot product to check if mouse is moving toward menu
        const dot = dx * toMenuX + dy * toMenuY;
        const isMovingTowardMenu = dot >= 0;

        if (isMovingTowardMenu) {
            // If moving toward menu, give more time before closing
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
                timeoutRef.current = setTimeout(() => {
                    setActiveMenu(null);
                }, 500); // Reduced delay for better UX
            }
        }
    }, [mouseMovement, activeMenu]);


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
        // Only update stage position if we're not dragging a point
        if (!isDraggingPoint) {
            setStagePos({
                x: e.target.x(),
                y: e.target.y(),
            });
        } else {
            // If we're dragging a point, prevent stage movement by resetting position
            e.target.position({ x: stagePos.x, y: stagePos.y });
        }
    };

    const handleStageDoubleClick = (e: any) => {
        // Failsafe: double-click on empty space always enables stage dragging
        setIsDraggingPoint(false);
    };

    const handleStageMouseDown = (e: any) => {
        // Check if we clicked on interactive elements (points) vs draggable areas (background, empty space)
        const targetClassName = e.target.getClassName();
        const targetName = e.target.name ? e.target.name() : '';

        // Interactive elements that should NOT allow canvas dragging
        const isInteractiveElement = targetClassName === 'Group' ||
            targetClassName === 'Circle' ||
            (targetName !== '' && targetName !== undefined);

        if (!isInteractiveElement) {
            // We clicked on draggable area (empty space, background image, or racing line), allow stage dragging
            setIsDraggingPoint(false);
        } else {
            // We clicked on an interactive element (point), disable stage dragging
            setIsDraggingPoint(true);
        }
    };

    const handleStageMouseUp = (e: any) => {
        // Always reset isDraggingPoint on mouse up if we clicked on draggable areas
        const targetClassName = e.target.getClassName();
        const targetName = e.target.name ? e.target.name() : '';

        // Interactive elements that should NOT allow canvas dragging
        const isInteractiveElement = targetClassName === 'Group' ||
            targetClassName === 'Circle' ||
            (targetName !== '' && targetName !== undefined);

        if (!isInteractiveElement) {
            setIsDraggingPoint(false);
        }
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
    };

    function createInitialShapes() {

        apiService.post('/racingmap/map/infolists', { name: mapEditorContext.mapSelected }).then((result) => {
            const data = result.data as MapInfo;

            // Check if we have at least 4 points
            if (!data.points || (data.points as any).length < 4) {
                // Create 4 points forming a circle, ensuring they stay within canvas bounds
                const padding = 50; // Minimum distance from canvas edges
                const centerX = stageSize.width / 2;
                const centerY = stageSize.height / 2;
                const maxRadius = Math.min(stageSize.width, stageSize.height) / 2 - padding;
                const radius = Math.min(maxRadius, Math.min(stageSize.width, stageSize.height) / 4);

                const circlePoints: RacingTurningPoint[] = [
                    {
                        type: 0,
                        index: 0,
                        position: [centerX, centerY - radius], // Top
                        description: "",
                        info: "",
                        variables: undefined
                    },
                    {
                        type: 0,
                        index: 1,
                        position: [centerX + radius, centerY], // Right
                        description: "",
                        info: "",
                        variables: undefined
                    },
                    {
                        type: 0,
                        index: 2,
                        position: [centerX, centerY + radius], // Bottom
                        description: "",
                        info: "",
                        variables: undefined
                    },
                    {
                        type: 0,
                        index: 3,
                        position: [centerX - radius, centerY], // Left
                        description: "",
                        info: "",
                        variables: undefined
                    }
                ];

                setTurningPoints(circlePoints);
            } else {
                setTurningPoints(data.points.map((point) => {
                    return {
                        type: point.type,
                        index: point.index,
                        position: [point.position[0], point.position[1]],
                        description: point.description || "",
                        info: point.info || "",
                        variables: point.variables as [{ key: string, value: string }] | undefined
                    };
                }));
            }

            // Load uploaded map image if available
            loadUploadedMapImage();
        }).catch((e) => {
        });
    }

    function loadUploadedMapImage() {
        apiService.post('/racingmap/map/image', { name: mapEditorContext.mapSelected }).then((result) => {
            const data = result.data as { imageData: string; mimetype: string };
            if (data && data.imageData && data.mimetype) {
                // Create image from base64 data
                const imageSrc = `data:${data.mimetype};base64,${data.imageData}`;

                const img = new window.Image();
                img.onload = () => {
                    setUploadedMapImage(img);
                };
                img.src = imageSrc;
            }
        }).catch((e) => {
            console.error('Failed to load uploaded image:', e);
        });
    }

    const handleImageUploaded = () => {
        // Reload the uploaded image after successful upload
        loadUploadedMapImage();
    };

    function handleDragMove(e: any, id: any) {
        // Prevent event bubbling to stage during point dragging
        e.cancelBubble = true;

        // Let Konva handle the visual dragging naturally
        // We'll update the state only on drag end for better performance
    }

    const handleDragEnd = (e: any, id: any) => {
        // Prevent event bubbling to stage
        e.cancelBubble = true;

        // Update the state with the final position after drag ends
        const newX = e.target.x();
        const newY = e.target.y();

        setTurningPoints(turningPoints.map((turningPoint) => {
            if (turningPoint.index !== id) return turningPoint;

            // Apply boundary constraints
            let constrainedX = newX;
            let constrainedY = newY;

            if (constrainedX < 0) constrainedX = 0;
            if (constrainedX > stageSize.width) constrainedX = stageSize.width;
            if (constrainedY < 0) constrainedY = 0;
            if (constrainedY > stageSize.height) constrainedY = stageSize.height;

            // Update the target position if it was constrained
            if (constrainedX !== newX || constrainedY !== newY) {
                e.target.position({ x: constrainedX, y: constrainedY });
            }

            return { ...turningPoint, position: [constrainedX, constrainedY] };
        }));

        // Re-enable stage dragging when point drag ends
        setIsDraggingPoint(false);
    };

    const handleDragStart = (e: any, id: any) => {
        // Prevent event bubbling to stage
        e.cancelBubble = true;

        // Disable stage dragging when point drag starts
        setIsDraggingPoint(true);
    };

    const handleEnterMenu = (id: any) => {
        // Clear any existing timeout to close menu
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
        }
        setActiveMenu(id);
    };

    const handleLeaveMenu = (id: any) => {
        // Start a timeout to close the menu after a short delay
        timeoutRef.current = setTimeout(() => {
            setActiveMenu(null);
        }, 300); // Reduced delay for better UX
    };

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

    function openEditDialog(point: RacingTurningPoint) {
        setEditingPoint(point);
        setEditFormData({
            type: point.type,
            description: point.description || '',
            info: point.info || '',
            variables: point.variables || []
        });
        setEditDialogOpen(true);
    }

    function savePointEdits() {
        if (!editingPoint) return;

        setTurningPoints(prevState =>
            prevState.map(point =>
                point.index === editingPoint.index
                    ? {
                        ...point,
                        type: editFormData.type,
                        description: editFormData.description,
                        info: editFormData.info,
                        variables: editFormData.variables as [{ key: string, value: string }]
                    }
                    : point
            )
        );

        setEditDialogOpen(false);
        setEditingPoint(null);
    }

    function addVariable() {
        setEditFormData(prev => ({
            ...prev,
            variables: [...prev.variables, { key: '', value: '' }]
        }));
    }

    function updateVariable(index: number, field: 'key' | 'value', value: string) {
        setEditFormData(prev => ({
            ...prev,
            variables: prev.variables.map((variable, i) =>
                i === index ? { ...variable, [field]: value } : variable
            )
        }));
    }

    function removeVariable(index: number) {
        setEditFormData(prev => ({
            ...prev,
            variables: prev.variables.filter((_, i) => i !== index)
        }));
    }

    // Function to get point styling based on type
    function getPointStyling(type: number) {
        switch (type) {
            case 0: // Default
                return {
                    fill: '#00ff0dff', // Amber
                    icon: DotFilledIcon,
                    label: 'Standard Turn'
                };
            case 1: // Corner Start
                return {
                    fill: '#f31c1cff', // Red
                    icon: TriangleRightIcon,
                    label: 'Corner Start'
                };
            case 2: // Cornering
                return {
                    fill: '#e2f10aff', // Yellowish green
                    icon: CursorArrowIcon,
                    label: 'Cornering'
                };
            case 3: // Corner End
                return {
                    fill: '#9ce22bff', // Purple
                    icon: PlayIcon,
                    label: 'Corner End'
                };
            case 4: // Start/Finish
                return {
                    fill: '#3b12f3ff', // Dark Red
                    icon: StopIcon,
                    label: 'Start/Finish'
                };
            default:
                return {
                    fill: '#6b7280', // Gray
                    icon: DotFilledIcon,
                    label: 'Unknown'
                };
        }
    }

    return (

        <div ref={containerRef} style={{ width: '100%', height: '90%', position: 'relative' }}>
            {mapEditorContext.mapSelected && (
                <SettingsMenu
                    mapName={mapEditorContext.mapSelected}
                    onImageUploaded={handleImageUploaded}
                    turningPoints={turningPoints}
                    onSaveSuccess={() => {
                        // Optional: Add any callback logic after successful save
                    }}
                />
            )}

            {/* Zoom Controls */}
            <div style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                zIndex: 1000,
                display: 'flex',
                flexDirection: 'column',
                gap: '5px'
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

            <ContextMenu.Root>
                <ContextMenu.Trigger>
                    <div>
                        <Stage
                            width={stageSize.width}
                            height={stageSize.height}
                            ref={stageRef}
                            scaleX={stageScale}
                            scaleY={stageScale}
                            x={stagePos.x}
                            y={stagePos.y}
                            draggable={!isDraggingPoint}
                            onDragEnd={handleStageDrag}
                            onWheel={handleWheel}
                            onMouseDown={handleStageMouseDown}
                            onMouseUp={handleStageMouseUp}
                            onDblClick={handleStageDoubleClick}
                        >
                            <Layer>

                                {/* Display uploaded map image if available, otherwise use default */}
                                {uploadedMapImage ? (
                                    (() => {
                                        const imageAspectRatio = uploadedMapImage.width / uploadedMapImage.height;
                                        const stageAspectRatio = stageSize.width / stageSize.height;

                                        let displayWidth, displayHeight;

                                        if (imageAspectRatio > stageAspectRatio) {
                                            // Image is wider than stage - fit by width
                                            displayWidth = stageSize.width;
                                            displayHeight = stageSize.width / imageAspectRatio;
                                        } else {
                                            // Image is taller than stage - fit by height
                                            displayHeight = stageSize.height;
                                            displayWidth = stageSize.height * imageAspectRatio;
                                        }

                                        return (
                                            <Image
                                                x={0}
                                                y={0}
                                                image={uploadedMapImage}
                                                width={displayWidth}
                                                height={displayHeight}
                                            />
                                        );
                                    })()
                                ) : mapImage && (
                                    (() => {
                                        const imageAspectRatio = mapImage.width / mapImage.height;
                                        const stageAspectRatio = stageSize.width / stageSize.height;

                                        let displayWidth, displayHeight;

                                        if (imageAspectRatio > stageAspectRatio) {
                                            // Image is wider than stage - fit by width
                                            displayWidth = stageSize.width;
                                            displayHeight = stageSize.width / imageAspectRatio;
                                        } else {
                                            // Image is taller than stage - fit by height
                                            displayHeight = stageSize.height;
                                            displayWidth = stageSize.height * imageAspectRatio;
                                        }

                                        return (
                                            <Image
                                                x={0}
                                                y={0}
                                                image={mapImage}
                                                width={displayWidth}
                                                height={displayHeight}
                                            />
                                        );
                                    })()
                                )}

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
                                {

                                    turningPoints.map((
                                        turningPoint: {
                                            position: Point,
                                            type: number,
                                            index: number, //type and index are used together. some points are index sensitive
                                            description?: string,
                                            info?: string,
                                            variables?: [{ key: string, value: string }],
                                        }) => (

                                        <Group
                                            key={turningPoint.index} id={`group-${turningPoint.index}`}
                                            x={turningPoint.position[0]} y={turningPoint.position[1]}
                                            draggable
                                            onDragStart={(e) => handleDragStart(e, turningPoint.index)}
                                            onDragMove={(e) => handleDragMove(e, turningPoint.index)}
                                            onDragEnd={(e) => handleDragEnd(e, turningPoint.index)}
                                            onMouseEnter={() => handleEnterMenu(turningPoint.index)}
                                            onMouseLeave={() => handleLeaveMenu(turningPoint.index)}>

                                            {(() => {
                                                const pointStyle = getPointStyling(turningPoint.type);
                                                const IconComponent = pointStyle.icon;

                                                return (
                                                    <>
                                                        {/* Main circle with type-specific color */}
                                                        <Circle
                                                            key={turningPoint.index}
                                                            radius={14}
                                                            fill={pointStyle.fill}
                                                            stroke="#ffffff"
                                                            strokeWidth={2}
                                                            name={turningPoint.index.toString()}
                                                        />

                                                        {/* Icon overlay */}
                                                        <Html>
                                                            <div style={{
                                                                position: 'absolute',
                                                                top: '-12px',
                                                                left: '-12px',
                                                                width: '24px',
                                                                height: '24px',
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                justifyContent: 'center',
                                                                pointerEvents: 'none'
                                                            }}>
                                                                <IconComponent
                                                                    style={{
                                                                        width: '14px',
                                                                        height: '14px',
                                                                        color: 'white',
                                                                        filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.3))'
                                                                    }}
                                                                />
                                                            </div>
                                                        </Html>
                                                    </>
                                                );
                                            })()}

                                            <Html>
                                                {turningPoint.index === activeMenu && !isDraggingPoint &&
                                                    <div
                                                        ref={activeMenu === turningPoint.index ? menuRef : undefined}
                                                        onMouseEnter={() => handleEnterMenu(turningPoint.index)}
                                                        onMouseLeave={() => handleLeaveMenu(turningPoint.index)}
                                                        style={{
                                                            backgroundColor: 'white',
                                                            border: '1px solid #e2e8f0',
                                                            borderRadius: '8px',
                                                            boxShadow: '0 10px 38px -10px rgba(22, 23, 24, 0.35), 0 10px 20px -15px rgba(22, 23, 24, 0.2)',
                                                            padding: '4px',
                                                            minWidth: '180px',
                                                            zIndex: 1000,
                                                            position: 'relative'
                                                        }}
                                                    >
                                                        <div
                                                            onClick={() => openEditDialog(turningPoint)}
                                                            style={{
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                padding: '8px 12px',
                                                                borderRadius: '4px',
                                                                cursor: 'pointer',
                                                                fontSize: '14px',
                                                                fontWeight: '500',
                                                                color: '#374151',
                                                                transition: 'background-color 0.2s ease',
                                                                backgroundColor: 'transparent'
                                                            }}
                                                            onMouseEnter={(e) => {
                                                                e.currentTarget.style.backgroundColor = '#f8fafc';
                                                            }}
                                                            onMouseLeave={(e) => {
                                                                e.currentTarget.style.backgroundColor = 'transparent';
                                                            }}
                                                        >
                                                            <Pencil1Icon style={{ marginRight: '8px', width: '16px', height: '16px', color: '#6366f1' }} />
                                                            Edit Properties
                                                        </div>

                                                        <div
                                                            onClick={() => deleteTurningPoint(turningPoint.index)}
                                                            style={{
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                padding: '8px 12px',
                                                                borderRadius: '4px',
                                                                cursor: 'pointer',
                                                                fontSize: '14px',
                                                                fontWeight: '500',
                                                                color: '#374151',
                                                                transition: 'background-color 0.2s ease',
                                                                backgroundColor: 'transparent'
                                                            }}
                                                            onMouseEnter={(e) => {
                                                                e.currentTarget.style.backgroundColor = '#fef2f2';
                                                                e.currentTarget.style.color = '#dc2626';
                                                            }}
                                                            onMouseLeave={(e) => {
                                                                e.currentTarget.style.backgroundColor = 'transparent';
                                                                e.currentTarget.style.color = '#374151';
                                                            }}
                                                        >
                                                            <TrashIcon style={{ marginRight: '8px', width: '16px', height: '16px', color: '#ef4444' }} />
                                                            Delete Point
                                                        </div>

                                                        {/* Point info display */}
                                                        {(turningPoint.description || turningPoint.info) && (
                                                            <>
                                                                <div style={{
                                                                    height: '1px',
                                                                    backgroundColor: '#e2e8f0',
                                                                    margin: '4px 8px'
                                                                }} />
                                                                <div style={{
                                                                    padding: '8px 12px',
                                                                    fontSize: '12px',
                                                                    color: '#6b7280'
                                                                }}>
                                                                    {turningPoint.description && (
                                                                        <div style={{ marginBottom: '2px', fontWeight: '500' }}>
                                                                            {turningPoint.description}
                                                                        </div>
                                                                    )}
                                                                    {turningPoint.info && (
                                                                        <div style={{ fontSize: '11px', color: '#9ca3af' }}>
                                                                            {turningPoint.info}
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            </>
                                                        )}
                                                    </div>
                                                }
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

            {/* Point Edit Dialog */}
            <Dialog.Root open={editDialogOpen} onOpenChange={setEditDialogOpen}>
                <Dialog.Content style={{ maxWidth: 500 }}>
                    <Dialog.Title>Edit Point Properties</Dialog.Title>
                    <Dialog.Description size="2" mb="4">
                        Configure the properties for turning point #{editingPoint?.index}
                    </Dialog.Description>

                    <Flex direction="column" gap="4">
                        {/* Point Type */}
                        <div>
                            <Text size="2" weight="medium" mb="2" as="div">Point Type</Text>
                            <Select.Root
                                value={editFormData.type.toString()}
                                onValueChange={(value) => setEditFormData(prev => ({ ...prev, type: parseInt(value) }))}
                            >
                                <Select.Trigger style={{ width: '100%' }} />
                                <Select.Content>
                                    <Select.Item value="0">Default (0)</Select.Item>
                                    <Select.Item value="1">Corner Start (1)</Select.Item>
                                    <Select.Item value="2">Cornering (2)</Select.Item>
                                    <Select.Item value="3">Corner End (3)</Select.Item>
                                    <Select.Item value="4">Start/Finish (4)</Select.Item>
                                </Select.Content>
                            </Select.Root>
                        </div>

                        {/* Description */}
                        <div>
                            <Text size="2" weight="medium" mb="2" as="div">Description</Text>
                            <TextField.Root
                                value={editFormData.description}
                                onChange={(e) => setEditFormData(prev => ({ ...prev, description: e.target.value }))}
                                placeholder="e.g., Turn 1 - Fast right hander"
                            />
                        </div>

                        {/* Info */}
                        <div>
                            <Text size="2" weight="medium" mb="2" as="div">Additional Info</Text>
                            <TextField.Root
                                value={editFormData.info}
                                onChange={(e) => setEditFormData(prev => ({ ...prev, info: e.target.value }))}
                                placeholder="e.g., Watch for late braking here"
                            />
                        </div>

                        {/* Variables */}
                        <div>
                            <Flex justify="between" align="center" mb="2">
                                <Text size="2" weight="medium">Variables</Text>
                                <Button size="1" variant="soft" onClick={addVariable}>
                                    Add Variable
                                </Button>
                            </Flex>
                            {editFormData.variables.map((variable, index) => (
                                <Flex key={index} gap="2" mb="2" align="center">
                                    <TextField.Root
                                        placeholder="Key"
                                        value={variable.key}
                                        onChange={(e) => updateVariable(index, 'key', e.target.value)}
                                        style={{ flex: 1 }}
                                    />
                                    <TextField.Root
                                        placeholder="Value"
                                        value={variable.value}
                                        onChange={(e) => updateVariable(index, 'value', e.target.value)}
                                        style={{ flex: 1 }}
                                    />
                                    <Button
                                        size="1"
                                        variant="soft"
                                        color="red"
                                        onClick={() => removeVariable(index)}
                                    >
                                        Remove
                                    </Button>
                                </Flex>
                            ))}
                        </div>
                    </Flex>

                    <Flex gap="3" mt="4" justify="end">
                        <Dialog.Close>
                            <Button variant="soft" color="gray">
                                Cancel
                            </Button>
                        </Dialog.Close>
                        <Button onClick={savePointEdits}>
                            Save Changes
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>
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

