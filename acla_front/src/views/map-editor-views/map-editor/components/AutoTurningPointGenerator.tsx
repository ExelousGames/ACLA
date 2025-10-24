import { ChangeEvent, Dispatch, SetStateAction, useEffect, useState } from 'react';
import { Button, Flex, Slider, Text, TextField } from '@radix-ui/themes';
import { MagicWandIcon } from '@radix-ui/react-icons';
import { Point } from 'utils/curve-tobezier/points-on-curve';
import { RacingTurningPoint } from '../types';

type StageSize = { width: number; height: number };

interface AutoTurningPointGeneratorProps {
    mapImage: HTMLImageElement | null;
    stageSize: StageSize;
    onGenerate: (points: RacingTurningPoint[]) => void;
    disabled?: boolean;
    brightnessThreshold?: number;
}

const DEFAULT_BRIGHTNESS_THRESHOLD = 210; // 0-255, assume white track on dark background
const DEFAULT_ANGLE_STEP = 6; // degrees between radial samples (~60 points per lap)
const DEFAULT_MIN_DISTANCE_FACTOR = 0.025; // percent of longest dimension used to prune noisy points
const DEFAULT_SMOOTHING_ITERATIONS = 2;
const MIN_REQUIRED_POINTS = 4;

const AutoTurningPointGenerator = ({
    mapImage,
    stageSize,
    onGenerate,
    disabled = false,
    brightnessThreshold = DEFAULT_BRIGHTNESS_THRESHOLD,
}: AutoTurningPointGeneratorProps) => {
    const [isProcessing, setIsProcessing] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [brightness, setBrightness] = useState(brightnessThreshold);
    const [angleStep, setAngleStep] = useState(DEFAULT_ANGLE_STEP);
    const [minDistanceFactor, setMinDistanceFactor] = useState(DEFAULT_MIN_DISTANCE_FACTOR);
    const [smoothingIterations, setSmoothingIterations] = useState(DEFAULT_SMOOTHING_ITERATIONS);
    const [showOptions, setShowOptions] = useState(false);
    const normalizedDefaultBrightness = Math.round(brightnessThreshold);

    useEffect(() => {
        setBrightness(normalizedDefaultBrightness);
    }, [normalizedDefaultBrightness]);

    const handleGenerate = async () => {
        if (!mapImage) {
            setErrorMessage('No map image available for analysis.');
            return;
        }

        setIsProcessing(true);
        setStatusMessage('Analyzing track…');
        setErrorMessage(null);

        try {
            // Yield once so the button state updates before heavy processing
            await new Promise(requestAnimationFrame);
            const generatedPoints = detectTurningPointsFromImage(mapImage, stageSize, {
                brightnessThreshold: brightness,
                angleStep,
                minDistanceFactor,
                smoothingIterations,
            });

            if (generatedPoints.length < MIN_REQUIRED_POINTS) {
                throw new Error('Not enough track detail detected to construct turning points.');
            }

            onGenerate(generatedPoints);
            setStatusMessage(`Generated ${generatedPoints.length} turning points from map image.`);
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to generate turning points.';
            setErrorMessage(message);
            setStatusMessage(null);
        } finally {
            setIsProcessing(false);
        }
    };

    const buttonDisabled = disabled || isProcessing || !mapImage;
    const isDefaultSettings =
        brightness === normalizedDefaultBrightness &&
        angleStep === DEFAULT_ANGLE_STEP &&
        Math.abs(minDistanceFactor - DEFAULT_MIN_DISTANCE_FACTOR) < 1e-6 &&
        smoothingIterations === DEFAULT_SMOOTHING_ITERATIONS;

    const handleResetOptions = () => {
        setBrightness(normalizedDefaultBrightness);
        setAngleStep(DEFAULT_ANGLE_STEP);
        setMinDistanceFactor(DEFAULT_MIN_DISTANCE_FACTOR);
        setSmoothingIterations(DEFAULT_SMOOTHING_ITERATIONS);
    };

    const handleSliderChange = (setter: Dispatch<SetStateAction<number>>, value: number | undefined) => {
        if (typeof value === 'number' && Number.isFinite(value)) {
            setter(value);
        }
    };

    const handleBrightnessInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        const next = Number(event.target.value);
        if (!Number.isNaN(next)) {
            setBrightness(Math.round(clampNumber(next, 0, 255)));
        }
    };

    const handleAngleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        const next = Number(event.target.value);
        if (!Number.isNaN(next)) {
            setAngleStep(Math.round(clampNumber(next, 1, 45)));
        }
    };

    const handleSpacingInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        const next = Number(event.target.value);
        if (!Number.isNaN(next)) {
            const normalized = clampNumber(next, 0.5, 12);
            const rounded = Math.round(normalized * 10) / 10;
            setMinDistanceFactor(rounded / 100);
        }
    };

    const handleSmoothingInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        const next = Number(event.target.value);
        if (!Number.isNaN(next)) {
            setSmoothingIterations(Math.max(0, Math.round(clampNumber(next, 0, 10))));
        }
    };

    return (
        <div style={{
            position: 'absolute',
            top: '10px',
            left: '135px',
            zIndex: 1000,
            display: 'flex',
            flexDirection: 'column',
            gap: '6px'
        }}>
            <Button
                onClick={handleGenerate}
                disabled={buttonDisabled}
                variant="soft"
                size="2"
                style={{
                    backgroundColor: 'white',
                    cursor: buttonDisabled ? 'not-allowed' : 'pointer'
                }}
            >
                <MagicWandIcon />
                {isProcessing ? 'Analyzing…' : 'Auto Generate Points'}
            </Button>
            <Button
                onClick={() => setShowOptions((prev) => !prev)}
                variant="surface"
                size="1"
                disabled={isProcessing}
            >
                {showOptions ? 'Hide Tuning Options' : 'Show Tuning Options'}
            </Button>
            {showOptions && (
                <div
                    style={{
                        padding: '10px 12px',
                        backgroundColor: 'rgba(255,255,255,0.95)',
                        borderRadius: '8px',
                        border: '1px solid #e5e7eb',
                        boxShadow: '0 8px 20px rgba(0,0,0,0.08)',
                        width: '280px'
                    }}
                >
                    <Flex direction="column" gap="3">
                        <Flex direction="column" gap="1">
                            <Text size="1" color="gray">
                                Brightness threshold
                            </Text>
                            <Flex align="center" gap="2">
                                <Slider
                                    value={[brightness]}
                                    min={50}
                                    max={255}
                                    step={1}
                                    onValueChange={(values) => handleSliderChange(setBrightness, values[0])}
                                    disabled={isProcessing}
                                    style={{ width: '140px' }}
                                />
                                <TextField.Root
                                    type="number"
                                    min={0}
                                    max={255}
                                    value={brightness.toString()}
                                    onChange={handleBrightnessInputChange}
                                    disabled={isProcessing}
                                    style={{ width: '70px' }}
                                />
                            </Flex>
                        </Flex>
                        <Flex direction="column" gap="1">
                            <Text size="1" color="gray">
                                Angle step (degrees)
                            </Text>
                            <Flex align="center" gap="2">
                                <Slider
                                    value={[angleStep]}
                                    min={2}
                                    max={20}
                                    step={1}
                                    onValueChange={(values) => handleSliderChange(setAngleStep, values[0])}
                                    disabled={isProcessing}
                                    style={{ width: '140px' }}
                                />
                                <TextField.Root
                                    type="number"
                                    min={1}
                                    max={45}
                                    value={angleStep.toString()}
                                    onChange={handleAngleInputChange}
                                    disabled={isProcessing}
                                    style={{ width: '70px' }}
                                />
                            </Flex>
                        </Flex>
                        <Flex direction="column" gap="1">
                            <Text size="1" color="gray">
                                Point spacing (% of longest side)
                            </Text>
                            <Flex align="center" gap="2">
                                <Slider
                                    value={[minDistanceFactor * 100]}
                                    min={0.5}
                                    max={12}
                                    step={0.1}
                                    onValueChange={(values) => {
                                        const percent = values[0];
                                        if (typeof percent === 'number' && Number.isFinite(percent)) {
                                            const clamped = clampNumber(percent, 0.5, 12);
                                            const rounded = Math.round(clamped * 10) / 10;
                                            setMinDistanceFactor(rounded / 100);
                                        }
                                    }}
                                    disabled={isProcessing}
                                    style={{ width: '140px' }}
                                />
                                <TextField.Root
                                    type="number"
                                    min={0.5}
                                    max={12}
                                    step={0.1}
                                    value={(minDistanceFactor * 100).toFixed(1)}
                                    onChange={handleSpacingInputChange}
                                    disabled={isProcessing}
                                    style={{ width: '70px' }}
                                />
                            </Flex>
                        </Flex>
                        <Flex direction="column" gap="1">
                            <Text size="1" color="gray">
                                Smoothing iterations
                            </Text>
                            <Flex align="center" gap="2">
                                <Slider
                                    value={[smoothingIterations]}
                                    min={0}
                                    max={6}
                                    step={1}
                                    onValueChange={(values) => handleSliderChange(setSmoothingIterations, values[0])}
                                    disabled={isProcessing}
                                    style={{ width: '140px' }}
                                />
                                <TextField.Root
                                    type="number"
                                    min={0}
                                    max={10}
                                    value={smoothingIterations.toString()}
                                    onChange={handleSmoothingInputChange}
                                    disabled={isProcessing}
                                    style={{ width: '70px' }}
                                />
                            </Flex>
                        </Flex>
                        <Flex justify="end">
                            <Button
                                size="1"
                                variant="soft"
                                disabled={isDefaultSettings || isProcessing}
                                onClick={handleResetOptions}
                            >
                                Reset to defaults
                            </Button>
                        </Flex>
                    </Flex>
                </div>
            )}
            {statusMessage && (
                <div
                    style={{
                        padding: '6px 10px',
                        backgroundColor: '#d1fae5',
                        border: '1px solid #34d399',
                        borderRadius: '6px',
                        color: '#047857',
                        fontSize: '13px',
                        maxWidth: '260px'
                    }}
                >
                    <Text size="2">
                        {statusMessage}
                    </Text>
                </div>
            )}
            {errorMessage && (
                <div
                    style={{
                        padding: '6px 10px',
                        backgroundColor: '#fee2e2',
                        border: '1px solid #ef4444',
                        borderRadius: '6px',
                        color: '#b91c1c',
                        fontSize: '13px',
                        maxWidth: '260px'
                    }}
                >
                    <Text size="2">
                        {errorMessage}
                    </Text>
                </div>
            )}
        </div>
    );
};

export default AutoTurningPointGenerator;

interface DetectionOptions {
    brightnessThreshold: number;
    angleStep: number;
    minDistanceFactor: number;
    smoothingIterations: number;
}

function detectTurningPointsFromImage(
    image: HTMLImageElement,
    stageSize: StageSize,
    { brightnessThreshold, angleStep, minDistanceFactor, smoothingIterations }: DetectionOptions
): RacingTurningPoint[] {
    const sanitizedBrightness = clampNumber(Math.round(brightnessThreshold), 0, 255);
    const sanitizedAngleStep = Math.max(1, Math.min(90, Math.round(angleStep)));
    const sanitizedMinDistanceFactor = Math.max(0.001, minDistanceFactor);
    const sanitizedSmoothingIterations = Math.max(0, Math.round(smoothingIterations));
    const imageWidth = image.naturalWidth || image.width;
    const imageHeight = image.naturalHeight || image.height;

    if (!imageWidth || !imageHeight) {
        throw new Error('Failed to read map image dimensions.');
    }

    const canvas = document.createElement('canvas');
    canvas.width = imageWidth;
    canvas.height = imageHeight;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
        throw new Error('Canvas context is not available in this environment.');
    }

    ctx.drawImage(image, 0, 0, imageWidth, imageHeight);
    const imageData = ctx.getImageData(0, 0, imageWidth, imageHeight);
    const mask = buildTrackMask(imageData, sanitizedBrightness);
    const skeletonMask = buildSkeletonMask(mask, imageWidth, imageHeight);
    const polylinePoints = extractPolylineFromSkeleton(skeletonMask, imageWidth, imageHeight);

    const centroid = computeCentroid(mask, imageWidth, imageHeight);
    if (!centroid) {
        throw new Error('Unable to locate white track pixels in the map image.');
    }

    const basePoints = polylinePoints.length >= MIN_REQUIRED_POINTS
        ? polylinePoints
        : sampleTrackByRadialSweep(mask, imageWidth, imageHeight, centroid, sanitizedAngleStep);

    if (basePoints.length < MIN_REQUIRED_POINTS) {
        return [];
    }

    const minDistancePixels = Math.max(1, Math.max(imageWidth, imageHeight) * sanitizedMinDistanceFactor);
    const cleaned = cleanAndSmooth(basePoints, minDistancePixels, sanitizedSmoothingIterations);
    const scaledPoints = scalePointsToStage(cleaned, imageWidth, imageHeight, stageSize);

    return scaledPoints.map((point, index) => ({
        index,
        type: 0,
        position: point,
        description: '',
        info: '',
        variables: [],
    }));
}

function buildTrackMask(imageData: ImageData, threshold: number): Uint8Array {
    const { data, width, height } = imageData;
    const mask = new Uint8Array(width * height);

    for (let i = 0; i < width * height; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        const brightness = (r + g + b) / 3;
        mask[i] = brightness >= threshold ? 1 : 0;
    }

    return mask;
}

function buildSkeletonMask(mask: Uint8Array, width: number, height: number): Uint8Array {
    const thinned = zhangSuenThinning(mask, width, height);
    pruneDanglingPixels(thinned, width, height);
    return thinned;
}

function computeCentroid(mask: Uint8Array, width: number, height: number): Point | null {
    let sumX = 0;
    let sumY = 0;
    let count = 0;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (mask[y * width + x]) {
                sumX += x;
                sumY += y;
                count += 1;
            }
        }
    }

    if (count === 0) {
        return null;
    }

    return [sumX / count, sumY / count];
}

function sampleTrackByRadialSweep(
    mask: Uint8Array,
    width: number,
    height: number,
    centroid: Point,
    angleStep: number
): Point[] {
    const points: Point[] = [];
    const maxRadius = Math.sqrt(width * width + height * height);

    for (let angle = 0; angle < 360; angle += angleStep) {
        const radians = (angle * Math.PI) / 180;
        const cos = Math.cos(radians);
        const sin = Math.sin(radians);

        let entered = false;
        let entryPoint: Point | null = null;
        let lastInsidePoint: Point | null = null;

        for (let r = 0; r < maxRadius; r++) {
            const x = Math.round(centroid[0] + cos * r);
            const y = Math.round(centroid[1] + sin * r);

            if (x < 0 || x >= width || y < 0 || y >= height) {
                break;
            }

            const isInside = mask[y * width + x] === 1;

            if (isInside) {
                if (!entered) {
                    entered = true;
                    entryPoint = [x, y];
                }
                lastInsidePoint = [x, y];
            } else if (entered) {
                break;
            }
        }

        if (entryPoint && lastInsidePoint) {
            const midPoint: Point = [
                (entryPoint[0] + lastInsidePoint[0]) / 2,
                (entryPoint[1] + lastInsidePoint[1]) / 2,
            ];
            points.push(midPoint);
        }
    }

    return points;
}

function cleanAndSmooth(points: Point[], minDistance: number, smoothingIterations: number): Point[] {
    if (points.length === 0) {
        return points;
    }

    const deduped: Point[] = [];
    const minDistanceSq = minDistance * minDistance;

    let lastPoint = points[0];
    deduped.push(lastPoint);

    for (let i = 1; i < points.length; i++) {
        const current = points[i];
        if (distanceSquared(current, lastPoint) >= minDistanceSq) {
            deduped.push(current);
            lastPoint = current;
        }
    }

    if (deduped.length < MIN_REQUIRED_POINTS) {
        return points;
    }

    if (smoothingIterations <= 0) {
        return deduped;
    }

    return smoothPath(deduped, smoothingIterations);
}

function smoothPath(points: Point[], iterations: number): Point[] {
    if (points.length < 3) {
        return points;
    }

    let smoothed = [...points];

    for (let iter = 0; iter < iterations; iter++) {
        const updated: Point[] = [];
        for (let i = 0; i < smoothed.length; i++) {
            const prev = smoothed[(i - 1 + smoothed.length) % smoothed.length];
            const current = smoothed[i];
            const next = smoothed[(i + 1) % smoothed.length];

            updated.push([
                (prev[0] + current[0] + next[0]) / 3,
                (prev[1] + current[1] + next[1]) / 3,
            ]);
        }
        smoothed = updated;
    }

    return smoothed;
}

function scalePointsToStage(
    points: Point[],
    imageWidth: number,
    imageHeight: number,
    stageSize: StageSize
): Point[] {
    if (points.length === 0) {
        return points;
    }

    const imageAspectRatio = imageWidth / imageHeight;
    const stageAspectRatio = stageSize.width / stageSize.height;

    let displayWidth: number;
    let displayHeight: number;

    if (imageAspectRatio > stageAspectRatio) {
        displayWidth = stageSize.width;
        displayHeight = stageSize.width / imageAspectRatio;
    } else {
        displayHeight = stageSize.height;
        displayWidth = stageSize.height * imageAspectRatio;
    }

    const scaleX = displayWidth / imageWidth;
    const scaleY = displayHeight / imageHeight;

    return points.map(([x, y]) => [x * scaleX, y * scaleY]);
}

function distanceSquared(a: Point, b: Point): number {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return dx * dx + dy * dy;
}

function clampNumber(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

function extractPolylineFromSkeleton(skeleton: Uint8Array, width: number, height: number): Point[] {
    const startIndex = findFirstSetBit(skeleton);
    if (startIndex === -1) {
        return [];
    }

    const startX = startIndex % width;
    const startY = Math.floor(startIndex / width);
    const startPoint: [number, number] = [startX, startY];

    const visited = new Uint8Array(width * height);
    const points: Point[] = [];
    let current: [number, number] = startPoint;
    let prev: [number, number] | null = null;
    let lastDirection: [number, number] | null = null;
    const maxIterations = width * height * 4;

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        const [cx, cy] = current;
        const index = cy * width + cx;

        points.push([cx + 0.5, cy + 0.5] as Point);
        visited[index] = 1;

        const neighborCoords = getNeighborCoords(cx, cy, width, height);
        const neighbors = neighborCoords.filter(({ x, y }) => skeleton[y * width + x] === 1);

        if (neighbors.length === 0) {
            break;
        }

        let candidates = neighbors.filter(({ x, y }) => !(prev && x === prev[0] && y === prev[1]));
        if (candidates.length === 0) {
            candidates = neighbors;
        }

        let chosen: { x: number; y: number } | null = null;

        const unvisited = candidates.filter(({ x, y }) => visited[y * width + x] === 0);
        const pool = unvisited.length > 0 ? unvisited : candidates;

        if (pool.length === 1 || !lastDirection) {
            chosen = pool[0];
        } else {
            let bestScore = -Infinity;
            for (const candidate of pool) {
                const direction: [number, number] = [candidate.x - cx, candidate.y - cy];
                const magnitude = Math.hypot(direction[0], direction[1]) || 1;
                const normalized: [number, number] = [direction[0] / magnitude, direction[1] / magnitude];
                const score = normalized[0] * lastDirection[0] + normalized[1] * lastDirection[1];
                if (score > bestScore) {
                    bestScore = score;
                    chosen = candidate;
                }
            }
        }

        if (!chosen) {
            break;
        }

        if (chosen.x === startX && chosen.y === startY) {
            points.push([startX + 0.5, startY + 0.5] as Point);
            break;
        }

        prev = current;
        const stepVector: [number, number] = [chosen.x - cx, chosen.y - cy];
        const length = Math.hypot(stepVector[0], stepVector[1]) || 1;
        lastDirection = [stepVector[0] / length, stepVector[1] / length];
        current = [chosen.x, chosen.y] as [number, number];
    }

    if (points.length < MIN_REQUIRED_POINTS) {
        return points;
    }

    // Remove duplicate trailing point if smoothing is going to wrap
    if (points.length > 1) {
        const first = points[0];
        const last = points[points.length - 1];
        if (distanceSquared(first, last) < 1e-6) {
            points.pop();
        }
    }

    return points;
}

function findFirstSetBit(mask: Uint8Array): number {
    for (let index = 0; index < mask.length; index++) {
        if (mask[index] === 1) {
            return index;
        }
    }
    return -1;
}

function getNeighborCoords(x: number, y: number, width: number, height: number): Array<{ x: number; y: number }> {
    const neighbors: Array<{ x: number; y: number }> = [];

    for (let offsetY = -1; offsetY <= 1; offsetY++) {
        for (let offsetX = -1; offsetX <= 1; offsetX++) {
            if (offsetX === 0 && offsetY === 0) {
                continue;
            }

            const nx = x + offsetX;
            const ny = y + offsetY;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                neighbors.push({ x: nx, y: ny });
            }
        }
    }

    return neighbors;
}

function pruneDanglingPixels(mask: Uint8Array, width: number, height: number): void {
    const toRemove: number[] = [];

    const neighborOffsets = [
        [-1, -1], [0, -1], [1, -1],
        [-1, 0], [1, 0],
        [-1, 1], [0, 1], [1, 1],
    ];

    let removed = false;
    do {
        removed = false;
        toRemove.length = 0;

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const index = y * width + x;
                if (mask[index] !== 1) {
                    continue;
                }

                let neighborCount = 0;
                for (const [ox, oy] of neighborOffsets) {
                    if (mask[(y + oy) * width + (x + ox)] === 1) {
                        neighborCount += 1;
                    }
                }

                if (neighborCount <= 1) {
                    toRemove.push(index);
                }
            }
        }

        if (toRemove.length > 0) {
            removed = true;
            for (const index of toRemove) {
                mask[index] = 0;
            }
        }
    } while (removed);
}

function zhangSuenThinning(mask: Uint8Array, width: number, height: number): Uint8Array {
    const working = mask.slice();
    const neighborIndices = [
        [0, -1], [1, -1], [1, 0], [1, 1],
        [0, 1], [-1, 1], [-1, 0], [-1, -1],
    ];

    const size = width * height;
    let pixelsRemoved = false;

    do {
        pixelsRemoved = false;
        const toRemoveStep1: number[] = [];
        const toRemoveStep2: number[] = [];

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const index = y * width + x;
                if (working[index] !== 1) {
                    continue;
                }

                const neighbors = neighborIndices.map(([ox, oy]) => working[(y + oy) * width + (x + ox)]);
                const neighborCount = neighbors.reduce((sum, value) => sum + value, 0);
                if (neighborCount < 2 || neighborCount > 6) {
                    continue;
                }

                const transitions = countZeroToOneTransitions(neighbors);
                if (transitions !== 1) {
                    continue;
                }

                const p2 = neighbors[0];
                const p4 = neighbors[2];
                const p6 = neighbors[4];
                const p8 = neighbors[6];

                if (p2 * p4 * p6 !== 0) {
                    continue;
                }

                if (p4 * p6 * p8 !== 0) {
                    continue;
                }

                toRemoveStep1.push(index);
            }
        }

        if (toRemoveStep1.length > 0) {
            pixelsRemoved = true;
            for (const index of toRemoveStep1) {
                working[index] = 0;
            }
        }

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const index = y * width + x;
                if (working[index] !== 1) {
                    continue;
                }

                const neighbors = neighborIndices.map(([ox, oy]) => working[(y + oy) * width + (x + ox)]);
                const neighborCount = neighbors.reduce((sum, value) => sum + value, 0);
                if (neighborCount < 2 || neighborCount > 6) {
                    continue;
                }

                const transitions = countZeroToOneTransitions(neighbors);
                if (transitions !== 1) {
                    continue;
                }

                const p2 = neighbors[0];
                const p4 = neighbors[2];
                const p6 = neighbors[4];
                const p8 = neighbors[6];

                if (p2 * p4 * p8 !== 0) {
                    continue;
                }

                if (p2 * p6 * p8 !== 0) {
                    continue;
                }

                toRemoveStep2.push(index);
            }
        }

        if (toRemoveStep2.length > 0) {
            pixelsRemoved = true;
            for (const index of toRemoveStep2) {
                working[index] = 0;
            }
        }
    } while (pixelsRemoved);

    for (let index = 0; index < size; index++) {
        working[index] = working[index] ? 1 : 0;
    }

    return working;
}

function countZeroToOneTransitions(neighbors: number[]): number {
    let transitions = 0;
    for (let i = 0; i < neighbors.length; i++) {
        const current = neighbors[i];
        const next = neighbors[(i + 1) % neighbors.length];
        if (current === 0 && next === 1) {
            transitions += 1;
        }
    }
    return transitions;
}
