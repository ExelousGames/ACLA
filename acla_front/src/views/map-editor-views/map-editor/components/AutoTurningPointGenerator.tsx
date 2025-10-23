import { useState } from 'react';
import { Button, Text } from '@radix-ui/themes';
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
                brightnessThreshold,
                angleStep: DEFAULT_ANGLE_STEP,
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
}

function detectTurningPointsFromImage(
    image: HTMLImageElement,
    stageSize: StageSize,
    { brightnessThreshold, angleStep }: DetectionOptions
): RacingTurningPoint[] {
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
    const mask = buildTrackMask(imageData, brightnessThreshold);

    const centroid = computeCentroid(mask, imageWidth, imageHeight);
    if (!centroid) {
        throw new Error('Unable to locate white track pixels in the map image.');
    }

    const radialPoints = sampleTrackByRadialSweep(mask, imageWidth, imageHeight, centroid, angleStep);
    if (radialPoints.length < MIN_REQUIRED_POINTS) {
        return [];
    }

    const cleaned = cleanAndSmooth(radialPoints, Math.max(imageWidth, imageHeight) * 0.025);
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

function cleanAndSmooth(points: Point[], minDistance: number): Point[] {
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

    return smoothPath(deduped, 2);
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
