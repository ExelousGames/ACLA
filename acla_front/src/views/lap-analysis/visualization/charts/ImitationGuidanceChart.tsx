import React, { useContext, useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Card, Text, Box, Grid, Badge, Separator, Progress, Flex } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';
import apiService from 'services/api.service';
import styles from './ImitationGuidanceChart.module.css';

interface ImitationGuidanceData {
    preloadSentences?: {
        throttle_guidance?: string[];
        brake_guidance?: string[];
        steering_guidance?: string[];
    };
    trackData?: {
        _guidance_enabled: boolean;
        _prediction_result: {
            total_corners: number;
            corner_predictions: {
                [key: string]: {
                    phases: {
                        [phaseName: string]: {
                            phase_position: number;
                            optimal_actions: {
                                optimal_throttle?: {
                                    value: number;
                                    description: string;
                                    change_rate: number;
                                    rapidity: string;
                                };
                                optimal_brake?: {
                                    value: number;
                                    description: string;
                                    change_rate: number;
                                    rapidity: string;
                                };
                                optimal_steering?: {
                                    value: number;
                                    description: string;
                                    change_rate: number;
                                    rapidity: string;
                                };
                                optimal_speed?: {
                                    value: number;
                                    description: string;
                                    change_rate: number;
                                    rapidity: string;
                                };
                            };
                            confidence: number;
                            actions_summary: string[];
                        };
                    };
                    corner_summary: {
                        phases_with_predictions: number;
                        corner_start_position: number;
                        corner_end_position: number;
                        average_confidence: number;
                    };
                };
            };
        };
    };
}

interface CurrentGuidance {
    corner: number;
    phase: string;
    phaseData: any;
    position: number;
    actions: {
        throttle: string;
        brake: string;
        steering: string;
        speed: string;
    };
    actionDetails: {
        throttle: { changeRate: number; value: number; rapidity: string } | null;
        brake: { changeRate: number; value: number; rapidity: string } | null;
        steering: { changeRate: number; value: number; rapidity: string } | null;
        speed: { changeRate: number; value: number; rapidity: string } | null;
    };
    confidence: number;
    humanReadableText: string;
    nextPhase?: {
        corner: number;
        name: string;
        data?: any;
        distance: number;
        timeToReach: number;
    };
}

// Configuration constants
const GUIDANCE_TRANSITION_THRESHOLD = 1; // seconds before switching to next phase guidance
const URGENT_TRANSITION_THRESHOLD = 0.5; // seconds for urgent next phase warnings

const ImitationGuidanceChart: React.FC<VisualizationProps> = ({
    id,
    data,
    config,
    width = '100%',
    height = 400
}) => {
    const analysisContext = useContext(AnalysisContext);
    const [isLoading, setIsLoading] = useState(false);
    const [currentGuidance, setCurrentGuidance] = useState<CurrentGuidance | null>(null);
    const [trackGuidanceData, setTrackGuidanceData] = useState<ImitationGuidanceData | null>(null);
    const [positionHistory, setPositionHistory] = useState<Array<{ position: number, timestamp: number, lap: number }>>([]);
    const [currentSpeed, setCurrentSpeed] = useState<number>(0);
    const [currentCornerIndex, setCurrentCornerIndex] = useState<number>(1); // Track which corner we're currently monitoring
    const [lastCompletedLap, setLastCompletedLap] = useState<number>(0); // Track last completed lap for detection
    const [isInActivePhase, setIsInActivePhase] = useState<boolean>(false); // Track if currently in a corner phase
    const [lastSentGuidanceMessage, setLastSentGuidanceMessage] = useState<string>(''); // Track last sent guidance to avoid duplicates

    const [currentPhaseData, setCurrentPhaseData] = useState<any>(null); // Store current guidance data for reference
    // Store corner boundaries for quick lookup
    const [cornerBoundaries, setCornerBoundaries] = useState<Array<{ cornerNum: number, start: number, end: number }>>([]);

    // Use refs to prevent unnecessary re-renders
    const lastPositionRef = useRef<number>(0);
    const lastCornerRef = useRef<number>(1);
    const lastGuidanceUpdateRef = useRef<number>(0);

    // Extract guidance data from the visualization data
    useEffect(() => {
        if (data?.trackData && data?.preloadSentences) {

            setTrackGuidanceData({
                trackData: data.trackData,
                preloadSentences: data.preloadSentences
            });
        }
    }, [data]);

    // Initialize corner boundaries when track data is loaded
    useEffect(() => {
        if (trackGuidanceData?.trackData?._prediction_result?.corner_predictions) {
            const cornerPredictions = trackGuidanceData.trackData._prediction_result.corner_predictions;
            const boundaries = Object.entries(cornerPredictions).map(([cornerKey, cornerData]) => ({
                cornerNum: parseInt(cornerKey.replace('corner_', '')),
                start: cornerData.corner_summary.corner_start_position,
                end: cornerData.corner_summary.corner_end_position
            }));
            setCornerBoundaries(boundaries);
            // Initialize to first corner
            if (boundaries.length > 0) {
                setCurrentCornerIndex(boundaries[0].cornerNum);
            }
        }
    }, [trackGuidanceData]);

    // Function to get normalized car position (0-1 range) - simplified since it's always available
    const normalizeCarPosition = useCallback((telemetryData: any): number => {
        if (!telemetryData) {
            return 0;
        }

        // Use the normalized position that's always available in telemetry data
        const position = telemetryData.Graphics_normalized_car_position || 0;
        return Math.max(0, Math.min(1, position));
    }, []);

    // Function to calculate approach speed based on position history
    const calculateApproachSpeed = useCallback((history: Array<{ position: number, timestamp: number, lap: number }>): number => {
        if (history.length < 3) return 0;

        const recentHistory = history.slice(-5);
        if (recentHistory.length < 3) return 0;

        const validDistances = recentHistory.slice(1).map((curr, i) => {
            const prev = recentHistory[i];
            let distance = curr.position - prev.position;

            // Only use same lap data - ignore lap changes completely
            if (curr.lap !== prev.lap) {
                return null;
            } else if (Math.abs(distance) > 0.5) {
                // Large position jump within same lap - likely telemetry glitch
                return null;
            }

            return {
                distance: Math.abs(distance),
                time: curr.timestamp - prev.timestamp
            };
        }).filter((d): d is { distance: number; time: number } => d !== null);

        const totalDistance = validDistances.reduce((sum, d) => sum + d.distance, 0);
        const totalTime = validDistances.reduce((sum, d) => sum + d.time, 0);

        return totalTime > 0 ? totalDistance / (totalTime / 1000) : 0;
    }, []);

    // Function to find the current corner and phase based on position with optimized corner tracking
    const findCurrentCornerAndPhase = useCallback((position: number, approachSpeed: number = 0, currentLap: number = 0) => {
        if (!trackGuidanceData?.trackData?._prediction_result?.corner_predictions || cornerBoundaries.length === 0) {
            console.log('No corner predictions or boundaries available');
            return null;
        }

        const cornerPredictions = trackGuidanceData.trackData._prediction_result.corner_predictions;

        // Find which corner the current position falls into
        const findCurrentCornerNumber = (position: number): number => {
            // Check each corner boundary to see if position falls within it
            for (const boundary of cornerBoundaries) {
                if (position >= boundary.start && position <= boundary.end) {
                    console.log(`Position ${position.toFixed(3)} is in corner ${boundary.cornerNum} (${boundary.start.toFixed(3)} - ${boundary.end.toFixed(3)})`);
                    // Update the current corner index for next time
                    if (boundary.cornerNum !== currentCornerIndex) {
                        setCurrentCornerIndex(boundary.cornerNum);
                    }
                    return boundary.cornerNum;
                }
            }

            // If not in any corner boundary, keep using current corner index
            console.log(`Position ${position.toFixed(3)} not in any corner, keeping current corner ${currentCornerIndex}`);
            return currentCornerIndex;
        };

        // Determine which corner we're currently in or approaching
        const activeCornerNumber = findCurrentCornerNumber(position);

        // Helper function to find phase and build result for a given corner
        const buildCornerResult = (cornerIndex: number) => {
            const cornerData = cornerPredictions[`corner_${cornerIndex}`];
            if (!cornerData) return null;

            const phaseResult = findPhaseInCorner(cornerData, position);
            if (!phaseResult) return null;

            const nextPhaseInfo = getNextPhaseInfo(cornerIndex, phaseResult.phaseName, position, approachSpeed, currentLap);
            return {
                corner: cornerIndex,
                phase: phaseResult.phaseName,
                phaseData: phaseResult.phaseData,
                nextPhase: nextPhaseInfo ? {
                    corner: nextPhaseInfo.corner,
                    name: nextPhaseInfo.name,
                    data: nextPhaseInfo.data,
                    distance: nextPhaseInfo.distance,
                    timeToReach: nextPhaseInfo.timeToReach,
                } : null
            };
        };

        // Helper function to determine which phase the position is in
        const findPhaseInCorner = (cornerData: any, position: number) => {
            const validPhases = Object.entries(cornerData.phases)
                .filter(([_, phaseData]: [string, any]) => phaseData.phase_position !== undefined);

            if (validPhases.length === 0) return null;

            let closestPhase = null;
            let minDistance = Infinity;

            for (const [phaseName, phaseData] of validPhases) {
                const distance = Math.abs(position - (phaseData as any).phase_position);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestPhase = { phaseName, phaseData };
                }
            }
            return closestPhase;
        };

        // Helper function to get next phase data directly
        const getNextPhaseInfo = (cornerNum: number, currentPhaseName: string, position: number, approachSpeed: number, currentLap: number = 0) => {
            const currentCornerData = cornerPredictions[`corner_${cornerNum}`];
            if (!currentCornerData) return null;

            const phases = Object.entries(currentCornerData.phases)
                .filter(([_, phaseData]: [string, any]) => phaseData.phase_position !== undefined);

            const currentPhaseIndex = phases.findIndex(([name, _]) => name === currentPhaseName);

            let nextPhaseName = null;
            let nextPhaseData = null;
            let nextPhasePosition = 0;
            let nextPhaseCorner = cornerNum;

            if (currentPhaseIndex >= 0 && currentPhaseIndex < phases.length - 1) {
                // Next phase is in same corner
                const [name, data] = phases[currentPhaseIndex + 1];
                nextPhaseName = name;
                nextPhaseData = data;
                nextPhasePosition = (data as any).phase_position;
                nextPhaseCorner = cornerNum;
            } else {
                // Look for next corner's first phase
                const totalCorners = Object.keys(cornerPredictions).length;
                const nextCornerNum = (cornerNum % totalCorners) + 1;
                const nextCornerData = cornerPredictions[`corner_${nextCornerNum}`];

                if (nextCornerData) {
                    const nextCornerPhases = Object.entries(nextCornerData.phases)
                        .filter(([_, phaseData]: [string, any]) => phaseData.phase_position !== undefined);

                    if (nextCornerPhases.length > 0) {
                        const [name, data] = nextCornerPhases[0];
                        nextPhaseName = name; // Just the phase name, no corner prefix
                        nextPhaseData = data;
                        nextPhasePosition = (data as any).phase_position;
                        nextPhaseCorner = nextCornerNum;
                    }
                }
            }

            if (!nextPhaseName || !nextPhaseData) return null;

            let distance = Math.abs(nextPhasePosition - position);

            // Simple distance calculation without wrap-around
            if (nextPhaseCorner !== cornerNum) {
                // Next phase is in a different corner - use simple distance
                distance = Math.abs(nextPhasePosition - position);
            }

            console.log(`Distance calculation: pos=${position.toFixed(3)}, nextPos=${nextPhasePosition.toFixed(3)}, corner=${cornerNum}, nextCorner=${nextPhaseCorner}, distance=${distance.toFixed(3)}`);

            const timeToReach = approachSpeed > 0 ? distance / approachSpeed : Infinity;

            return {
                corner: nextPhaseCorner,
                name: nextPhaseName,
                data: nextPhaseData,
                distance: distance,
                timeToReach: timeToReach,
            };
        };


        // Use the dynamically determined corner instead of static currentCornerIndex
        return buildCornerResult(activeCornerNumber);
    }, [trackGuidanceData, cornerBoundaries, currentCornerIndex]);

    // Function to generate human-readable guidance text
    const generateHumanReadableText = useCallback((actions: any, phase: string): string => {
        const actionTypes = ['optimal_throttle', 'optimal_brake', 'optimal_steering', 'optimal_speed'];

        if (!trackGuidanceData?.preloadSentences) {
            const descriptions = actionTypes
                .map(type => actions[type]?.description)
                .filter(Boolean);
            return descriptions.join('. ') + (descriptions.length ? '.' : '');
        }

        const sentences = trackGuidanceData.preloadSentences;
        const guidanceTexts = [];

        // Helper function to get sentence index based on rapidity
        const getSentenceIndex = (rapidity: string): number | null => {
            switch (rapidity?.toLowerCase()) {
                case 'gradual':
                    return 0;
                case 'moderate':
                    return 1;
                case 'quick':
                    return 2;
                case 'rapid':
                    return 3;
                default:
                    return null; // Return null if rapidity is not recognized
            }
        };

        // Add throttle guidance - only when change_rate indicates a position change is needed
        if (actions.optimal_throttle && sentences.throttle_guidance && actions.optimal_throttle.change_rate > 0) {
            const rapidityIndex = getSentenceIndex(actions.optimal_throttle.rapidity);
            if (rapidityIndex !== null && sentences.throttle_guidance[rapidityIndex]) {
                guidanceTexts.push(sentences.throttle_guidance[rapidityIndex]);
            }
        }

        // Add brake guidance - only when change_rate indicates a position change is needed
        if (actions.optimal_brake && sentences.brake_guidance && actions.optimal_brake.change_rate > 0) {
            const rapidityIndex = getSentenceIndex(actions.optimal_brake.rapidity);
            if (rapidityIndex !== null && sentences.brake_guidance[rapidityIndex]) {
                guidanceTexts.push(sentences.brake_guidance[rapidityIndex]);
            }
        }

        // Add steering guidance
        if (actions.optimal_steering && sentences.steering_guidance) {
            const rapidityIndex = getSentenceIndex(actions.optimal_steering.rapidity);
            if (rapidityIndex !== null && sentences.steering_guidance[rapidityIndex]) {
                guidanceTexts.push(sentences.steering_guidance[rapidityIndex]);
            }
        }

        const generatedText = guidanceTexts.join('. ') + (guidanceTexts.length ? '.' : '');

        // Send the generated guidance text to AI chat if it's not empty and different from the last one
        if (generatedText && analysisContext?.sendGuidanceToChat) {
            // Include phase information for context
            const fullGuidanceMessage = `${generatedText}`;

            // Only send if different from the last sent message to avoid spam
            if (fullGuidanceMessage !== lastSentGuidanceMessage) {
                analysisContext.sendGuidanceToChat(fullGuidanceMessage);
                setLastSentGuidanceMessage(fullGuidanceMessage);
            }
        }

        return generatedText;
    }, [trackGuidanceData, analysisContext, lastSentGuidanceMessage]);

    // Function to check if current position is within a phase
    const isPositionWithinPhase = useCallback((position: number, phaseData: any, nextPhaseData: any): boolean => {
        if (!phaseData?.phase_position) return false;

        const currentPhasePos = phaseData.phase_position;
        const nextPhasePos = nextPhaseData?.phase_position;

        // If there's no next phase data, use a tolerance around current phase
        if (!nextPhasePos) {
            const tolerance = 0.05; // 5% tolerance
            return Math.abs(position - currentPhasePos) <= tolerance;
        }

        // Simple linear check since lap wrapping is handled by completed lap count
        return position >= currentPhasePos && position < nextPhasePos;
    }, []);

    // Helper function to create CurrentGuidance from currentLocation
    const createGuidanceFromLocation = useCallback((currentLocation: any, normalizedPosition: number): CurrentGuidance => {
        const guidanceData = currentLocation.phaseData;
        const actions = (guidanceData as any).optimal_actions;
        const humanText = generateHumanReadableText(actions, currentLocation.phase);

        return {
            corner: currentLocation.corner,
            phase: currentLocation.phase,
            phaseData: currentLocation.phaseData,
            position: normalizedPosition,
            actions: {
                throttle: (actions.optimal_throttle && actions.optimal_throttle.change_rate > 0)
                    ? actions.optimal_throttle.description
                    : 'N/A',
                brake: (actions.optimal_brake && actions.optimal_brake.change_rate > 0)
                    ? actions.optimal_brake.description
                    : 'N/A',
                steering: actions.optimal_steering?.description || 'N/A',
                speed: actions.optimal_speed?.description || 'N/A'
            },
            actionDetails: {
                throttle: actions.optimal_throttle ? {
                    changeRate: actions.optimal_throttle.change_rate || 0,
                    value: actions.optimal_throttle.value || 0,
                    rapidity: actions.optimal_throttle.rapidity || 'N/A'
                } : null,
                brake: actions.optimal_brake ? {
                    changeRate: actions.optimal_brake.change_rate || 0,
                    value: actions.optimal_brake.value || 0,
                    rapidity: actions.optimal_brake.rapidity || 'N/A'
                } : null,
                steering: actions.optimal_steering ? {
                    changeRate: actions.optimal_steering.change_rate || 0,
                    value: actions.optimal_steering.value || 0,
                    rapidity: actions.optimal_steering.rapidity || 'N/A'
                } : null,
                speed: actions.optimal_speed ? {
                    changeRate: actions.optimal_speed.change_rate || 0,
                    value: actions.optimal_speed.value || 0,
                    rapidity: actions.optimal_speed.rapidity || 'N/A'
                } : null
            },
            confidence: (guidanceData as any).confidence || 0,
            humanReadableText: humanText,
            nextPhase: currentLocation.nextPhase || undefined
        };
    }, [generateHumanReadableText]);

    // Main effect to monitor live telemetry data and update guidance
    useEffect(() => {
        if (!analysisContext?.liveData || !trackGuidanceData) return;

        const telemetryData = analysisContext.liveData;
        const normalizedPosition = normalizeCarPosition(telemetryData);
        const currentTime = Date.now();
        const currentCompletedLaps = telemetryData.Graphics_completed_lap || 0;

        // Throttle updates - only update every 100ms to prevent excessive re-renders
        if (currentTime - lastGuidanceUpdateRef.current < 100) {
            return;
        }
        lastGuidanceUpdateRef.current = currentTime;

        // Check if position has changed significantly
        const positionChange = Math.abs(normalizedPosition - lastPositionRef.current);
        lastPositionRef.current = normalizedPosition;

        // Detect lap change first to reset corner tracking immediately
        if (currentCompletedLaps > lastCompletedLap) {
            // New lap detected - reset to corner 1 immediately
            setCurrentCornerIndex(1);
            setLastCompletedLap(currentCompletedLaps);
            lastCornerRef.current = 1;
        }

        // Update position history for approach speed calculation
        setPositionHistory(prev => {
            const newHistory = [...prev, { position: normalizedPosition, timestamp: currentTime, lap: currentCompletedLaps }];
            // Keep only last 3 seconds of data
            const fiveSecondsAgo = currentTime - 3000;
            return newHistory.filter(entry => entry.timestamp > fiveSecondsAgo);
        });

        // Calculate current approach speed to next phase
        const approachSpeed = calculateApproachSpeed(positionHistory);
        setCurrentSpeed(approachSpeed);

        // Only update guidance if there's meaningful change or no current guidance exists
        const needsUpdate = !currentGuidance ||
            positionChange > 0.001 || // Position changed significantly
            currentCornerIndex !== lastCornerRef.current; // Corner changed

        if (needsUpdate) {
            // Only update guidance if driver is not in the current phase (or no current guidance exists)
            const isInCurrentPhase = currentGuidance ?
                isPositionWithinPhase(normalizedPosition, currentGuidance.phaseData, currentGuidance.nextPhase?.data) :
                false;

            console.log(`Position check: ${normalizedPosition.toFixed(3)}, lap: ${currentCompletedLaps}, currentCorner: ${currentCornerIndex}, isInCurrentPhase: ${isInCurrentPhase}, currentPhase: ${currentGuidance?.phase || 'none'}`);

            if (!currentGuidance || !isInCurrentPhase) {
                const currentLocation = findCurrentCornerAndPhase(normalizedPosition, approachSpeed, currentCompletedLaps);
                console.log('Updating guidance. New location:', currentLocation);

                if (currentLocation) {
                    const guidance = createGuidanceFromLocation(currentLocation, normalizedPosition);
                    setCurrentGuidance(guidance);
                    setCurrentPhaseData(currentLocation); // Keep for backward compatibility with isPositionWithinPhase

                    // Log corner transition for debugging
                    if (currentGuidance && currentGuidance.corner !== currentLocation.corner) {
                        console.log(`Corner transition: ${currentGuidance.corner} -> ${currentLocation.corner}`);
                    }
                }
            }
            lastCornerRef.current = currentCornerIndex;
        }
        // Keep showing the last guidance when no current location is found (on straights)
    }, [analysisContext?.liveData, trackGuidanceData]);

    if (isLoading) {
        return (
            <Card className={styles.chartCard} style={{ width, height }}>
                <div className={styles.loadingContainer}>
                    <Text>Loading guidance data...</Text>
                </div>
            </Card>
        );
    }

    if (!trackGuidanceData) {
        return (
            <Card className={styles.chartCard} style={{ width, height }}>
                <Text size="2">Track guidance not available. Enable track guidance through AI chat.</Text>
            </Card>
        );
    }

    return (
        <Card className={styles.chartCard} style={{ width, height }}>
            <Box className={styles.chartHeader}>
                <Text size="4" weight="bold">Real-Time Track Guidance</Text>
                <Text size="2" color="gray">Live telemetry-based driving guidance</Text>
            </Box>

            <Separator className={styles.separator} />

            <Box className={styles.contentContainer}>
                {currentGuidance ? (
                    <>
                        <Box className={styles.behaviorSection}>
                            <Text className={styles.sectionTitle} size="3" weight="medium">

                                `Last Known Location: {currentGuidance.corner === 0 ? currentGuidance.phase : `Corner ${currentGuidance.corner} - ${currentGuidance.phase}`}`
                            </Text>

                            <Box className={styles.confidenceContainer}>
                                <Flex justify="between" align="center" className={styles.confidenceBadges}>
                                    <Text size="2">Position: {(currentGuidance.position * 100).toFixed(1)}%</Text>
                                    <Badge color="blue">
                                        Confidence: {(currentGuidance.confidence * 100).toFixed(0)}%
                                    </Badge>
                                </Flex>
                                <Progress
                                    value={currentGuidance.confidence * 100}
                                    className={styles.confidenceProgress}
                                />
                                {/* Next Phase Info - Always visible when available */}
                                {currentGuidance.nextPhase && (
                                    <Box style={{ marginTop: '8px' }}>
                                        <Flex justify="between" align="center">
                                            <Text size="2" color="gray">Next Phase:</Text>
                                            <Text size="2" weight="medium" color="blue">
                                                {currentGuidance.nextPhase.corner !== currentGuidance.corner
                                                    ? `Corner ${currentGuidance.nextPhase.corner} - ${currentGuidance.nextPhase.name}`
                                                    : currentGuidance.nextPhase.name}
                                            </Text>
                                        </Flex>
                                        <Flex justify="between" align="center" style={{ marginTop: '4px' }}>
                                            <Text size="1" color="gray">
                                                Distance: {(currentGuidance.nextPhase.distance * 100).toFixed(1)}%
                                            </Text>
                                            <Text size="1" color="gray">
                                                ETA: {currentGuidance.nextPhase.timeToReach < 3
                                                    ? `${currentGuidance.nextPhase.timeToReach.toFixed(1)}s`
                                                    : 'Soon'}
                                            </Text>
                                        </Flex>
                                    </Box>
                                )}
                            </Box>
                        </Box>

                        <Separator className={styles.separator} />

                        <Box className={styles.behaviorSection}>
                            <Text className={styles.sectionTitle} size="3" weight="medium">
                                AI Guidance
                            </Text>
                            <Box style={{
                                padding: '12px',
                                backgroundColor: 'var(--blue-2)',
                                borderRadius: '8px',
                                marginBottom: '16px'
                            }}>
                                <Text size="2" style={{ lineHeight: '1.6' }}>
                                    {currentGuidance.humanReadableText ||
                                        `${currentGuidance.phase}: Follow the recommended actions for optimal performance.`}
                                </Text>
                            </Box>

                            {/* Next Phase Section - show when timeToReach <= 10s AND we're not already showing next phase guidance */}
                            {currentGuidance.nextPhase &&
                                currentGuidance.nextPhase.timeToReach <= 10 &&
                                !currentGuidance.phase.startsWith('⚠️ NOW:') &&
                                !currentGuidance.phase.startsWith('🔄 Next:') && (
                                    <Box style={{
                                        padding: '12px',
                                        backgroundColor: currentGuidance.nextPhase.timeToReach < 5 ? 'var(--orange-2)' : 'var(--green-2)',
                                        borderRadius: '8px',
                                        marginTop: '12px',
                                        border: currentGuidance.nextPhase.timeToReach < 3 ? '2px solid var(--orange-7)' : '1px solid var(--green-7)'
                                    }}>
                                        <Flex justify="between" align="center" style={{ marginBottom: '8px' }}>
                                            <Text size="2" weight="medium">
                                                📍 Next Phase
                                            </Text>
                                            <Badge color={currentGuidance.nextPhase.timeToReach < 5 ? "orange" : "green"}>
                                                {currentGuidance.nextPhase.timeToReach < 60 ?
                                                    `${currentGuidance.nextPhase.timeToReach.toFixed(1)}s` :
                                                    'Soon'
                                                }
                                            </Badge>
                                        </Flex>
                                        <Text size="2" style={{ lineHeight: '1.6' }}>
                                            Next: {currentGuidance.nextPhase.name}
                                        </Text>
                                        <Box style={{ marginTop: '8px' }}>
                                            <Grid columns="3" gap="2">
                                                <Box>
                                                    <Text size="1" color="gray">Next Phase</Text>
                                                    <Text size="1" style={{ display: 'block', fontWeight: '500' }}>
                                                        {currentGuidance.nextPhase.name}
                                                    </Text>
                                                </Box>
                                                <Box>
                                                    <Text size="1" color="gray">Distance</Text>
                                                    <Text size="1" style={{ display: 'block', fontWeight: '500' }}>
                                                        {(currentGuidance.nextPhase.distance * 100).toFixed(1)}%
                                                    </Text>
                                                </Box>
                                            </Grid>
                                        </Box>
                                    </Box>
                                )}
                        </Box>

                        <Separator className={styles.separator} />

                        <Box className={styles.actionComparison}>
                            <Text className={styles.comparisonTitle} size="3" weight="medium">
                                Recommended Actions
                            </Text>
                            <Box className={styles.comparisonContainer}>
                                <Grid columns="2" gap="3">
                                    <Box className={styles.comparisonBox}>
                                        <Text className={styles.comparisonBoxTitle} size="2" weight="medium">
                                            🚗 Throttle
                                        </Text>
                                        <Box className={styles.comparisonValues}>
                                            <Text className={styles.comparisonValue} size="2">
                                                {currentGuidance.actions.throttle}
                                            </Text>
                                            {currentGuidance.actionDetails.throttle && (
                                                <Text size="1" color="gray" style={{ marginTop: '4px' }}>
                                                    Change Rate: {currentGuidance.actionDetails.throttle.changeRate.toFixed(10)}
                                                    {currentGuidance.actionDetails.throttle.changeRate > 0 && (
                                                        <Text size="1" color="blue" style={{ marginLeft: '8px' }}>
                                                            ({currentGuidance.actionDetails.throttle.rapidity})
                                                        </Text>
                                                    )}
                                                </Text>
                                            )}
                                        </Box>
                                    </Box>

                                    <Box className={styles.comparisonBox}>
                                        <Text className={styles.comparisonBoxTitle} size="2" weight="medium">
                                            🛑 Brake
                                        </Text>
                                        <Box className={styles.comparisonValues}>
                                            <Text className={styles.comparisonValue} size="2">
                                                {currentGuidance.actions.brake}
                                            </Text>
                                            {currentGuidance.actionDetails.brake && (
                                                <Text size="1" color="gray" style={{ marginTop: '4px' }}>
                                                    Change Rate: {currentGuidance.actionDetails.brake.changeRate.toFixed(10)}
                                                    {currentGuidance.actionDetails.brake.changeRate > 0 && (
                                                        <Text size="1" color="blue" style={{ marginLeft: '8px' }}>
                                                            ({currentGuidance.actionDetails.brake.rapidity})
                                                        </Text>
                                                    )}
                                                </Text>
                                            )}
                                        </Box>
                                    </Box>

                                    <Box className={styles.comparisonBox}>
                                        <Text className={styles.comparisonBoxTitle} size="2" weight="medium">
                                            🎯 Steering
                                        </Text>
                                        <Box className={styles.comparisonValues}>
                                            <Text className={styles.comparisonValue} size="2">
                                                {currentGuidance.actions.steering}
                                            </Text>
                                            {currentGuidance.actionDetails.steering && (
                                                <Text size="1" color="gray" style={{ marginTop: '4px' }}>
                                                    Change Rate: {currentGuidance.actionDetails.steering.changeRate.toFixed(10)}
                                                    {currentGuidance.actionDetails.steering.changeRate > 0 && (
                                                        <Text size="1" color="blue" style={{ marginLeft: '8px' }}>
                                                            ({currentGuidance.actionDetails.steering.rapidity})
                                                        </Text>
                                                    )}
                                                </Text>
                                            )}
                                        </Box>
                                    </Box>

                                    <Box className={styles.comparisonBox}>
                                        <Text className={styles.comparisonBoxTitle} size="2" weight="medium">
                                            ⚡ Speed
                                        </Text>
                                        <Box className={styles.comparisonValues}>
                                            <Text className={styles.comparisonValue} size="2">
                                                {currentGuidance.actions.speed}
                                            </Text>
                                            {currentGuidance.actionDetails.speed && (
                                                <Text size="1" color="gray" style={{ marginTop: '4px' }}>
                                                    Change Rate: {currentGuidance.actionDetails.speed.changeRate.toFixed(10)}
                                                    {currentGuidance.actionDetails.speed.changeRate > 0 && (
                                                        <Text size="1" color="blue" style={{ marginLeft: '8px' }}>
                                                            ({currentGuidance.actionDetails.speed.rapidity})
                                                        </Text>
                                                    )}
                                                </Text>
                                            )}
                                        </Box>
                                    </Box>
                                </Grid>
                            </Box>
                        </Box>
                    </>
                ) : (
                    <Box style={{ textAlign: 'center', padding: '20px' }}>
                        <Text size="3" color="gray">
                            No specific guidance available for current track position
                        </Text>
                        <Text size="2" color="gray" style={{ display: 'block', marginTop: '8px' }}>
                            Continue driving to receive AI-powered guidance
                        </Text>

                        {/* Debug information */}
                        {analysisContext?.liveData && (
                            <Box style={{
                                marginTop: '16px',
                                padding: '12px',
                                backgroundColor: 'var(--gray-2)',
                                borderRadius: '8px',
                                textAlign: 'left'
                            }}>
                                <Text size="1" weight="medium" style={{ display: 'block', marginBottom: '8px' }}>
                                    Debug Info:
                                </Text>
                                <Text size="1" style={{ display: 'block', fontFamily: 'monospace' }}>
                                    Position: {normalizeCarPosition(analysisContext.liveData).toFixed(4)}
                                </Text>
                                <Text size="1" style={{ display: 'block', fontFamily: 'monospace' }}>
                                    Approach Speed: {(currentSpeed * 100).toFixed(2)}%/s
                                </Text>
                                <Text size="1" style={{ display: 'block', fontFamily: 'monospace' }}>
                                    Position History: {positionHistory.length} points
                                </Text>
                                <Text size="1" style={{ display: 'block', fontFamily: 'monospace' }}>
                                    Available data: {Object.keys(analysisContext.liveData).join(', ')}
                                </Text>
                                {trackGuidanceData?.trackData?._prediction_result && (
                                    <Text size="1" style={{ display: 'block', fontFamily: 'monospace' }}>
                                        Corners: {trackGuidanceData.trackData._prediction_result.total_corners}
                                    </Text>
                                )}
                            </Box>
                        )}
                    </Box>
                )}
            </Box>
        </Card>
    );
};

export default ImitationGuidanceChart;
