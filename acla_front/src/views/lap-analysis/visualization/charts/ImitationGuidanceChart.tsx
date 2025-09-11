import React, { useContext, useState, useEffect } from 'react';
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
    position: number;
    actions: {
        throttle: string;
        brake: string;
        steering: string;
        speed: string;
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

    // Store corner boundaries for quick lookup
    const [cornerBoundaries, setCornerBoundaries] = useState<Array<{ cornerNum: number, start: number, end: number }>>([]);

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
    const normalizeCarPosition = (telemetryData: any): number => {
        if (!telemetryData) {
            return 0;
        }

        // Use the normalized position that's always available in telemetry data
        const position = telemetryData.Graphics_normalized_car_position || 0;
        return Math.max(0, Math.min(1, position));
    };

    // Function to calculate approach speed based on position history
    const calculateApproachSpeed = (history: Array<{ position: number, timestamp: number, lap: number }>): number => {
        if (history.length < 3) return 0;

        const recentHistory = history.slice(-5);
        if (recentHistory.length < 3) return 0;

        const validDistances = recentHistory.slice(1).map((curr, i) => {
            const prev = recentHistory[i];
            let distance = curr.position - prev.position;

            // Handle lap changes and wrap-around
            if (curr.lap !== prev.lap) {
                distance += (curr.lap - prev.lap) * 1.0;
            } else if (Math.abs(distance) > 0.5) {
                return null; // Mark telemetry glitches
            }

            return {
                distance: Math.abs(distance),
                time: curr.timestamp - prev.timestamp
            };
        }).filter((d): d is { distance: number; time: number } => d !== null);

        const totalDistance = validDistances.reduce((sum, d) => sum + d.distance, 0);
        const totalTime = validDistances.reduce((sum, d) => sum + d.time, 0);

        return totalTime > 0 ? totalDistance / (totalTime / 1000) : 0;
    };

    // Function to find the current corner and phase based on position with optimized corner tracking
    const findCurrentCornerAndPhase = (position: number, approachSpeed: number = 0) => {
        if (!trackGuidanceData?.trackData?._prediction_result?.corner_predictions || cornerBoundaries.length === 0) {
            console.log('No corner predictions or boundaries available');
            return null;
        }

        const cornerPredictions = trackGuidanceData.trackData._prediction_result.corner_predictions;

        // Helper function to find phase and build result for a given corner
        const buildCornerResult = (cornerIndex: number) => {
            const cornerData = cornerPredictions[`corner_${cornerIndex}`];
            if (!cornerData) return null;

            const phaseResult = findPhaseInCorner(cornerData, position);
            if (!phaseResult) return null;

            const nextPhaseInfo = getNextPhaseInfo(cornerIndex, phaseResult.phaseName, position, approachSpeed);
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
        const getNextPhaseInfo = (cornerNum: number, currentPhaseName: string, position: number, approachSpeed: number) => {
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
            let distance = nextPhasePosition - position;
            if (distance < 0) distance += 1.0; // Handle wrap-around

            const timeToReach = approachSpeed > 0 ? distance / approachSpeed : Infinity;

            return {
                corner: nextPhaseCorner,
                name: nextPhaseName,
                data: nextPhaseData,
                distance: distance,
                timeToReach: timeToReach,
            };
        };

        // Get current corner boundary
        const currentCornerBoundary = cornerBoundaries.find(b => b.cornerNum === currentCornerIndex);
        if (!currentCornerBoundary) {
            console.log('Current corner boundary not found');
            return null;
        }

        // Check if position is within current corner
        if (position >= currentCornerBoundary.start && position <= currentCornerBoundary.end) {
            return buildCornerResult(currentCornerIndex);
        }

        // First, check if we're in any corner boundary at all
        let foundCorner = null;
        for (const boundary of cornerBoundaries) {
            if (position >= boundary.start && position <= boundary.end) {
                foundCorner = boundary.cornerNum;
                break;
            }
        }

        // Only update currentCornerIndex if we found a corner and it's different from current
        if (foundCorner !== null && foundCorner !== currentCornerIndex) {
            setCurrentCornerIndex(foundCorner);
            return buildCornerResult(foundCorner);
        } else if (foundCorner !== null) {
            // We're in the same corner, use current corner index
            return buildCornerResult(currentCornerIndex);
        }

        // Position is between corners (straight section) - don't update currentCornerIndex
        return null;
    };

    // Function to generate human-readable guidance text
    const generateHumanReadableText = (actions: any, phase: string): string => {
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
                case 'gradually':
                    return 0;
                case 'moderately':
                    return 1;
                case 'quickly':
                    return 2;
                case 'rapidly':
                    return 3;
                default:
                    return null; // Return null if rapidity is not recognized
            }
        };

        // Add throttle guidance
        if (actions.optimal_throttle && sentences.throttle_guidance) {
            const rapidityIndex = getSentenceIndex(actions.optimal_throttle.rapidity);
            if (rapidityIndex !== null && sentences.throttle_guidance[rapidityIndex]) {
                guidanceTexts.push(sentences.throttle_guidance[rapidityIndex]);
            }
        }

        // Add brake guidance
        if (actions.optimal_brake && sentences.brake_guidance) {
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
            const fullGuidanceMessage = `${phase}: ${generatedText}`;

            // Only send if different from the last sent message to avoid spam
            if (fullGuidanceMessage !== lastSentGuidanceMessage) {
                analysisContext.sendGuidanceToChat(fullGuidanceMessage);
                setLastSentGuidanceMessage(fullGuidanceMessage);
            }
        }

        return generatedText;
    };

    // Main effect to monitor live telemetry data and update guidance
    useEffect(() => {
        if (!analysisContext?.liveData || !trackGuidanceData) return;

        const telemetryData = analysisContext.liveData;
        const normalizedPosition = normalizeCarPosition(telemetryData);
        const currentTime = Date.now();
        const currentCompletedLaps = telemetryData.Graphics_completed_lap || 0;

        // Detect lap change first to reset corner tracking immediately
        if (currentCompletedLaps > lastCompletedLap) {
            // New lap detected - reset to corner 1 immediately
            setCurrentCornerIndex(1);
            setLastCompletedLap(currentCompletedLaps);
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

        // Find current phase and next phase
        const currentLocation = findCurrentCornerAndPhase(normalizedPosition, approachSpeed);

        if (currentLocation) {
            setIsInActivePhase(true);

            // Determine if we should show next phase guidance
            const shouldUseNextPhase = currentLocation.nextPhase &&
                currentLocation.nextPhase.timeToReach > 0 &&
                currentLocation.nextPhase.timeToReach <= GUIDANCE_TRANSITION_THRESHOLD &&
                currentLocation.nextPhase.data;

            // Use next phase data for guidance if conditions are met, otherwise use current phase
            const guidanceData = shouldUseNextPhase ? currentLocation.nextPhase!.data : currentLocation.phaseData;
            const isNextPhaseGuidance = shouldUseNextPhase;
            const isUrgentTransition = shouldUseNextPhase && currentLocation.nextPhase!.timeToReach <= URGENT_TRANSITION_THRESHOLD;

            const actions = (guidanceData as any).optimal_actions;
            const humanText = generateHumanReadableText(actions, shouldUseNextPhase ? currentLocation.nextPhase!.name : currentLocation.phase);

            const guidance: CurrentGuidance = {
                corner: shouldUseNextPhase ? currentLocation.nextPhase!.corner : currentLocation.corner,
                phase: shouldUseNextPhase ? currentLocation.nextPhase!.name : currentLocation.phase,
                position: normalizedPosition,
                actions: {
                    throttle: actions.optimal_throttle?.description || 'N/A',
                    brake: actions.optimal_brake?.description || 'N/A',
                    steering: actions.optimal_steering?.description || 'N/A',
                    speed: actions.optimal_speed?.description || 'N/A'
                },
                confidence: (guidanceData as any).confidence || 0,
                humanReadableText: isNextPhaseGuidance
                    ? (isUrgentTransition
                        ? `⚠️ URGENT: Prepare now! ${humanText}`
                        : `🔄 Prepare for upcoming phase: ${humanText}`)
                    : humanText,
                nextPhase: currentLocation.nextPhase || undefined
            };

            setCurrentGuidance(guidance);
        } else {
            setIsInActivePhase(false);
        }
        // Keep showing the last guidance when no current location is found (on straights)
    }, [analysisContext?.liveData, trackGuidanceData, positionHistory]);

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
                                {isInActivePhase
                                    ? (currentGuidance.phase.startsWith('⚠️ NOW:') || currentGuidance.phase.startsWith('🔄 Next:')
                                        ? `Guidance Active: ${currentGuidance.phase}`
                                        : `Current Location: ${currentGuidance.corner === 0 ? currentGuidance.phase : `Corner ${currentGuidance.corner} - ${currentGuidance.phase}`}`)
                                    : `Last Known Location: ${currentGuidance.corner === 0 ? currentGuidance.phase : `Corner ${currentGuidance.corner} - ${currentGuidance.phase}`} (On Straight)`
                                }
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
