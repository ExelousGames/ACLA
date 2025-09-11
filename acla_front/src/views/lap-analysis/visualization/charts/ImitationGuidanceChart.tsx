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
    phase: string;
    corner: number;
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
        name: string;
        data?: any;
        distance: number;
        timeToReach: number;
        approachSpeed: number;
    };
}

// Configuration constants
const GUIDANCE_TRANSITION_THRESHOLD = 5; // seconds before switching to next phase guidance
const URGENT_TRANSITION_THRESHOLD = 3; // seconds for urgent next phase warnings

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
    const calculateApproachSpeed = (currentPosition: number, history: Array<{ position: number, timestamp: number, lap: number }>): number => {
        if (history.length < 3) return 0; // Need at least 3 data points for reliable calculation

        // Use last 5 data points for better accuracy
        const recentHistory = history.slice(-5);
        if (recentHistory.length < 3) return 0;

        let totalDistance = 0;
        let totalTime = 0;

        for (let i = 1; i < recentHistory.length; i++) {
            const prevPoint = recentHistory[i - 1];
            const currPoint = recentHistory[i];

            let distance = currPoint.position - prevPoint.position;

            // Handle lap wrap-around using lap counter
            const lapDifference = currPoint.lap - prevPoint.lap;
            if (lapDifference > 0) {
                // Crossed finish line - add full lap distance
                distance += lapDifference * 1.0;
            } else if (lapDifference < 0) {
                // Went backwards across finish line (rare case)
                distance -= Math.abs(lapDifference) * 1.0;
            } else if (Math.abs(distance) > 0.5) {
                // Same lap but large position jump - likely telemetry glitch, skip this point
                continue;
            }

            totalDistance += Math.abs(distance);
            totalTime += currPoint.timestamp - prevPoint.timestamp;
        }

        return totalTime > 0 ? totalDistance / (totalTime / 1000) : 0; // positions per second
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
                    name: nextPhaseInfo.name,
                    data: nextPhaseInfo.data,
                    distance: nextPhaseInfo.distance,
                    timeToReach: nextPhaseInfo.timeToReach,
                    approachSpeed: nextPhaseInfo.approachSpeed
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

            if (currentPhaseIndex >= 0 && currentPhaseIndex < phases.length - 1) {
                // Next phase is in same corner
                const [name, data] = phases[currentPhaseIndex + 1];
                nextPhaseName = name;
                nextPhaseData = data;
                nextPhasePosition = (data as any).phase_position;
            } else {
                // Look for next corner's first phase
                const totalCorners = Object.keys(cornerPredictions).length;
                const nextCornerNum = cornerNum >= totalCorners ? 1 : cornerNum + 1;
                const nextCornerData = cornerPredictions[`corner_${nextCornerNum}`];

                if (nextCornerData) {
                    const nextCornerPhases = Object.entries(nextCornerData.phases)
                        .filter(([_, phaseData]: [string, any]) => phaseData.phase_position !== undefined);

                    if (nextCornerPhases.length > 0) {
                        const [name, data] = nextCornerPhases[0];
                        nextPhaseName = `Corner ${nextCornerNum} - ${name}`;
                        nextPhaseData = data;
                        nextPhasePosition = (data as any).phase_position;
                    }
                }
            }

            if (!nextPhaseName || !nextPhaseData) return null;

            let distance = nextPhasePosition - position;
            if (distance < 0) distance += 1.0; // Handle wrap-around

            const timeToReach = approachSpeed > 0 ? distance / approachSpeed : Infinity;

            return {
                name: nextPhaseName,
                data: nextPhaseData,
                distance: distance,
                timeToReach: timeToReach,
                approachSpeed: approachSpeed
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

        // Position is outside current corner - check adjacent corners
        const totalCorners = cornerBoundaries.length;
        const cornersToCheck = [
            (currentCornerIndex % totalCorners) + 1, // Next corner
            currentCornerIndex === 1 ? totalCorners : currentCornerIndex - 1 // Previous corner
        ];

        for (const cornerIndex of cornersToCheck) {
            const boundary = cornerBoundaries.find(b => b.cornerNum === cornerIndex);
            if (boundary && position >= boundary.start && position <= boundary.end) {
                setCurrentCornerIndex(cornerIndex);
                return buildCornerResult(cornerIndex);
            }
        }

        // Position is between corners (straight section)
        return null;
    };

    // Function to generate human-readable guidance text
    const generateHumanReadableText = (actions: any, phase: string): string => {
        const actionTypes = ['optimal_throttle', 'optimal_brake', 'optimal_steering', 'optimal_speed'];

        // Handle straight sections and approaching corners with direct descriptions
        if (phase.includes('Straight') || phase.includes('Approaching')) {
            const descriptions = actionTypes
                .filter(type => actions[type]?.description && (type !== 'optimal_brake' || actions[type].value > 0))
                .map(type => actions[type].description);
            return descriptions.join('. ') + (descriptions.length ? '.' : '');
        }

        // For corner phases, use preloaded sentences if available
        if (!trackGuidanceData?.preloadSentences) {
            const descriptions = actionTypes
                .map(type => actions[type]?.description)
                .filter(Boolean);
            return descriptions.join('. ') + (descriptions.length ? '.' : '');
        }

        const sentences = trackGuidanceData.preloadSentences;
        const guidanceTexts = [];

        // Select appropriate guidance based on current actions and phase
        if (actions.optimal_throttle && sentences.throttle_guidance) {
            const index = Math.min(
                Math.floor(actions.optimal_throttle.value * sentences.throttle_guidance.length),
                sentences.throttle_guidance.length - 1
            );
            guidanceTexts.push(sentences.throttle_guidance[index]);
        }

        if (actions.optimal_brake && sentences.brake_guidance) {
            const index = Math.min(
                Math.floor(actions.optimal_brake.value * sentences.brake_guidance.length),
                sentences.brake_guidance.length - 1
            );
            guidanceTexts.push(sentences.brake_guidance[index]);
        }

        if (actions.optimal_steering && sentences.steering_guidance) {
            const index = Math.min(
                Math.floor(Math.abs(actions.optimal_steering.value) * sentences.steering_guidance.length),
                sentences.steering_guidance.length - 1
            );
            guidanceTexts.push(sentences.steering_guidance[index]);
        }

        return guidanceTexts.join('. ') + (guidanceTexts.length ? '.' : '');
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
            // Keep only last 5 seconds of data
            const fiveSecondsAgo = currentTime - 5000;
            return newHistory.filter(entry => entry.timestamp > fiveSecondsAgo);
        });

        // Calculate current approach speed to next phase
        const approachSpeed = calculateApproachSpeed(normalizedPosition, positionHistory);
        setCurrentSpeed(approachSpeed);

        // Find current corner and phase with approach speed
        const currentLocation = findCurrentCornerAndPhase(normalizedPosition, approachSpeed);

        if (currentLocation) {
            setIsInActivePhase(true);

            // Determine guidance phase based on approach speed
            const shouldUseNextPhase = currentLocation.nextPhase &&
                currentLocation.nextPhase.timeToReach > 0 &&
                currentLocation.nextPhase.timeToReach <= GUIDANCE_TRANSITION_THRESHOLD &&
                currentLocation.nextPhase.data;

            const {
                phase: guidancePhase,
                data: guidancePhaseData,
                isNext: isNextPhaseGuidance,
                isUrgent: isUrgentTransition
            } = shouldUseNextPhase
                    ? {
                        phase: currentLocation.nextPhase!.name,
                        data: currentLocation.nextPhase!.data,
                        isNext: true,
                        isUrgent: currentLocation.nextPhase!.timeToReach <= URGENT_TRANSITION_THRESHOLD
                    }
                    : {
                        phase: currentLocation.phase,
                        data: currentLocation.phaseData,
                        isNext: false,
                        isUrgent: false
                    };

            const actions = (guidancePhaseData as any).optimal_actions;
            const humanText = generateHumanReadableText(actions, guidancePhase);

            const guidance: CurrentGuidance = {
                phase: isNextPhaseGuidance
                    ? (isUrgentTransition ? `⚠️ NOW: ${guidancePhase}` : `🔄 Next: ${guidancePhase}`)
                    : guidancePhase,
                corner: currentLocation.corner,
                position: normalizedPosition,
                actions: {
                    throttle: actions.optimal_throttle?.description || 'N/A',
                    brake: actions.optimal_brake?.description || 'N/A',
                    steering: actions.optimal_steering?.description || 'N/A',
                    speed: actions.optimal_speed?.description || 'N/A'
                },
                confidence: (guidancePhaseData as any).confidence || 0,
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
                                                <Box>
                                                    <Text size="1" color="gray">Approach Speed</Text>
                                                    <Text size="1" style={{ display: 'block', fontWeight: '500' }}>
                                                        {(currentGuidance.nextPhase.approachSpeed * 100).toFixed(1)}%/s
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
