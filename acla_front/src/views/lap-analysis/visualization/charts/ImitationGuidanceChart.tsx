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
        distance: number;
        timeToReach: number;
        approachSpeed: number;
        predictiveGuidance: string;
    };
}

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
    const [positionHistory, setPositionHistory] = useState<Array<{ position: number, timestamp: number }>>([]);
    const [currentSpeed, setCurrentSpeed] = useState<number>(0);
    const [currentCornerIndex, setCurrentCornerIndex] = useState<number>(1); // Track which corner we're currently monitoring
    const [lastCompletedLap, setLastCompletedLap] = useState<number>(0); // Track last completed lap for detection

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
    const calculateApproachSpeed = (currentPosition: number, history: Array<{ position: number, timestamp: number }>): number => {
        if (history.length < 2) return 0;

        // Use last 5 data points for better accuracy
        const recentHistory = history.slice(-5);
        if (recentHistory.length < 2) return 0;

        let totalDistance = 0;
        let totalTime = 0;

        for (let i = 1; i < recentHistory.length; i++) {
            const prevPoint = recentHistory[i - 1];
            const currPoint = recentHistory[i];

            let distance = currPoint.position - prevPoint.position;

            // Handle lap wrap-around
            if (distance < -0.5) {
                distance += 1.0; // Crossed finish line
            } else if (distance > 0.5) {
                distance -= 1.0; // Went backwards across finish line
            }

            totalDistance += Math.abs(distance);
            totalTime += currPoint.timestamp - prevPoint.timestamp;
        }

        return totalTime > 0 ? totalDistance / (totalTime / 1000) : 0; // positions per second
    };

    // Function to find the next phase and calculate approach metrics
    const findNextPhase = (currentCorner: number, currentPhase: string, currentPosition: number, approachSpeed: number) => {
        if (!trackGuidanceData?.trackData?._prediction_result?.corner_predictions) return null;

        const cornerPredictions = trackGuidanceData.trackData._prediction_result.corner_predictions;

        // Get current corner data
        const currentCornerData = cornerPredictions[`corner_${currentCorner}`];
        if (!currentCornerData) return null;

        // Get all phases in current corner (already in correct order)
        const phases = Object.entries(currentCornerData.phases)
            .filter(([_, phaseData]: [string, any]) => phaseData.phase_position !== undefined);

        // Find current phase index
        const currentPhaseIndex = phases.findIndex(([name, _]) => name === currentPhase);

        let nextPhase = null;
        let nextPhasePosition = 0;

        if (currentPhaseIndex >= 0 && currentPhaseIndex < phases.length - 1) {
            // Next phase is in same corner
            const [nextPhaseName, nextPhaseData] = phases[currentPhaseIndex + 1];
            nextPhase = nextPhaseName;
            nextPhasePosition = (nextPhaseData as any).phase_position;
        } else {
            // Look for next corner's first phase
            const totalCorners = Object.keys(cornerPredictions).length;
            const nextCornerNum = currentCorner >= totalCorners ? 1 : currentCorner + 1;
            const nextCornerData = cornerPredictions[`corner_${nextCornerNum}`];

            if (nextCornerData) {
                const nextCornerPhases = Object.entries(nextCornerData.phases)
                    .filter(([_, phaseData]: [string, any]) => phaseData.phase_position !== undefined);

                if (nextCornerPhases.length > 0) {
                    const [nextPhaseName, nextPhaseData] = nextCornerPhases[0];
                    nextPhase = `Corner ${nextCornerNum} - ${nextPhaseName}`;
                    nextPhasePosition = (nextPhaseData as any).phase_position;
                }
            }
        }

        if (!nextPhase) return null;

        // Calculate distance to next phase
        let distance = nextPhasePosition - currentPosition;
        if (distance < 0) {
            distance += 1.0; // Handle wrap-around to next lap
        }

        // Calculate time to reach based on current approach speed
        const timeToReach = approachSpeed > 0 ? distance / approachSpeed : Infinity;

        // Generate predictive guidance based on next phase - only show for entry phases
        const generatePredictiveGuidance = (phaseName: string, timeToReach: number): string => {
            // Only show predictive guidance for entry/brake phases
            if (phaseName.toLowerCase().includes('entry') || phaseName.toLowerCase().includes('brake')) {
                const timeStr = timeToReach < 5 ? `${timeToReach.toFixed(1)}s` : 'soon';

                //time to reach is in seconds (accounting for human reaction time ~0.3s + preparation)
                if (timeToReach < 1.5) {
                    return `🚨 BRAKE NOW! ${timeStr} to braking point - execute your braking plan!`;
                } else if (timeToReach < 3) {
                    return `⚠️ Brake zone in ${timeStr}! Prepare to brake - find your markers and get ready!`;
                } else if (timeToReach < 6) {
                    return `📍 Corner entry approaching in ${timeStr}. Start looking for braking markers and prepare for turn-in.`;
                } else {
                    return `🏁 Next: ${phaseName} in ${timeStr}. Maintain current pace and prepare for corner approach.`;
                }
            }

            // Return empty string for non-entry phases (no predictive guidance)
            return '';
        };

        const predictiveGuidance = generatePredictiveGuidance(nextPhase, timeToReach);

        // Always return next phase info (system needs to know what's coming)
        return {
            name: nextPhase,
            distance: distance,
            timeToReach: timeToReach,
            approachSpeed: approachSpeed,
            predictiveGuidance: predictiveGuidance // Entry phases get guidance text, others get empty string
        };
    };

    // Function to find the current corner and phase based on position with optimized corner tracking
    const findCurrentCornerAndPhase = (position: number, approachSpeed: number = 0) => {
        if (!trackGuidanceData?.trackData?._prediction_result?.corner_predictions || cornerBoundaries.length === 0) {
            console.log('No corner predictions or boundaries available');
            return null;
        }

        const cornerPredictions = trackGuidanceData.trackData._prediction_result.corner_predictions;

        // Helper function to determine which phase the position is in - simplified to find closest phase
        const findPhaseInCorner = (cornerData: any, position: number) => {
            // Get all phases in the corner (already in correct order)
            const phases = Object.entries(cornerData.phases);
            const validPhases = phases.filter(([_, phaseData]: [string, any]) => phaseData.phase_position !== undefined);

            if (validPhases.length === 0) return null;

            // Simply find the closest phase by distance
            let closestPhase = null;
            let minDistance = Infinity;

            for (const [phaseName, phaseData] of validPhases) {

                // Calculate distance considering wrap-around
                const distance = Math.abs(position - (phaseData as any).phase_position);

                // Find the phase with the minimum distance
                if (distance < minDistance) {
                    minDistance = distance;
                    closestPhase = { phaseName, phaseData };
                }
            }

            return closestPhase;
        };

        // Get current corner boundary
        const currentCornerBoundary = cornerBoundaries.find(b => b.cornerNum === currentCornerIndex);
        if (!currentCornerBoundary) {
            console.log('Current corner boundary not found');
            return null;
        }

        // Check if position is still within current corner
        const isInCurrentCorner = position >= currentCornerBoundary.start && position <= currentCornerBoundary.end;

        if (isInCurrentCorner) {
            // Position is in current corner, find the phase
            const cornerData = cornerPredictions[`corner_${currentCornerIndex}`];
            if (cornerData) {
                const phaseResult = findPhaseInCorner(cornerData, position);
                if (phaseResult) {
                    const nextPhaseInfo = findNextPhase(currentCornerIndex, phaseResult.phaseName, position, approachSpeed);
                    return {
                        corner: currentCornerIndex,
                        phase: phaseResult.phaseName,
                        phaseData: phaseResult.phaseData,
                        nextPhase: nextPhaseInfo
                    };
                }
            }
        } else {
            // Position is outside current corner - update current corner index
            // Check next corner first (most likely scenario)
            const totalCorners = cornerBoundaries.length;
            const nextCornerIndex = (currentCornerIndex % totalCorners) + 1;
            const nextCornerBoundary = cornerBoundaries.find(b => b.cornerNum === nextCornerIndex);

            if (nextCornerBoundary) {
                const isInNextCorner = position >= nextCornerBoundary.start && position <= nextCornerBoundary.end;

                if (isInNextCorner) {
                    setCurrentCornerIndex(nextCornerIndex);
                    const cornerData = cornerPredictions[`corner_${nextCornerIndex}`];
                    if (cornerData) {
                        const phaseResult = findPhaseInCorner(cornerData, position);
                        if (phaseResult) {
                            const nextPhaseInfo = findNextPhase(nextCornerIndex, phaseResult.phaseName, position, approachSpeed);
                            return {
                                corner: nextCornerIndex,
                                phase: phaseResult.phaseName,
                                phaseData: phaseResult.phaseData,
                                nextPhase: nextPhaseInfo
                            };
                        }
                    }
                }
            }

            // If not in next corner, check previous corner (in case car went backwards)
            const prevCornerIndex = currentCornerIndex === 1 ? totalCorners : currentCornerIndex - 1;
            const prevCornerBoundary = cornerBoundaries.find(b => b.cornerNum === prevCornerIndex);

            if (prevCornerBoundary) {
                const isInPrevCorner = position >= prevCornerBoundary.start && position <= prevCornerBoundary.end;

                if (isInPrevCorner) {
                    setCurrentCornerIndex(prevCornerIndex);
                    const cornerData = cornerPredictions[`corner_${prevCornerIndex}`];
                    if (cornerData) {
                        const phaseResult = findPhaseInCorner(cornerData, position);
                        if (phaseResult) {
                            const nextPhaseInfo = findNextPhase(prevCornerIndex, phaseResult.phaseName, position, approachSpeed);
                            return {
                                corner: prevCornerIndex,
                                phase: phaseResult.phaseName,
                                phaseData: phaseResult.phaseData,
                                nextPhase: nextPhaseInfo
                            };
                        }
                    }
                }
            }

            // Position is between corners (straight section) - no guidance needed
            return null;
        }

        return null;
    };

    // Function to generate human-readable guidance text
    const generateHumanReadableText = (actions: any, phase: string): string => {
        let guidanceText = '';

        // Handle straight sections and approaching corners with direct descriptions
        if (phase.includes('Straight') || phase.includes('Approaching')) {
            if (actions.optimal_throttle) {
                guidanceText += actions.optimal_throttle.description + '. ';
            }
            if (actions.optimal_brake && actions.optimal_brake.value > 0) {
                guidanceText += actions.optimal_brake.description + '. ';
            }
            if (actions.optimal_steering) {
                guidanceText += actions.optimal_steering.description + '. ';
            }
            if (actions.optimal_speed) {
                guidanceText += actions.optimal_speed.description + '. ';
            }
            return guidanceText.trim();
        }

        // For corner phases, use preloaded sentences if available
        if (!trackGuidanceData?.preloadSentences) {
            // Fallback to action descriptions if no preloaded sentences
            if (actions.optimal_throttle) {
                guidanceText += actions.optimal_throttle.description + '. ';
            }
            if (actions.optimal_brake) {
                guidanceText += actions.optimal_brake.description + '. ';
            }
            if (actions.optimal_steering) {
                guidanceText += actions.optimal_steering.description + '. ';
            }
            if (actions.optimal_speed) {
                guidanceText += actions.optimal_speed.description + '. ';
            }
            return guidanceText.trim();
        }

        const sentences = trackGuidanceData.preloadSentences;

        // Select appropriate guidance based on current actions and phase
        if (actions.optimal_throttle && sentences.throttle_guidance) {
            const throttleIndex = Math.min(
                Math.floor(actions.optimal_throttle.value * sentences.throttle_guidance.length),
                sentences.throttle_guidance.length - 1
            );
            guidanceText += sentences.throttle_guidance[throttleIndex] + '. ';
        }

        if (actions.optimal_brake && sentences.brake_guidance) {
            const brakeIndex = Math.min(
                Math.floor(actions.optimal_brake.value * sentences.brake_guidance.length),
                sentences.brake_guidance.length - 1
            );
            guidanceText += sentences.brake_guidance[brakeIndex] + '. ';
        }

        if (actions.optimal_steering && sentences.steering_guidance) {
            const steerIndex = Math.min(
                Math.floor(Math.abs(actions.optimal_steering.value) * sentences.steering_guidance.length),
                sentences.steering_guidance.length - 1
            );
            guidanceText += sentences.steering_guidance[steerIndex] + '. ';
        }

        return guidanceText.trim();
    };

    // Main effect to monitor live telemetry data and update guidance
    useEffect(() => {
        if (!analysisContext?.liveData || !trackGuidanceData) return;

        const telemetryData = analysisContext.liveData;
        const normalizedPosition = normalizeCarPosition(telemetryData);
        const currentTime = Date.now();

        // Update position history for approach speed calculation
        setPositionHistory(prev => {
            const newHistory = [...prev, { position: normalizedPosition, timestamp: currentTime }];
            // Keep only last 5 seconds of data
            const fiveSecondsAgo = currentTime - 5000;
            return newHistory.filter(entry => entry.timestamp > fiveSecondsAgo);
        });

        // Calculate current approach speed
        const approachSpeed = calculateApproachSpeed(normalizedPosition, positionHistory);
        setCurrentSpeed(approachSpeed);

        // Detect lap change and reset corner index if needed
        const currentCompletedLaps = telemetryData.Graphics_completed_lap || 0;
        if (currentCompletedLaps > lastCompletedLap) {
            // New lap detected - reset to corner 1
            setCurrentCornerIndex(1);
            setLastCompletedLap(currentCompletedLaps);
        }

        // Find current corner and phase with approach speed
        const currentLocation = findCurrentCornerAndPhase(normalizedPosition, approachSpeed);

        if (currentLocation) {
            const actions = (currentLocation.phaseData as any).optimal_actions;
            const humanText = generateHumanReadableText(actions, currentLocation.phase);

            // Enhanced human-readable text with phase context
            let enhancedGuidanceText = humanText;
            if (!enhancedGuidanceText || enhancedGuidanceText.length < 10) {
                // Generate contextual guidance based on phase type
                if (currentLocation.phase.includes('Straight')) {
                    enhancedGuidanceText = "You're on a straight section. Focus on maximizing speed and preparing for the next corner. Keep the car stable and straight.";
                } else if (currentLocation.phase.includes('Approaching')) {
                    enhancedGuidanceText = "Corner approaching! Start preparing for entry - check your speed, find your braking point, and position the car for optimal entry.";
                } else if (currentLocation.phase.toLowerCase().includes('entry')) {
                    enhancedGuidanceText = "Corner entry phase. Focus on smooth braking, proper turn-in timing, and setting up for the apex.";
                } else if (currentLocation.phase.toLowerCase().includes('apex')) {
                    enhancedGuidanceText = "At the apex! Maintain smooth steering input and prepare to get back on throttle as you exit the corner.";
                } else if (currentLocation.phase.toLowerCase().includes('exit')) {
                    enhancedGuidanceText = "Corner exit phase. Gradually increase throttle while unwinding steering to maximize acceleration down the straight.";
                } else {
                    enhancedGuidanceText = humanText || "Follow the AI recommendations for optimal performance in this section.";
                }
            }

            const guidance: CurrentGuidance = {
                phase: currentLocation.phase,
                corner: currentLocation.corner,
                position: normalizedPosition,
                actions: {
                    throttle: actions.optimal_throttle?.description || 'N/A',
                    brake: actions.optimal_brake?.description || 'N/A',
                    steering: actions.optimal_steering?.description || 'N/A',
                    speed: actions.optimal_speed?.description || 'N/A'
                },
                confidence: (currentLocation.phaseData as any).confidence || 0,
                humanReadableText: enhancedGuidanceText,
                nextPhase: currentLocation.nextPhase || undefined
            };

            setCurrentGuidance(guidance);
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
                                {findCurrentCornerAndPhase(normalizeCarPosition(analysisContext?.liveData || {}))
                                    ? `Current Location: ${currentGuidance.corner === 0 ? currentGuidance.phase : `Corner ${currentGuidance.corner} - ${currentGuidance.phase}`}`
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

                            {/* Next Phase Section - show based on timeToReach logic */}
                            {currentGuidance.nextPhase && (
                                // Show predictive guidance for entry phases when timeToReach > 3s
                                // Show next phase info for any phase when timeToReach <= 1s
                                (currentGuidance.nextPhase.predictiveGuidance && currentGuidance.nextPhase.timeToReach > 3) ||
                                currentGuidance.nextPhase.timeToReach <= 1
                            ) && (
                                    <Box style={{
                                        padding: '12px',
                                        backgroundColor: currentGuidance.nextPhase.timeToReach < 5 ? 'var(--orange-2)' : 'var(--green-2)',
                                        borderRadius: '8px',
                                        marginTop: '12px',
                                        border: currentGuidance.nextPhase.timeToReach < 3 ? '2px solid var(--orange-7)' : '1px solid var(--green-7)'
                                    }}>
                                        <Flex justify="between" align="center" style={{ marginBottom: '8px' }}>
                                            <Text size="2" weight="medium">
                                                {currentGuidance.nextPhase.timeToReach <= 10 ? '📍 Next Phase' : '🔮 Predictive Guidance'}
                                            </Text>
                                            <Badge color={currentGuidance.nextPhase.timeToReach < 5 ? "orange" : "green"}>
                                                {currentGuidance.nextPhase.timeToReach < 60 ?
                                                    `${currentGuidance.nextPhase.timeToReach.toFixed(1)}s` :
                                                    'Soon'
                                                }
                                            </Badge>
                                        </Flex>
                                        <Text size="2" style={{ lineHeight: '1.6' }}>
                                            {currentGuidance.nextPhase.timeToReach <= 10
                                                ? `Next: ${currentGuidance.nextPhase.name}`
                                                : currentGuidance.nextPhase.predictiveGuidance
                                            }
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
