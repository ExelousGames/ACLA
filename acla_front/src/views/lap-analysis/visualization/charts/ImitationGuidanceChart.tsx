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

    // Extract guidance data from the visualization data
    useEffect(() => {
        if (data?.trackData && data?.preloadSentences) {
            setTrackGuidanceData({
                trackData: data.trackData,
                preloadSentences: data.preloadSentences
            });
        }
    }, [data]);

    // Function to normalize car position (0-1 range)
    const normalizeCarPosition = (telemetryData: any): number => {
        if (!telemetryData) {
            console.log('No telemetry data provided');
            return 0;
        }

        // Check for Assetto Corsa Competizione normalized position (primary)
        if (telemetryData.Graphics_normalized_car_position !== undefined) {
            const position = telemetryData.Graphics_normalized_car_position;
            console.log('Using Graphics_normalized_car_position:', position);
            return Math.max(0, Math.min(1, position));
        }

        // Check for generic normalized position
        if (telemetryData.normalizedPosition !== undefined) {
            console.log('Using normalizedPosition:', telemetryData.normalizedPosition);
            return Math.max(0, Math.min(1, telemetryData.normalizedPosition));
        }

        // Calculate from lap progress if available
        if (telemetryData.lapProgress !== undefined) {
            console.log('Using lapProgress:', telemetryData.lapProgress);
            return Math.max(0, Math.min(1, telemetryData.lapProgress));
        }

        // Calculate from distance traveled and track length
        if (telemetryData.Graphics_distance_traveled !== undefined) {
            // Estimate track length from distance and position if we have both
            if (telemetryData.Graphics_normalized_car_position !== undefined && telemetryData.Graphics_normalized_car_position > 0) {
                const estimatedTrackLength = telemetryData.Graphics_distance_traveled / telemetryData.Graphics_normalized_car_position;
                const position = (telemetryData.Graphics_distance_traveled % estimatedTrackLength) / estimatedTrackLength;
                console.log('Using distance_traveled calculation:', telemetryData.Graphics_distance_traveled, 'estimated track length:', estimatedTrackLength, '=', position);
                return Math.max(0, Math.min(1, position));
            }
        }

        // Calculate from lap distance and track length
        if (telemetryData.lapDistance !== undefined && telemetryData.trackLength) {
            const position = telemetryData.lapDistance / telemetryData.trackLength;
            console.log('Using lapDistance/trackLength:', telemetryData.lapDistance, '/', telemetryData.trackLength, '=', position);
            return Math.max(0, Math.min(1, position));
        }

        // Check for spline position (common in racing games)
        if (telemetryData.splinePosition !== undefined) {
            console.log('Using splinePosition:', telemetryData.splinePosition);
            return Math.max(0, Math.min(1, telemetryData.splinePosition));
        }

        // Fallback: use time-based calculation if lap time is available
        if (telemetryData.Graphics_current_time !== undefined && telemetryData.Graphics_best_time && telemetryData.Graphics_best_time > 0) {
            const position = Math.min(telemetryData.Graphics_current_time / telemetryData.Graphics_best_time, 1.0);
            console.log('Using ACC time-based calculation:', telemetryData.Graphics_current_time, '/', telemetryData.Graphics_best_time, '=', position);
            return Math.max(0, position);
        }

        // Generic time-based fallback
        if (telemetryData.currentLapTime !== undefined && telemetryData.bestLapTime && telemetryData.bestLapTime > 0) {
            const position = Math.min(telemetryData.currentLapTime / telemetryData.bestLapTime, 1.0);
            console.log('Using generic time-based calculation:', telemetryData.currentLapTime, '/', telemetryData.bestLapTime, '=', position);
            return Math.max(0, position);
        }

        console.log('No suitable position data found in telemetry');
        console.log('Available fields:', Object.keys(telemetryData).filter(k => k.includes('position') || k.includes('distance') || k.includes('progress')));
        return 0;
    };

    // Function to find the current corner and phase based on position
    const findCurrentCornerAndPhase = (position: number) => {
        if (!trackGuidanceData?.trackData?._prediction_result?.corner_predictions) {
            console.log('No corner predictions available');
            return null;
        }

        const cornerPredictions = trackGuidanceData.trackData._prediction_result.corner_predictions;

        // Helper function to determine which phase the position is in
        const findPhaseInCorner = (cornerData: any, position: number) => {
            const phases = Object.entries(cornerData.phases);

            // Sort phases by their position to create proper ranges
            const sortedPhases = phases
                .filter(([_, phaseData]: [string, any]) => phaseData.phase_position !== undefined)
                .sort(([_, a]: [string, any], [__, b]: [string, any]) => a.phase_position - b.phase_position);

            if (sortedPhases.length === 0) return null;

            // Check each phase range
            for (let i = 0; i < sortedPhases.length; i++) {
                const [currentPhaseName, currentPhaseData] = sortedPhases[i];
                const currentPhasePos = (currentPhaseData as any).phase_position;

                let phaseStart, phaseEnd;

                if (i === 0) {
                    // First phase: from corner start to halfway to next phase
                    phaseStart = cornerData.corner_summary.corner_start_position;
                    if (i + 1 < sortedPhases.length) {
                        const nextPhasePos = (sortedPhases[i + 1][1] as any).phase_position;
                        phaseEnd = (currentPhasePos + nextPhasePos) / 2;
                    } else {
                        phaseEnd = cornerData.corner_summary.corner_end_position;
                    }
                } else if (i === sortedPhases.length - 1) {
                    // Last phase: from halfway from previous phase to corner end
                    const prevPhasePos = (sortedPhases[i - 1][1] as any).phase_position;
                    phaseStart = (prevPhasePos + currentPhasePos) / 2;
                    phaseEnd = cornerData.corner_summary.corner_end_position;
                } else {
                    // Middle phases: from halfway to previous to halfway to next
                    const prevPhasePos = (sortedPhases[i - 1][1] as any).phase_position;
                    const nextPhasePos = (sortedPhases[i + 1][1] as any).phase_position;
                    phaseStart = (prevPhasePos + currentPhasePos) / 2;
                    phaseEnd = (currentPhasePos + nextPhasePos) / 2;
                }

                // Check if position is within this phase range
                if (position >= phaseStart && position <= phaseEnd) {
                    return {
                        phaseName: currentPhaseName,
                        phaseData: currentPhaseData,
                        phaseStart,
                        phaseEnd
                    };
                }
            }

            // If no exact match, return the closest phase
            let closestPhase = null;
            let minDistance = Infinity;

            for (const [phaseName, phaseData] of sortedPhases) {
                const distance = Math.abs(position - (phaseData as any).phase_position);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestPhase = { phaseName, phaseData };
                }
            }

            return closestPhase;
        };

        // Check each corner to see if position is within its boundaries
        for (const [cornerKey, cornerData] of Object.entries(cornerPredictions)) {
            const cornerNum = parseInt(cornerKey.replace('corner_', ''));
            const cornerStart = cornerData.corner_summary.corner_start_position;
            const cornerEnd = cornerData.corner_summary.corner_end_position;

            // Handle wrap-around case (corner that crosses lap boundary)
            let isInCorner = false;
            if (cornerEnd < cornerStart) {
                // Corner wraps around (e.g., start=0.9, end=0.1)
                isInCorner = position >= cornerStart || position <= cornerEnd;
            } else {
                // Normal corner
                isInCorner = position >= cornerStart && position <= cornerEnd;
            }

            if (isInCorner) {
                const phaseResult = findPhaseInCorner(cornerData, position);
                if (phaseResult) {
                    return {
                        corner: cornerNum,
                        phase: phaseResult.phaseName,
                        phaseData: phaseResult.phaseData
                    };
                }
            }
        }

        // If not in any corner, check between corners (straights)
        // Find the closest upcoming and previous corners
        let closestUpcomingCorner = null;
        let closestPreviousCorner = null;
        let upcomingDistance = Infinity;
        let previousDistance = Infinity;

        for (const [cornerKey, cornerData] of Object.entries(cornerPredictions)) {
            const cornerNum = parseInt(cornerKey.replace('corner_', ''));
            const cornerStart = cornerData.corner_summary.corner_start_position;
            const cornerEnd = cornerData.corner_summary.corner_end_position;

            // Calculate distance to upcoming corner
            let distanceToStart;
            if (cornerStart >= position) {
                // Corner is ahead on this lap
                distanceToStart = cornerStart - position;
            } else {
                // Corner is on next lap (wrap around)
                distanceToStart = (1 - position) + cornerStart;
            }

            // Calculate distance from previous corner
            let distanceFromEnd;
            if (cornerEnd <= position) {
                // Corner is behind on this lap
                distanceFromEnd = position - cornerEnd;
            } else {
                // Corner ended on previous lap (wrap around)
                distanceFromEnd = position + (1 - cornerEnd);
            }

            // Track closest upcoming corner
            if (distanceToStart < upcomingDistance && distanceToStart < 0.3) { // Within 30% of track
                upcomingDistance = distanceToStart;
                closestUpcomingCorner = { cornerNum, cornerData, distance: distanceToStart };
            }

            // Track closest previous corner
            if (distanceFromEnd < previousDistance && distanceFromEnd < 0.3) { // Within 30% of track
                previousDistance = distanceFromEnd;
                closestPreviousCorner = { cornerNum, cornerData, distance: distanceFromEnd };
            }
        }

        // Provide different guidance based on position on straight
        if (closestUpcomingCorner || closestPreviousCorner) {
            // Determine if we're approaching a corner or on a straight
            const isApproachingCorner = closestUpcomingCorner && upcomingDistance < 0.08; // Within 8%
            const isOnStraight = closestPreviousCorner && previousDistance > 0.02; // More than 2% from last corner

            if (isApproachingCorner && closestUpcomingCorner) {
                // Approaching corner - provide entry preparation guidance
                const phases = Object.entries(closestUpcomingCorner.cornerData.phases);
                const entryPhase = phases.find(([name, _]) =>
                    name.toLowerCase().includes('entry') ||
                    name.toLowerCase().includes('brake')
                );

                if (entryPhase) {
                    const [phaseName, phaseData] = entryPhase;
                    return {
                        corner: closestUpcomingCorner.cornerNum,
                        phase: `Approaching Corner ${closestUpcomingCorner.cornerNum}`,
                        phaseData: {
                            ...phaseData,
                            optimal_actions: {
                                optimal_throttle: {
                                    value: 0.8,
                                    description: "Maintain high throttle until braking point",
                                    change_rate: 0,
                                    rapidity: "steady"
                                },
                                optimal_brake: {
                                    value: 0.0,
                                    description: "No braking yet, prepare for upcoming corner",
                                    change_rate: 0,
                                    rapidity: "ready"
                                },
                                optimal_steering: {
                                    value: 0.0,
                                    description: "Keep steering straight, line up for corner entry",
                                    change_rate: 0,
                                    rapidity: "smooth"
                                },
                                optimal_speed: {
                                    value: 200,
                                    description: "Maintain high speed until braking zone",
                                    change_rate: 0,
                                    rapidity: "maintain"
                                }
                            }
                        } as any
                    };
                }
            } else if (isOnStraight) {
                // On straight section - provide straight-line guidance
                const straightGuidance = {
                    optimal_actions: {
                        optimal_throttle: {
                            value: 1.0,
                            description: "Full throttle on straight section",
                            change_rate: 0,
                            rapidity: "maximum"
                        },
                        optimal_brake: {
                            value: 0.0,
                            description: "No braking required on straight",
                            change_rate: 0,
                            rapidity: "none"
                        },
                        optimal_steering: {
                            value: 0.0,
                            description: "Keep car straight and stable",
                            change_rate: 0,
                            rapidity: "minimal"
                        },
                        optimal_speed: {
                            value: 250,
                            description: "Maximize speed on straight section",
                            change_rate: 1,
                            rapidity: "increasing"
                        }
                    },
                    confidence: 0.9,
                    phase_position: position
                };

                return {
                    corner: 0, // Special case for straights
                    phase: closestUpcomingCorner ?
                        `Straight (${upcomingDistance < 0.15 ? 'approaching' : 'toward'} Corner ${closestUpcomingCorner.cornerNum})` :
                        'Straight Section',
                    phaseData: straightGuidance
                };
            }
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

        // Find current corner and phase
        const currentLocation = findCurrentCornerAndPhase(normalizedPosition);

        // Enhanced debug logging for phase detection
        if (Math.random() < 0.2) { // Log 20% of the time for debugging
            console.log('Position:', normalizedPosition.toFixed(4));
            if (trackGuidanceData?.trackData?._prediction_result?.corner_predictions) {
                const corners = trackGuidanceData.trackData._prediction_result.corner_predictions;
                console.log('Available corners:');
                Object.entries(corners).forEach(([key, corner]: [string, any]) => {
                    console.log(`  ${key}: ${corner.corner_summary.corner_start_position.toFixed(4)} - ${corner.corner_summary.corner_end_position.toFixed(4)}`);
                });
            }
            if (currentLocation) {
                if (currentLocation.corner === 0) {
                    console.log('Found location: Straight section -', currentLocation.phase);
                } else {
                    console.log('Found location:', `Corner ${currentLocation.corner} - ${currentLocation.phase}`);
                }
            } else {
                console.log('No location match found');
            }
        }

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
                humanReadableText: enhancedGuidanceText
            };

            setCurrentGuidance(guidance);
        } else {
            // No specific guidance available for current position
            setCurrentGuidance(null);
        }
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
                                Current Location: {currentGuidance.corner === 0 ? currentGuidance.phase : `Corner ${currentGuidance.corner} - ${currentGuidance.phase}`}
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
