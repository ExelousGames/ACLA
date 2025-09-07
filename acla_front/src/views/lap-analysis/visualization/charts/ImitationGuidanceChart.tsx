import React, { useContext, useState, useEffect } from 'react';
import { Card, Text, Box, Grid, Badge, Separator, Progress, Flex } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';
import apiService from 'services/api.service';
import styles from './ImitationGuidanceChart.module.css';

interface ImitationGuidanceData {
    message: string;
    guidance_result: {
        success: boolean;
        timestamp: string;
        behavior_guidance: {
            predicted_driving_style: string;
            confidence: number;
            style_recommendations: string[];
            alternative_styles: {
                [key: string]: number;
            };
        };
        action_guidance: {
            optimal_actions: {
                optimal_speed: number;
                optimal_steering: number;
                optimal_throttle: number;
                optimal_brake: number;
                optimal_gear: number;
                optimal_player_pos_x: number;
                optimal_player_pos_y: number;
                optimal_player_pos_z: number;
                optimal_track_position: number;
            };
            action_recommendations: {
                point_similarity: number;
                speed: {
                    user_value: number;
                    expert_value: number;
                    difference: number;
                    percentage_diff: number;
                };
                throttle: {
                    user_value: number;
                    expert_value: number;
                    difference: number;
                    percentage_diff: number;
                };
                brake: {
                    user_value: number;
                    expert_value: number;
                    difference: number;
                    percentage_diff: number;
                };
                steering: {
                    user_value: number;
                    expert_value: number;
                    difference: number;
                    percentage_diff: number;
                };
                gear: {
                    user_value: number;
                    expert_value: number;
                    difference: number;
                    gear_optimal: boolean;
                };
                position: {
                    user_position: {
                        x: number;
                        y: number;
                        z: number;
                    };
                    expert_position: {
                        x: number;
                        y: number;
                        z: number;
                    };
                    difference: {
                        x: number;
                        y: number;
                        z: number;
                    };
                    lateral_distance: number;
                    vertical_difference: number;
                };
            };
            performance_insights: {
                speed_efficiency: number;
                throttle_efficiency: number;
                brake_efficiency: number;
                overall_efficiency: number;
                performance_level: string;
                improvement_potential: string;
            };
        };
    };
    timestamp: string;
    success: boolean;
}

const ImitationGuidanceChart: React.FC<VisualizationProps> = ({
    id,
    data,
    config,
    width = '100%',
    height = 400
}) => {
    const analysisContext = useContext(AnalysisContext);
    const [guidanceData, setGuidanceData] = useState<ImitationGuidanceData | null>(null);
    const [isInitialLoading, setIsInitialLoading] = useState(true);
    const [isUpdating, setIsUpdating] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Fetch imitation learning guidance
    useEffect(() => {
        // Don't start interval if no live data
        if (!analysisContext?.liveData) {
            setIsInitialLoading(false);
            return;
        }

        const fetchGuidance = async () => {
            // Only show loading spinner on initial load
            if (!guidanceData) {
                setIsInitialLoading(true);
            } else {
                setIsUpdating(true);
            }
            setError(null);

            try {
                const response = await apiService.post('/racing-session/imitation-learning-guidance', {
                    current_telemetry: analysisContext.liveData,
                    track_name: analysisContext.recordedSessioStaticsData?.track || "unknown",
                    car_name: analysisContext.recordedSessioStaticsData?.car_model || "unknown",
                    guidance_type: "both", // "actions", "behavior", or "both"
                });

                const responseData = response.data as ImitationGuidanceData;
                if (responseData.success) {
                    setGuidanceData(responseData);
                } else {
                    setError('Failed to get guidance data');
                }
            } catch (error) {
                console.error('Error fetching imitation learning guidance:', error);
                setError('Failed to fetch guidance data');
            } finally {
                setIsInitialLoading(false);
                setIsUpdating(false);
            }
        };

        // Fetch guidance every 5 seconds when live data changes (less aggressive than 2 seconds)
        const interval = setInterval(fetchGuidance, 5000);

        // Initial fetch
        fetchGuidance();

        return () => clearInterval(interval);
    }, [analysisContext?.liveData, analysisContext?.recordedSessioStaticsData]);

    const getDrivingStyleColor = (style: string) => {
        switch (style.toLowerCase()) {
            case 'aggressive': return 'red';
            case 'conservative': return 'blue';
            case 'defensive': return 'yellow';
            case 'optimal': return 'green';
            case 'smooth': return 'purple';
            default: return 'gray';
        }
    };

    const getPerformanceLevelColor = (level: string) => {
        switch (level.toLowerCase()) {
            case 'expert': return 'green';
            case 'advanced': return 'blue';
            case 'intermediate': return 'yellow';
            case 'beginner': return 'red';
            default: return 'gray';
        }
    };

    const renderBehaviorGuidance = (behavior: ImitationGuidanceData['guidance_result']['behavior_guidance']) => (
        <Box className={styles.behaviorSection}>
            <Text size="3" weight="bold" className={styles.sectionTitle}>Driving Style Analysis</Text>

            <Box className={styles.confidenceContainer}>
                <Flex align="center" gap="3" className={styles.confidenceBadges}>
                    <Badge color={getDrivingStyleColor(behavior.predicted_driving_style)} size="2">
                        {behavior.predicted_driving_style.charAt(0).toUpperCase() + behavior.predicted_driving_style.slice(1)}
                    </Badge>
                    <Text size="2" color="gray">Confidence: {(behavior.confidence * 100).toFixed(1)}%</Text>
                </Flex>
                <Progress value={behavior.confidence * 100} className={styles.confidenceProgress} />
            </Box>

            <Box className={styles.styleDistribution}>
                <Text size="2" weight="medium" className={styles.distributionTitle}>Style Distribution:</Text>
                <Box className={styles.distributionList}>
                    {Object.entries(behavior.alternative_styles).map(([style, confidence]) => (
                        <Box key={style} className={styles.distributionItem}>
                            <Flex align="center" justify="between" className={styles.distributionItemHeader}>
                                <Text size="1" className={styles.distributionStyle}>
                                    {style.charAt(0).toUpperCase() + style.slice(1)}
                                </Text>
                                <Text size="1" color="gray" className={styles.distributionConfidence}>
                                    {(confidence * 100).toFixed(0)}%
                                </Text>
                            </Flex>
                            <Progress
                                value={confidence * 100}
                                className={styles.distributionProgress}
                                color={getDrivingStyleColor(style)}
                            />
                        </Box>
                    ))}
                </Box>
            </Box>

            <Box>
                <Text size="2" weight="medium" className={styles.recommendationsTitle}>Recommendations:</Text>
                <Box className={styles.recommendationsList}>
                    {behavior.style_recommendations.map((rec, index) => (
                        <Box key={index} className={styles.recommendationItem}>
                            <Text size="1" color="gray" className={styles.recommendationText}>
                                • {rec}
                            </Text>
                        </Box>
                    ))}
                </Box>
            </Box>
        </Box>
    );

    const renderActionGuidance = (actions: ImitationGuidanceData['guidance_result']['action_guidance']) => (
        <Box>
            <Text size="3" weight="bold" className={styles.sectionTitle}>Performance Analysis</Text>

            <Box className={styles.actionSection}>
                <Grid columns="3" gap="3" className={styles.performanceGrid}>
                    <Box className={styles.performanceItem}>
                        <Text size="1" color="gray" className={styles.performanceLabel}>Performance Level</Text>
                        <Badge color={getPerformanceLevelColor(actions.performance_insights.performance_level)} size="2">
                            {actions.performance_insights.performance_level}
                        </Badge>
                    </Box>
                    <Box className={styles.performanceItem}>
                        <Text size="1" color="gray" className={styles.performanceLabel}>Overall Efficiency</Text>
                        <Text size="2" weight="bold">{actions.performance_insights.overall_efficiency.toFixed(1)}%</Text>
                    </Box>
                    <Box className={styles.performanceItem}>
                        <Text size="1" color="gray" className={styles.performanceLabel}>Point Similarity</Text>
                        <Text size="2" weight="bold">{actions.action_recommendations.point_similarity.toFixed(1)}%</Text>
                    </Box>
                </Grid>
            </Box>

            <Separator className={styles.separator} />

            <Box className={styles.actionComparison}>
                <Text size="2" weight="medium" className={styles.comparisonTitle}>Action Comparison:</Text>
                <Box className={styles.comparisonContainer}>
                    {/* Speed and Throttle Row */}
                    <Box className={styles.comparisonRow}>
                        <Box className={styles.comparisonBox}>
                            <Text size="1" weight="medium" className={styles.comparisonBoxTitle}>Speed</Text>
                            <Box className={styles.comparisonValues}>
                                <Text size="1" color="blue" className={styles.comparisonValue}>User: {actions.action_recommendations.speed.user_value.toFixed(1)}</Text>
                                <Text size="1" color="green" className={styles.comparisonValue}>Expert: {actions.action_recommendations.speed.expert_value.toFixed(1)}</Text>
                                <Text size="1" color="gray" className={styles.comparisonDiff}>
                                    Diff: {actions.action_recommendations.speed.percentage_diff.toFixed(1)}%
                                </Text>
                            </Box>
                        </Box>
                        <Box className={styles.comparisonBox}>
                            <Text size="1" weight="medium" className={styles.comparisonBoxTitle}>Throttle</Text>
                            <Box className={styles.comparisonValues}>
                                <Text size="1" color="blue" className={styles.comparisonValue}>User: {actions.action_recommendations.throttle.user_value.toFixed(3)}</Text>
                                <Text size="1" color="green" className={styles.comparisonValue}>Expert: {actions.action_recommendations.throttle.expert_value.toFixed(3)}</Text>
                                <Text size="1" color="gray" className={styles.comparisonDiff}>
                                    Diff: {actions.action_recommendations.throttle.percentage_diff.toFixed(1)}%
                                </Text>
                            </Box>
                        </Box>
                    </Box>
                    {/* Brake and Gear Row */}
                    <Box className={styles.comparisonRow}>
                        <Box className={styles.comparisonBox}>
                            <Text size="1" weight="medium" className={styles.comparisonBoxTitle}>Brake</Text>
                            <Box className={styles.comparisonValues}>
                                <Text size="1" color="blue" className={styles.comparisonValue}>User: {actions.action_recommendations.brake.user_value.toFixed(3)}</Text>
                                <Text size="1" color="green" className={styles.comparisonValue}>Expert: {actions.action_recommendations.brake.expert_value.toFixed(3)}</Text>
                                <Text size="1" color="gray" className={styles.comparisonDiff}>
                                    Diff: {actions.action_recommendations.brake.percentage_diff.toFixed(1)}%
                                </Text>
                            </Box>
                        </Box>
                        <Box className={styles.comparisonBox}>
                            <Text size="1" weight="medium" className={styles.comparisonBoxTitle}>Gear</Text>
                            <Box className={styles.comparisonValues}>
                                <Text size="1" color="blue" className={styles.comparisonValue}>User: {actions.action_recommendations.gear.user_value}</Text>
                                <Flex align="center" gap="2" className={styles.gearExpert}>
                                    <Text size="1" color="green" className={styles.comparisonValue}>Expert: {actions.action_recommendations.gear.expert_value}</Text>
                                    {actions.action_recommendations.gear.gear_optimal && (
                                        <Badge color="green" size="1">✓</Badge>
                                    )}
                                </Flex>
                                <Text size="1" color="gray" className={styles.comparisonDiff}>
                                    {actions.action_recommendations.gear.gear_optimal ? 'Optimal' : 'Sub-optimal'}
                                </Text>
                            </Box>
                        </Box>
                    </Box>
                    {/* Position Row */}
                    <Box className={styles.comparisonRow}>
                        <Box className={styles.comparisonBox}>
                            <Text size="1" weight="medium" className={styles.comparisonBoxTitle}>Lateral Distance</Text>
                            <Box className={styles.comparisonValues}>
                                <Text size="1" color="gray" className={styles.comparisonValue}>Distance: {actions.action_recommendations.position.lateral_distance.toFixed(2)}m</Text>
                                <Text size="1" color="gray" className={styles.comparisonValue}>Vertical: {Math.abs(actions.action_recommendations.position.vertical_difference).toFixed(2)}m</Text>
                            </Box>
                        </Box>
                        <Box className={styles.comparisonBox}>
                            <Text size="1" weight="medium" className={styles.comparisonBoxTitle}>Track Position</Text>
                            <Box className={styles.comparisonValues}>
                                <Text size="1" color="green" className={styles.comparisonValue}>Optimal: {actions.optimal_actions.optimal_track_position.toFixed(3)}</Text>
                            </Box>
                        </Box>
                    </Box>
                </Box>
            </Box>

            <Separator className={styles.separator} />

            <Box>
                <Text size="2" weight="medium" className={styles.efficiencyTitle}>Efficiency Breakdown:</Text>
                <Grid columns="3" gap="2">
                    <Box className={styles.efficiencyItem}>
                        <Text size="1" className={styles.efficiencyLabel}>Speed</Text>
                        <Progress
                            value={actions.performance_insights.speed_efficiency}
                            color="blue"
                            className={styles.efficiencyProgress}
                        />
                        <Text size="1" color="gray">{actions.performance_insights.speed_efficiency.toFixed(1)}%</Text>
                    </Box>
                    <Box className={styles.efficiencyItem}>
                        <Text size="1" className={styles.efficiencyLabel}>Throttle</Text>
                        <Progress
                            value={actions.performance_insights.throttle_efficiency}
                            color="green"
                            className={styles.efficiencyProgress}
                        />
                        <Text size="1" color="gray">{actions.performance_insights.throttle_efficiency.toFixed(1)}%</Text>
                    </Box>
                    <Box className={styles.efficiencyItem}>
                        <Text size="1" className={styles.efficiencyLabel}>Brake</Text>
                        <Progress
                            value={actions.performance_insights.brake_efficiency}
                            color="red"
                            className={styles.efficiencyProgress}
                        />
                        <Text size="1" color="gray">{actions.performance_insights.brake_efficiency.toFixed(1)}%</Text>
                    </Box>
                </Grid>
            </Box>
        </Box>
    );

    return (
        <Card style={{ width, height }} className={styles.chartCard}>
            <Flex justify="between" align="center" className={styles.chartHeader}>
                <Text size="3" weight="bold">AI Driving Guidance</Text>
                {isUpdating && (
                    <Badge color="blue" size="1">Updating...</Badge>
                )}
            </Flex>

            {isInitialLoading && (
                <Box className={styles.loadingContainer}>
                    <Text color="gray">Loading guidance...</Text>
                </Box>
            )}

            {error && (
                <Box className={styles.loadingContainer}>
                    <Text color="red">{error}</Text>
                </Box>
            )}

            {!guidanceData && !isInitialLoading && !error && (
                <Box className={styles.loadingContainer}>
                    <Text color="gray">No guidance data available</Text>
                </Box>
            )}

            {guidanceData && guidanceData.success && guidanceData.guidance_result.success && !isInitialLoading && (
                <Box className={styles.contentContainer}>
                    {renderBehaviorGuidance(guidanceData.guidance_result.behavior_guidance)}
                    <Separator className={styles.mainSeparator} />
                    {renderActionGuidance(guidanceData.guidance_result.action_guidance)}

                    <Box className={styles.timestamp}>
                        Last updated: {new Date(guidanceData.timestamp).toLocaleTimeString()}
                    </Box>
                </Box>
            )}
        </Card>
    );
};

export default ImitationGuidanceChart;
