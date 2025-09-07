// Example of how to use the visualization system programmatically

import { visualizationController } from './VisualizationController';

// Example functions that can be called from anywhere in the app

export const addVisualizationExamples = {
    // Add a speed chart with sample data
    addSpeedChart: () => {
        const sampleSpeedData = [
            { time: 0, speed: 0 },
            { time: 10, speed: 50 },
            { time: 20, speed: 120 },
            { time: 30, speed: 180 },
            { time: 40, speed: 200 },
            { time: 50, speed: 160 },
            { time: 60, speed: 80 },
            { time: 70, speed: 0 }
        ];

        return visualizationController.executeCommand({
            action: 'add',
            type: 'speed-chart',
            data: sampleSpeedData,
            config: { lineColor: '#3b82f6', showGrid: true }
        });
    },

    // Add telemetry overview
    addTelemetryOverview: () => {
        return visualizationController.executeCommand({
            action: 'add',
            type: 'telemetry-overview'
        });
    },

    // Add lap time chart
    addLapTimeChart: () => {
        const sampleLapData = [
            { lapNumber: 1, lapTime: 95.432 },
            { lapNumber: 2, lapTime: 92.156 },
            { lapNumber: 3, lapTime: 91.789 },
            { lapNumber: 4, lapTime: 93.221 },
            { lapNumber: 5, lapTime: 90.987 }
        ];

        return visualizationController.executeCommand({
            action: 'add',
            type: 'lap-time-chart',
            data: sampleLapData
        });
    },

    // Add map visualization
    addMapVisualization: () => {
        return visualizationController.executeCommand({
            action: 'add',
            type: 'map-visualization'
        });
    },

    // Clear all visualizations
    clearAll: () => {
        return visualizationController.executeCommand({
            action: 'clear'
        });
    },

    // Add multiple visualizations for demo
    addDemoSet: () => {
        addVisualizationExamples.clearAll();
        addVisualizationExamples.addMapVisualization();
        addVisualizationExamples.addTelemetryOverview();
        addVisualizationExamples.addSpeedChart();
        addVisualizationExamples.addLapTimeChart();
    }
};

// These functions can be exposed to the AI chat system
export const visualizationCommands = {
    add_speed_chart: addVisualizationExamples.addSpeedChart,
    add_telemetry_overview: addVisualizationExamples.addTelemetryOverview,
    add_lap_time_chart: addVisualizationExamples.addLapTimeChart,
    add_map_visualization: addVisualizationExamples.addMapVisualization,
    clear_visualizations: addVisualizationExamples.clearAll,
    demo_visualizations: addVisualizationExamples.addDemoSet,

    // Get available visualization types
    get_available_types: () => {
        return visualizationController.getAvailableTypes();
    },

    // Get current visualizations
    get_current_visualizations: () => {
        return visualizationController.getCurrentInstances();
    }
};
