# Dynamic Visualization System

A flexible, runtime-configurable data visualization system for the ACLA frontend that allows adding, removing, and automatically arranging visualization components.

## Features

- **Dynamic Component Addition**: Add visualization components at runtime
- **Auto-Layout Correction**: Automatically arranges components for optimal layout
- **Shared Context Integration**: Uses AnalysisContext to access session and telemetry data
- **AI Integration**: Can be controlled programmatically from AI chat commands
- **Responsive Design**: Adapts layout based on number of components
- **TypeScript Support**: Fully typed for better development experience

## Architecture

### Core Components

1. **VisualizationRegistry**: Manages available visualization types
2. **DynamicVisualizationManager**: Main container component that handles layout
3. **VisualizationController**: Programmatic API for controlling visualizations
4. **Individual Chart Components**: Specific visualization implementations

### Layout System

The auto-layout system adapts based on the number of visualizations:

- **1 visualization**: Full width (100%)
- **2 visualizations**: Side by side (48% each)
- **3 visualizations**: Two on top (48% each), one below (100%)
- **4+ visualizations**: Grid layout with equal cells

## Usage

### Basic Usage in Components

```tsx
import DynamicVisualizationManager from '../../../components/visualization/DynamicVisualizationManager';

const MyComponent = () => {
    const handleLayoutChange = (instances) => {
        // Handle layout changes
        console.log('Visualizations updated:', instances);
    };

    return (
        <DynamicVisualizationManager 
            onLayoutChange={handleLayoutChange}
        />
    );
};
```

### Programmatic Control

```tsx
import { visualizationController } from '../../../components/visualization';

// Add a speed chart
visualizationController.executeCommand({
    action: 'add',
    type: 'speed-chart',
    data: speedData,
    config: { lineColor: '#ff0000' }
});

// Remove a specific visualization
visualizationController.executeCommand({
    action: 'remove',
    id: 'speed-chart_123456'
});

// Clear all visualizations
visualizationController.executeCommand({
    action: 'clear'
});
```

### Using Examples/Presets

```tsx
import { addVisualizationExamples } from '../../../components/visualization';

// Add demo visualizations (includes map)
addVisualizationExamples.addDemoSet();

// Add individual charts
addVisualizationExamples.addSpeedChart();
addVisualizationExamples.addTelemetryOverview();
addVisualizationExamples.addMapVisualization();
```

### AI Integration

The system exposes commands that can be called from AI chat:

```tsx
import { visualizationCommands } from '../../../components/visualization';

// Available commands for AI
const commands = {
    'add_speed_chart': visualizationCommands.add_speed_chart,
    'add_telemetry_overview': visualizationCommands.add_telemetry_overview,
    'add_map_visualization': visualizationCommands.add_map_visualization,
    'clear_visualizations': visualizationCommands.clear_visualizations,
    // ... more commands
};
```

## Available Visualizations

### Speed Chart
- **Type**: `speed-chart`
- **Description**: Line chart showing speed over time
- **Data Source**: `analysisContext.recordedSessionData` (filtered for speed data)
- **Config**: `lineColor`, `showGrid`

### Telemetry Overview
- **Type**: `telemetry-overview`
- **Description**: Statistical overview of telemetry data
- **Data Source**: `analysisContext.recordedSessionData`
- **Displays**: Average speed, max speed, lap count, data points

### Lap Time Chart
- **Type**: `lap-time-chart`
- **Description**: Bar chart showing lap times
- **Data Source**: `analysisContext.recordedSessionData` (filtered for lap times)
- **Config**: `barColor`, `showAverage`

### Track Map
- **Type**: `map-visualization`
- **Description**: Interactive track map with session data overlay
- **Data Source**: Uses the same map component as the main map section
- **Config**: None (inherits from SessionAnalysisMap)
- **Special**: Automatically gets more space in layouts due to square aspect ratio needs

## Creating Custom Visualizations

### 1. Create the Component

```tsx
import React, { useContext } from 'react';
import { AnalysisContext } from '../../../views/lap-analysis/session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';

const MyCustomChart: React.FC<VisualizationProps> = ({ id, data, config, width, height }) => {
    const analysisContext = useContext(AnalysisContext);
    
    return (
        <div style={{ width, height }}>
            {/* Your visualization implementation */}
        </div>
    );
};

export default MyCustomChart;
```

### 2. Register the Component

```tsx
import { visualizationRegistry } from '../VisualizationRegistry';
import MyCustomChart from './MyCustomChart';

visualizationRegistry.register('my-custom-chart', {
    component: MyCustomChart,
    name: 'My Custom Chart',
    description: 'A custom visualization',
    defaultConfig: {
        // Default configuration
    },
    minWidth: 300,
    minHeight: 200,
    preferredAspectRatio: 16/9
});
```

## Context Integration

The system is integrated with `AnalysisContext` to share data across components:

```tsx
interface AnalysisContextType {
    // ... existing properties
    activeVisualizations: VisualizationInstance[];
    setActiveVisualizations: Dispatch<SetStateAction<VisualizationInstance[]>>;
}
```

All visualization components have access to:
- `recordedSessionData`: Session telemetry data
- `liveData`: Real-time telemetry
- `sessionSelected`: Current session info
- Other context properties

## Styling

The system includes responsive CSS that:
- Adapts to container size
- Provides hover effects
- Handles drag indicators
- Supports mobile layouts

## Best Practices

1. **Data Access**: Use `AnalysisContext` for accessing shared session data
2. **Performance**: Memoize expensive calculations in visualization components
3. **Responsive**: Design visualizations to work at different sizes
4. **Error Handling**: Handle missing or invalid data gracefully
5. **Config Management**: Use default configurations and allow overrides

## Examples

The system comes with working examples in `examples.ts` that demonstrate:
- Adding different chart types
- Clearing visualizations
- Demo datasets for testing
- Integration with AI commands

This system provides a flexible foundation for adding data visualizations that can be controlled both through the UI and programmatically, making it ideal for AI-driven analytics applications.
