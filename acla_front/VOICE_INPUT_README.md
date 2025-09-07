# Voice Input Feature for ACLA AI Chat

Thi### Troubleshooting PyAudio Installation

PyAudio can be challenging to install on Windows. If you encounter issues:

1. **Install from requirements.txt**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download pre-built wheel** (if requirements.txt fails):
   - Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   - Download the appropriate wheel for your Python version and architecture
   - Install with: `pip install path_to_downloaded_file.whl`nt explains how to set up and use the voice input feature in the ACLA AI Chat component.

## Overview

The AI Chat now supports voice input in both web browser and Electron desktop environments:

- **Web Browser**: Uses the Web Speech API (requires internet connection)
- **Electron Desktop**: Uses local Python-based speech recognition (offline capable)

## 🚀 Quick Start

### Automatic Setup (Recommended)

The speech recognition dependencies are automatically installed when you start or build the Electron app:

```bash
npm run start:electron  # Automatically installs speech dependencies
npm run build:electron  # Automatically installs speech dependencies
```

### Manual Setup (if needed)

If automatic setup fails, you can install manually:

```bash
npm run setup-speech    # Run just the speech recognition setup
```

## Setup for Electron Desktop App

### Prerequisites

1. **Python Installation**: Ensure Python 3.7+ is installed and available in your system PATH
2. **Microphone Access**: The application will request microphone permissions

### What Gets Installed Automatically

The setup process will install these Python packages from `requirements.txt`:
- `pyaccsharedmemory` - ACC game data access
- `SpeechRecognition==3.10.0` - Core speech recognition library
- `pocketsphinx==0.1.15` - Offline speech recognition engine
- `pyaudio==0.2.11` - Audio capture (may require manual installation on some systems)

### Troubleshooting PyAudio Installation

PyAudio can be challenging to install on Windows. If you encounter issues:

1. **Download pre-built wheel**:
   - Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   - Download the appropriate wheel for your Python version and architecture
   - Install with: `pip install path_to_wheel_file.whl`

2. **Alternative for Windows**:
   ```bash
   # Install Microsoft Visual C++ Build Tools first, then:
   pip install --upgrade setuptools wheel
   pip install pyaudio
   ```

3. **Without PyAudio**: The application will still work but will fall back to online recognition only

## Features

### Voice Input Button

- **Location**: Next to the text input field in the AI chat
- **Appearance**: Microphone icon that changes color based on status:
  - Gray: Ready to record
  - Red (pulsing): Currently recording
  - Orange: Error state

### Recording Modes

1. **Local Mode (Electron)**:
   - Uses offline speech recognition when possible
   - Falls back to online recognition if offline fails
   - Works without internet connection (with proper setup)

2. **Web Mode (Browser)**:
   - Uses Google's Web Speech API
   - Requires internet connection
   - High accuracy but needs network access

### Usage

1. **Start Recording**: Click the microphone button
2. **Speak Clearly**: The system will listen for up to 10 seconds
3. **Auto-Send**: Recognized text is automatically sent to the AI
4. **Manual Stop**: Click the red microphone button to stop early

### Visual Indicators

- **Desktop Mode Badge**: Shows when running in Electron
- **Voice Error Badge**: Appears in header when there are issues
- **Microphone Icon**: Appears next to voice messages in chat
- **Status Messages**: Error messages appear in chat with 🎤 emoji

## Error Handling

Common errors and solutions:

### "Network error"
- **Web Mode**: Check internet connection
- **Electron Mode**: Install speech recognition dependencies

### "Microphone access denied"
- Grant microphone permissions in browser/system settings

### "Speech recognition not available"
- **Web Mode**: Use a supported browser (Chrome, Edge recommended)
- **Electron Mode**: Run the setup script to install Python dependencies

### "No speech detected"
- Speak more clearly or closer to the microphone
- Check microphone is working in system settings

## Technical Details

### Dependencies

- **Python packages**: `SpeechRecognition`, `pyaudio`, `pocketsphinx`
- **Web APIs**: Web Speech API (browser only)
- **Electron APIs**: Custom IPC handlers for speech recognition

### Supported Languages

- Primary: English (US)
- Can be configured for other languages by modifying the recognition settings

### Privacy

- **Electron Mode**: All processing happens locally when using offline recognition
- **Web Mode**: Audio is sent to Google's servers for processing
- **Data**: No audio data is stored permanently by the application

## Development

### Adding New Languages

Edit the language setting in `ai-chat.tsx`:

```typescript
recognition.lang = 'en-US'; // Change to desired language code
```

### Extending Functionality

The speech recognition system is modular and can be extended to:
- Support additional recognition engines
- Add voice commands for specific actions
- Implement continuous listening mode
- Add speaker identification

## Troubleshooting

If voice input is not working:

1. **Check Browser Support**: Voice input works best in Chrome/Edge
2. **Verify Microphone**: Test microphone in system settings
3. **Check Permissions**: Ensure microphone access is granted
4. **Install Dependencies**: Run the setup script for Electron mode
5. **Check Console**: Look for error messages in developer tools

For additional support, check the console logs which provide detailed error information.
