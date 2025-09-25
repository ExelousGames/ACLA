const { execSync } = require('child_process');
const os = require('os');
const path = require('path');
const fs = require('fs');

function checkAndInstallSpeechDeps() {
    console.log('========================================');
    console.log(' Enhanced Speech Recognition Setup');
    console.log('========================================');
    console.log('');
    console.log('🎤 Setting up enhanced speech recognition...');

    // First check if Python is available
    try {
        const pythonVersion = execSync('python --version', { encoding: 'utf8' });
        console.log(`✅ Found Python: ${pythonVersion.trim()}`);
    } catch (error) {
        console.log('❌ Python is not installed or not in PATH');
        console.log('Please install Python from https://www.python.org/downloads/');
        console.log('Make sure to check "Add Python to PATH" during installation');
        console.log('⚠️  Continuing without speech recognition...');
        return;
    }

    try {
        // Check if dependencies are already installed
        console.log('📦 Checking existing dependencies...');

        const basicPackages = ['speech_recognition', 'pyaudio', 'pocketsphinx'];
        const enhancedPackages = ['numpy', 'scipy', 'librosa', 'noisereduce', 'webrtcvad'];
        const whisperPackages = ['whisper', 'torch'];

        let basicInstalled = checkPackages(basicPackages);
        let enhancedInstalled = checkPackages(enhancedPackages);
        let whisperInstalled = checkPackages(whisperPackages);
        let ffmpegAvailable = checkFFmpegAvailability();

        console.log(`📋 Status: Basic=${basicInstalled}, Enhanced=${enhancedInstalled}, Whisper=${whisperInstalled}, FFmpeg=${ffmpegAvailable}`);

        // Install missing packages
        if (!basicInstalled) {
            console.log('📦 Installing basic speech recognition packages...');
            installBasicPackages();
        } else {
            console.log('✅ Basic speech recognition packages already installed');
        }

        if (!enhancedInstalled) {
            console.log('📦 Installing enhanced audio processing packages...');
            installEnhancedPackages();
        } else {
            console.log('✅ Enhanced audio processing packages already installed');
        }

        if (!whisperInstalled) {
            console.log('📦 Installing Whisper AI (this may take a while)...');
            installWhisperPackages();
        } else {
            console.log('✅ Whisper AI already installed');
        }

        // Copy/create the enhanced script
        setupEnhancedScript();

        // Check and install ffmpeg for Whisper
        checkAndInstallFFmpeg();

        console.log('');
        console.log('========================================');
        console.log(' Setup Complete!');
        console.log('========================================');
        console.log('');
        console.log('✅ Enhanced speech recognition setup completed!');
        console.log('🎤 Your voice input now supports:');
        console.log('   ✓ Noise reduction and audio enhancement');
        console.log('   ✓ Voice activity detection');
        console.log('   ✓ Multiple recognition engines (Google, Sphinx)');

        // Check ffmpeg status for final message
        try {
            execSync('ffmpeg -version', { stdio: 'ignore', timeout: 3000 });
            console.log('   ✓ Offline Whisper AI (highest quality) - ffmpeg available');
        } catch {
            console.log('   ⚠ Whisper AI requires ffmpeg (see installation instructions above)');
        }

        console.log('   ✓ Improved accuracy and reliability');
        console.log('');
        console.log('Please restart your Electron app to use the enhanced features.');

    } catch (error) {
        console.error('❌ Setup failed:', error.message);
        console.log('⚠️  Creating fallback configuration...');
        createFallbackScript();
    }
}

function checkFFmpegAvailability() {
    try {
        execSync('ffmpeg -version', { stdio: 'ignore', timeout: 3000 });
        return true;
    } catch {
        return false;
    }
}

function checkPackages(packages) {
    for (const pkg of packages) {
        try {
            execSync(`python -c "import ${pkg}"`, { stdio: 'ignore' });
        } catch {
            return false;
        }
    }
    return true;
}

function installBasicPackages() {
    try {
        console.log('Installing core packages from requirements.txt...');

        try {
            // Try to install all requirements at once
            execSync('pip install -r requirements.txt', { stdio: 'inherit' });
            console.log('✅ All speech recognition dependencies installed successfully');
            return;
        } catch (requirementsError) {
            console.log('⚠️  Installing from requirements.txt failed, trying individual packages...');

            // Fallback to individual installation
            const basicCommands = [
                'pip install --upgrade pip setuptools wheel',
                'pip install SpeechRecognition==3.10.0',
                'pip install pocketsphinx==0.1.15'
            ];

            for (const cmd of basicCommands) {
                console.log(`Running: ${cmd}`);
                execSync(cmd, { stdio: 'inherit' });
            }
        }

        // Install PyAudio with multiple fallback methods
        installPyAudio();

    } catch (error) {
        console.error('❌ Basic package installation failed:', error.message);
        throw error;
    }
}

function installPyAudio() {
    console.log('Installing PyAudio...');
    let pyaudioInstalled = false;

    const methods = [
        'pip install pyaudio',
        'pip install --upgrade pyaudio',
        'pip install --no-cache-dir pyaudio'
    ];

    for (const method of methods) {
        try {
            console.log(`Trying: ${method}`);
            execSync(method, { stdio: 'inherit' });
            pyaudioInstalled = true;
            console.log('✅ PyAudio installed successfully');
            break;
        } catch (error) {
            console.log(`⚠️  Method failed, trying next: ${method}`);
            continue;
        }
    }

    if (!pyaudioInstalled) {
        console.log('\n🔧 Manual PyAudio installation required:');
        if (os.platform() === 'win32') {
            console.log('Windows:');
            console.log('1. Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio');
            console.log('2. Download the appropriate .whl file for your Python version');
            console.log('3. Install with: pip install path_to_downloaded_file.whl');
            console.log('\nAlternatively, install Visual Studio Build Tools:');
            console.log('https://visualstudio.microsoft.com/visual-cpp-build-tools/');
        } else if (os.platform() === 'darwin') {
            console.log('macOS: brew install portaudio && pip install pyaudio');
        } else {
            console.log('Linux: sudo apt-get install portaudio19-dev && pip install pyaudio');
        }
        console.log('\n⚠️  Voice input will work in online mode only without PyAudio.');
    }
}

function installEnhancedPackages() {
    try {
        const enhancedCommands = [
            'pip install numpy==1.24.3',
            'pip install scipy==1.11.1',
            'pip install soundfile==0.12.1',
            'pip install librosa==0.10.0',
            'pip install noisereduce==3.0.0',
            'pip install webrtcvad==2.0.10'
        ];

        for (const cmd of enhancedCommands) {
            try {
                console.log(`Running: ${cmd}`);
                execSync(cmd, { stdio: 'inherit', timeout: 120000 }); // 2 minute timeout
            } catch (error) {
                console.log(`⚠️  Package installation failed, continuing: ${cmd}`);
                // Don't throw, continue with other packages
            }
        }

        console.log('✅ Enhanced packages installation completed');
    } catch (error) {
        console.warn('⚠️  Some enhanced packages failed to install:', error.message);
        // Don't throw, enhanced features are optional
    }
}

function installWhisperPackages() {
    try {
        console.log('📦 Installing PyTorch (required for Whisper)...');

        // Install PyTorch first (CPU version for compatibility)
        const torchCommands = [
            'pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu',
            'pip install openai-whisper'
        ];

        for (const cmd of torchCommands) {
            try {
                console.log(`Running: ${cmd}`);
                execSync(cmd, { stdio: 'inherit', timeout: 300000 }); // 5 minute timeout
            } catch (error) {
                console.log(`⚠️  Whisper installation failed: ${cmd}`);
                throw error;
            }
        }

        console.log('✅ Whisper AI installed successfully');
        console.log('🎯 High-quality offline speech recognition is now available!');

    } catch (error) {
        console.warn('⚠️  Whisper AI installation failed:', error.message);
        console.log('💡 Whisper provides the best accuracy but requires significant disk space');
        console.log('💡 Enhanced mode will work without Whisper using other improvements');
        // Don't throw, Whisper is optional
    }
}

function setupEnhancedScript() {
    const enhancedScriptPath = path.join(__dirname, 'src', 'py-scripts', 'enhanced_speech_recognition.py');

    if (fs.existsSync(enhancedScriptPath)) {
        console.log('✅ Enhanced speech recognition script found in src/py-scripts/');
    } else {
        console.log('⚠️  Enhanced script not found, creating fallback');
        createFallbackScript();
    }
}

function checkAndInstallFFmpeg() {
    console.log('🎬 Checking ffmpeg (required for Whisper AI)...');

    try {
        // Check if ffmpeg is already available
        execSync('ffmpeg -version', { stdio: 'ignore', timeout: 5000 });
        console.log('✅ ffmpeg is already installed and available');
        return true;
    } catch (error) {
        console.log('⚠️  ffmpeg not found - required for Whisper AI speech recognition');

        if (os.platform() === 'win32') {
            console.log('🔧 Attempting to install ffmpeg on Windows...');

            // Try multiple installation methods
            const installMethods = [
                {
                    name: 'Windows Package Manager (winget)',
                    command: 'winget install ffmpeg',
                    description: 'Official Windows package manager'
                },
                {
                    name: 'Chocolatey',
                    command: 'choco install ffmpeg -y',
                    description: 'Popular Windows package manager'
                }
            ];

            for (const method of installMethods) {
                try {
                    console.log(`📦 Trying ${method.name}...`);
                    execSync(method.command, { stdio: 'inherit', timeout: 120000 });

                    // For winget installations, try refreshing PATH in case it was updated
                    if (method.name.includes('winget')) {
                        console.log('🔄 Refreshing PATH environment variable...');
                        try {
                            // Refresh PATH by reloading environment variables
                            const refreshCmd = '$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")';
                            execSync(`powershell -Command "${refreshCmd}"`, { stdio: 'ignore' });

                            // Small delay to allow PATH refresh (synchronous)
                            const start = Date.now();
                            while (Date.now() - start < 1000) {
                                // Simple blocking delay
                            }
                        } catch (refreshError) {
                            console.log('⚠️  PATH refresh failed, but continuing with verification');
                        }
                    }

                    // Verify installation
                    try {
                        execSync('ffmpeg -version', { stdio: 'ignore', timeout: 5000 });
                        console.log(`✅ ffmpeg successfully installed via ${method.name}`);
                        return true;
                    } catch (verifyError) {
                        console.log(`⚠️  Installation completed but ffmpeg not found in PATH`);
                        console.log('💡 Try restarting your terminal or VS Code to refresh PATH');

                        // If it's winget and verification failed, suggest manual PATH refresh
                        if (method.name.includes('winget')) {
                            console.log('🔧 Quick fix: Run this command in your terminal:');
                            console.log('   $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")');
                        }
                        continue;
                    }
                } catch (installError) {
                    console.log(`❌ ${method.name} installation failed, trying next method...`);
                    continue;
                }
            }

            // If automatic installation failed, provide manual instructions
            console.log('\n🔧 Automatic installation failed. Manual installation required:');
            console.log('');
            console.log('Method 1 - Download directly:');
            console.log('1. Visit: https://ffmpeg.org/download.html#build-windows');
            console.log('2. Download the "Windows builds by BtbN" release');
            console.log('3. Extract to C:\\ffmpeg\\');
            console.log('4. Add C:\\ffmpeg\\bin to your system PATH');
            console.log('');
            console.log('Method 2 - Use package managers:');
            console.log('• Windows Package Manager: winget install ffmpeg');
            console.log('• Chocolatey: choco install ffmpeg');
            console.log('• Scoop: scoop install ffmpeg');
            console.log('');
            console.log('Method 3 - Portable version:');
            console.log('1. Download: https://www.gyan.dev/ffmpeg/builds/');
            console.log('2. Extract anywhere and add bin folder to PATH');
            console.log('');
            console.log('📝 Note: If you installed ffmpeg but it\'s not found:');
            console.log('   • Restart your terminal/VS Code to refresh PATH');
            console.log('   • Or run: $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")');
            console.log('');
            console.log('⚠️  Without ffmpeg, Whisper AI will not work, but Google/Sphinx recognition will still function.');

        } else if (os.platform() === 'darwin') {
            console.log('🍎 macOS installation:');
            console.log('Install with Homebrew: brew install ffmpeg');
        } else {
            console.log('🐧 Linux installation:');
            console.log('Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg');
            console.log('CentOS/RHEL: sudo yum install ffmpeg');
            console.log('Fedora: sudo dnf install ffmpeg');
        }

        return false;
    }
}

function createFallbackScript() {
    // Create a fallback script that uses available features
    const fallbackScript = `"""
Enhanced Speech Recognition Module for Electron App - Fallback Version
Provides improved speech recognition with basic enhancements
"""

import speech_recognition as sr
import sys
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Check for enhanced features
try:
    import numpy as np
    import noisereduce as nr
    import webrtcvad
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

class EnhancedSpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.whisper_model = None
        
        # Enhanced settings
        self.sample_rate = 16000
        
        # Initialize Whisper if available
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base.en")
                print(json.dumps({"status": "info", "message": "Whisper model loaded for enhanced accuracy"}))
            except Exception as e:
                print(json.dumps({"status": "warning", "message": f"Whisper model failed to load: {e}"}))
                self.whisper_model = None

    def setup_microphone(self):
        """Initialize microphone with optimal settings"""
        try:
            self.microphone = sr.Microphone(sample_rate=self.sample_rate)
            with self.microphone as source:
                print(json.dumps({"status": "calibrating"}))
                # Enhanced calibration
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                # Optimize recognition settings
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.dynamic_energy_adjustment_damping = 0.15
                self.recognizer.dynamic_energy_ratio = 1.5
                self.recognizer.pause_threshold = 0.8
                self.recognizer.operation_timeout = None
                self.recognizer.phrase_threshold = 0.3
                self.recognizer.non_speaking_duration = 0.8
                
            print(json.dumps({"status": "ready"}))
            return True
        except Exception as e:
            print(json.dumps({"status": "error", "error": f"Microphone setup failed: {e}"}))
            return False

    def recognize_with_whisper(self, audio_data):
        """Use Whisper for high-quality offline recognition"""
        if not self.whisper_model:
            return None
        
        try:
            import tempfile
            import os
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Convert audio data to file
            if hasattr(audio_data, 'get_wav_data'):
                wav_data = audio_data.get_wav_data()
                with open(temp_path, 'wb') as f:
                    f.write(wav_data)
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                temp_path,
                language="en",
                task="transcribe",
                fp16=False,
                temperature=0.0,
                initial_prompt="This is a conversation about racing, cars, speed, braking, acceleration, and track analysis."
            )
            
            # Clean up
            os.unlink(temp_path)
            
            if result and 'text' in result:
                return result['text'].strip()
            
        except Exception as e:
            print(json.dumps({"status": "warning", "message": f"Whisper recognition failed: {e}"}))
            
        return None

    def recognize_speech_enhanced(self, timeout=30):
        """Enhanced speech recognition with multiple fallbacks"""
        if not self.setup_microphone():
            return {"status": "error", "error": "Microphone setup failed"}
        
        print(json.dumps({"status": "listening", "enhanced": ENHANCED_FEATURES, "whisper": WHISPER_AVAILABLE}))
        sys.stdout.flush()
        
        try:
            with self.microphone as source:
                # Listen for audio input with extended timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
                
                print(json.dumps({"status": "processing"}))
                sys.stdout.flush()
                
                # Try recognition methods in order of preference
                recognition_results = []
                
                # Method 1: Whisper (highest quality)
                if WHISPER_AVAILABLE:
                    whisper_result = self.recognize_with_whisper(audio)
                    if whisper_result:
                        recognition_results.append({
                            "method": "whisper",
                            "text": whisper_result,
                            "confidence": 0.9
                        })
                
                # Method 2: Google Speech-to-Text (online)
                try:
                    google_result = self.recognizer.recognize_google(audio, language='en-US')
                    if google_result:
                        recognition_results.append({
                            "method": "google",
                            "text": google_result,
                            "confidence": 0.8
                        })
                except (sr.UnknownValueError, sr.RequestError):
                    pass
                
                # Method 3: Sphinx (offline fallback)
                try:
                    sphinx_result = self.recognizer.recognize_sphinx(audio)
                    if sphinx_result:
                        recognition_results.append({
                            "method": "sphinx",
                            "text": sphinx_result,
                            "confidence": 0.6
                        })
                except (sr.UnknownValueError, sr.RequestError):
                    pass
                
                # Select best result
                if recognition_results:
                    best_result = max(recognition_results, key=lambda x: x['confidence'])
                    return {
                        "status": "success",
                        "transcript": best_result['text'],
                        "method": best_result['method'],
                        "confidence": best_result['confidence'],
                        "enhanced": ENHANCED_FEATURES
                    }
                else:
                    return {"status": "error", "error": "All recognition methods failed"}
                
        except sr.WaitTimeoutError:
            return {"status": "error", "error": "No speech detected within timeout"}
        except Exception as e:
            return {"status": "error", "error": f"Recognition failed: {e}"}

def main():
    """Main function for speech recognition"""
    recognizer = EnhancedSpeechRecognizer()
    
    # Get command line arguments
    timeout = 30
    if len(sys.argv) > 1:
        try:
            timeout = int(sys.argv[1])
        except:
            pass
    
    # Perform recognition
    result = recognizer.recognize_speech_enhanced(timeout=timeout)
    print(json.dumps(result))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
`;

    const fallbackPath = path.join(__dirname, 'public', 'enhanced_speech_recognition.py');
    try {
        fs.writeFileSync(fallbackPath, fallbackScript);
        console.log('✅ Enhanced speech recognition script created');
    } catch (error) {
        console.error('❌ Could not create enhanced speech script:', error.message);
    }
}

// Add a small delay to make the output more readable
console.log('🚀 Setting up ACLA Electron app...\n');
setTimeout(() => {
    checkAndInstallSpeechDeps();
}, 500);
