"""
Enhanced Speech Recognition Module for Electron App
Provides high-quality offline speech recognition with noise reduction and VAD
"""

import speech_recognition as sr
import sys
import json
import numpy as np
import tempfile
import os
import time
import threading
from typing import Optional, Dict, Any
import queue
import warnings

# Optional imports for enhanced functionality
try:
    import noisereduce as nr
    import librosa
    import soundfile as sf
    import webrtcvad
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False
    print(json.dumps({"status": "warning", "message": "Enhanced features not available, using basic recognition"}))

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class EnhancedSpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.audio_queue = queue.Queue()
        self.stop_listening = False
        self.whisper_model = None
        
        # Enhanced settings
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # seconds
        self.vad_aggressiveness = 2  # 0-3, higher = more aggressive
        
        # Initialize Whisper if available
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base.en")
                print(json.dumps({"status": "info", "message": "Whisper model loaded for enhanced accuracy"}))
            except Exception as e:
                print(json.dumps({"status": "warning", "message": f"Whisper model failed to load: {e}"}))
                self.whisper_model = None
        
        # VAD setup
        self.vad = None
        if ENHANCED_FEATURES:
            try:
                self.vad = webrtcvad.Vad(self.vad_aggressiveness)
            except:
                pass

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

    def enhance_audio(self, audio_data, sample_rate):
        """Apply noise reduction and audio enhancement"""
        if not ENHANCED_FEATURES:
            return audio_data
        
        try:
            # Convert to numpy array
            if hasattr(audio_data, 'get_wav_data'):
                wav_data = audio_data.get_wav_data()
                audio_np = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32)
                audio_np = audio_np / 32768.0  # Normalize to [-1, 1]
            else:
                audio_np = np.array(audio_data, dtype=np.float32)
            
            # Apply noise reduction
            enhanced_audio = nr.reduce_noise(y=audio_np, sr=sample_rate, stationary=False, prop_decrease=0.8)
            
            # Apply spectral subtraction for further enhancement
            enhanced_audio = self._spectral_subtraction(enhanced_audio, sample_rate)
            
            # Convert back to AudioData format
            enhanced_audio_int16 = (enhanced_audio * 32767).astype(np.int16)
            enhanced_wav = enhanced_audio_int16.tobytes()
            
            return sr.AudioData(enhanced_wav, sample_rate, 2)
        except Exception as e:
            print(json.dumps({"status": "warning", "message": f"Audio enhancement failed: {e}"}))
            return audio_data

    def _spectral_subtraction(self, audio, sample_rate):
        """Apply spectral subtraction for noise reduction"""
        try:
            # Simple spectral subtraction
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frames = int(0.5 * sample_rate / 512)
            noise_magnitude = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Subtract noise
            alpha = 2.0  # Over-subtraction factor
            enhanced_magnitude = magnitude - alpha * noise_magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio
        except:
            return audio

    def detect_speech(self, audio_chunk):
        """Use VAD to detect if audio contains speech"""
        if not self.vad or not ENHANCED_FEATURES:
            return True  # Assume speech if VAD not available
        
        try:
            # Convert audio to the format expected by webrtcvad (16-bit PCM, specific sample rates)
            if hasattr(audio_chunk, 'get_wav_data'):
                wav_data = audio_chunk.get_wav_data()
                audio_bytes = wav_data[44:]  # Skip WAV header
                
                # webrtcvad expects 10, 20, or 30 ms frames at 8, 16, 32, or 48 kHz
                frame_duration = 20  # ms
                frame_length = int(self.sample_rate * frame_duration / 1000)
                
                # Process in chunks
                is_speech = False
                for i in range(0, len(audio_bytes), frame_length * 2):
                    frame = audio_bytes[i:i + frame_length * 2]
                    if len(frame) == frame_length * 2:
                        try:
                            if self.vad.is_speech(frame, self.sample_rate):
                                is_speech = True
                                break
                        except:
                            continue
                
                return is_speech
        except:
            return True  # Default to assuming speech

    def recognize_with_whisper(self, audio_data):
        """Use Whisper for high-quality offline recognition"""
        if not self.whisper_model:
            return None
        
        try:
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
                best_of=1,
                beam_size=1,
                patience=1.0,
                length_penalty=1.0,
                suppress_tokens="-1",
                initial_prompt="This is a conversation about racing, cars, speed, braking, acceleration, and track analysis."
            )
            
            # Clean up
            os.unlink(temp_path)
            
            if result and 'text' in result:
                return result['text'].strip()
            
        except Exception as e:
            print(json.dumps({"status": "warning", "message": f"Whisper recognition failed: {e}"}))
            
        return None

    def recognize_speech_continuous(self, timeout=30, phrase_time_limit=10):
        """Enhanced continuous speech recognition with multiple fallbacks"""
        if not self.setup_microphone():
            return {"status": "error", "error": "Microphone setup failed"}
        
        print(json.dumps({"status": "listening", "enhanced": ENHANCED_FEATURES, "whisper": WHISPER_AVAILABLE}))
        sys.stdout.flush()
        
        start_time = time.time()
        collected_audio = []
        silence_threshold = 2.0  # seconds of silence to stop
        last_speech_time = start_time
        
        try:
            with self.microphone as source:
                while time.time() - start_time < timeout:
                    try:
                        # Listen for short chunks
                        audio_chunk = self.recognizer.listen(
                            source, 
                            timeout=1.0, 
                            phrase_time_limit=self.chunk_duration
                        )
                        
                        # Check if this chunk contains speech
                        contains_speech = self.detect_speech(audio_chunk)
                        
                        if contains_speech:
                            collected_audio.append(audio_chunk)
                            last_speech_time = time.time()
                            print(json.dumps({"status": "speech_detected"}))
                            sys.stdout.flush()
                        else:
                            # Check if we should stop due to silence
                            if time.time() - last_speech_time > silence_threshold and collected_audio:
                                break
                    
                    except sr.WaitTimeoutError:
                        # Check if we should stop due to silence
                        if time.time() - last_speech_time > silence_threshold and collected_audio:
                            break
                        continue
                
                # Process collected audio
                if not collected_audio:
                    return {"status": "error", "error": "No speech detected"}
                
                print(json.dumps({"status": "processing", "chunks_collected": len(collected_audio)}))
                sys.stdout.flush()
                
                # Combine audio chunks
                combined_audio = self._combine_audio_chunks(collected_audio)
                
                # Enhance audio quality
                enhanced_audio = self.enhance_audio(combined_audio, self.sample_rate)
                
                # Try recognition methods in order of preference
                recognition_results = []
                
                # Method 1: Whisper (highest quality)
                if WHISPER_AVAILABLE:
                    whisper_result = self.recognize_with_whisper(enhanced_audio)
                    if whisper_result:
                        recognition_results.append({
                            "method": "whisper",
                            "text": whisper_result,
                            "confidence": 0.9
                        })
                
                # Method 2: Google Speech-to-Text (online)
                try:
                    google_result = self.recognizer.recognize_google(enhanced_audio, language='en-US')
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
                    sphinx_result = self.recognizer.recognize_sphinx(enhanced_audio)
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
                        "alternatives": recognition_results
                    }
                else:
                    return {"status": "error", "error": "All recognition methods failed"}
                
        except Exception as e:
            return {"status": "error", "error": f"Recognition failed: {e}"}

    def _combine_audio_chunks(self, audio_chunks):
        """Combine multiple audio chunks into a single AudioData object"""
        if not audio_chunks:
            return None
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        try:
            # Combine audio data
            combined_frames = b''
            sample_rate = audio_chunks[0].sample_rate
            sample_width = audio_chunks[0].sample_width
            
            for chunk in audio_chunks:
                combined_frames += chunk.get_wav_data()[44:]  # Skip WAV headers except first
            
            # Create header for combined audio
            header = audio_chunks[0].get_wav_data()[:44]
            combined_wav_data = header + combined_frames
            
            return sr.AudioData(combined_wav_data, sample_rate, sample_width)
        except:
            return audio_chunks[0]  # Fallback to first chunk

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
    result = recognizer.recognize_speech_continuous(timeout=timeout)
    print(json.dumps(result))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
