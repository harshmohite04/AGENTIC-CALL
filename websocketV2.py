import os
import asyncio
import websockets
import json
import queue
import threading
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
import requests
import pygame
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import time
import wave
import tempfile

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "zT03pEAEi0VHKciJODfn")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 8000  # Smaller chunks for lower latency
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0  # seconds of silence before processing

# Initialize pygame mixer for low-latency audio playback
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.init()

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, text_queue):
        self.text_queue = text_queue
        self.current_text = ""
        self.in_thinking = False
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Skip reasoning tokens (thinking)
        if "<think>" in token:
            self.in_thinking = True
            return
        elif "</think>" in token:
            self.in_thinking = False
            return
        elif self.in_thinking:
            return
            
        self.current_text += token
        # Send text chunks when we have enough words or hit punctuation
        if len(self.current_text.split()) >= 8 or token in '.!?':
            clean_text = self.current_text.strip()
            if clean_text and not any(tag in clean_text for tag in ['<think>', '</think>']):
                self.text_queue.put(clean_text)
            self.current_text = ""
    
    def on_llm_end(self, response, **kwargs) -> None:
        if self.current_text.strip() and not self.in_thinking:
            clean_text = self.current_text.strip()
            if not any(tag in clean_text for tag in ['<think>', '</think>']):
                self.text_queue.put(clean_text)

class LowLatencyVoiceAssistant:
    def __init__(self, text_only_mode=False):
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.llm_text_queue = queue.Queue()
        self.tts_audio_queue = queue.Queue()
        self.audio_playback_queue = queue.Queue()  # Sequential audio playback
        self.text_only_mode = text_only_mode  # For debugging without audio
        
        # Initialize LLM with streaming
        self.streaming_handler = StreamingCallbackHandler(self.llm_text_queue)
        self.chat = ChatGroq(
            model="qwen/qwen3-32b",  # Changed from deepseek to avoid reasoning tokens
            temperature=0,
            streaming=True,
            callbacks=[self.streaming_handler]
        )
        
        self.system_prompt = """
        You are a helpful education counselor for coaching classes.
        - Fees: Class 11 ₹45,000 (offline), Class 12 ₹30,000, MH-CET ₹60,000.
        - Centers: Pune (FC Road), Mumbai (Andheri), Nashik (College Road).
        - Faculty: Prof. Arjun Deshmukh (Physics), Dr. Priya Kulkarni (Chemistry), etc.
        Speak Hinglish in a natural, empathetic tone. Keep responses concise and conversational.
        DO NOT include any reasoning or thinking process in your response. Only provide the final answer.
        """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{user_input}")
        ])
        
        self.is_listening = False
        self.is_speaking = False
        self.audio_lock = threading.Lock()  # Prevent concurrent audio playback
        self.current_tts_process = None  # Track current TTS process
        self.stop_current_audio = threading.Event()  # Signal to stop current audio
        
    async def stream_deepgram_transcription(self):
        """Stream audio to Deepgram for real-time transcription"""
        # Updated WebSocket URL with proper parameters
        uri = "wss://api.deepgram.com/v1/listen"
        
        extra_headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}"
        }
        
        # Connection parameters
        params = {
            "model": "nova-2",
            "language": "hi",
            "punctuate": "true",
            "smart_format": "true",
            "interim_results": "true",
            "endpointing": "300",
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE),
            "channels": str(CHANNELS)
        }
        
        # Add parameters to URI
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_uri = f"{uri}?{param_string}"
        
        connection_active = True
        
        try:
            async with websockets.connect(
                full_uri, 
                extra_headers=extra_headers,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                print("🔗 Connected to Deepgram")
                
                async def send_audio():
                    nonlocal connection_active
                    try:
                        while connection_active:
                            if not self.audio_queue.empty():
                                audio_data = self.audio_queue.get()
                                
                                # Ensure audio is in correct format
                                if audio_data.dtype != np.int16:
                                    # Convert float32 to int16
                                    audio_int16 = (audio_data * 32767).astype(np.int16)
                                else:
                                    audio_int16 = audio_data
                                
                                # Send audio bytes
                                await websocket.send(audio_int16.tobytes())
                            
                            await asyncio.sleep(0.02)  # Slightly longer interval
                            
                    except websockets.exceptions.ConnectionClosed:
                        print("🔌 WebSocket connection closed")
                        connection_active = False
                    except Exception as e:
                        print(f"⚠ Audio send error: {e}")
                        connection_active = False
                
                async def receive_transcripts():
                    nonlocal connection_active
                    try:
                        async for message in websocket:
                            try:
                                data = json.loads(message)
                                
                                # Debug: print message type
                                if 'type' in data:
                                    if data['type'] == 'Results':
                                        # Handle results message
                                        if 'channel' in data and 'alternatives' in data['channel']:
                                            alternatives = data['channel']['alternatives']
                                        elif 'results' in data:
                                            alternatives = data['results']['channels'][0]['alternatives']
                                        else:
                                            continue
                                        
                                        if alternatives and len(alternatives) > 0:
                                            transcript = alternatives[0].get('transcript', '')
                                            is_final = data.get('is_final', False)
                                            
                                            if is_final and transcript.strip():
                                                print(f"🗣 Final: {transcript}")
                                                self.transcript_queue.put(transcript.strip())
                                            elif transcript.strip():
                                                print(f"🗣 Interim: {transcript}")
                                    
                                    elif data['type'] == 'Metadata':
                                        print(f"📋 Metadata: {data.get('transaction_key', 'N/A')}")
                                    
                                    elif data['type'] == 'UtteranceEnd':
                                        print("⏹ Utterance ended")
                                
                            except (json.JSONDecodeError, KeyError) as e:
                                print(f"⚠ Parsing error: {e}")
                                continue
                                
                    except websockets.exceptions.ConnectionClosed:
                        print("🔌 Transcript receiver connection closed")
                        connection_active = False
                    except Exception as e:
                        print(f"⚠ Transcript receive error: {e}")
                        connection_active = False
                
                # Send keep-alive message
                async def keep_alive():
                    while connection_active:
                        try:
                            await websocket.ping()
                            await asyncio.sleep(30)
                        except:
                            connection_active = False
                            break
                
                # Run all tasks concurrently
                await asyncio.gather(
                    send_audio(),
                    receive_transcripts(),
                    keep_alive(),
                    return_exceptions=True
                )
                
        except Exception as e:
            print(f"❌ Deepgram connection error: {e}")
            print("🔄 Falling back to batch transcription...")
            await self.fallback_transcription()
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio status: {status}")
        
        if not self.is_speaking:  # Only listen when not speaking
            # Check for silence
            volume = np.linalg.norm(indata) * 10
            if volume > SILENCE_THRESHOLD:
                self.is_listening = True
                # Flatten and ensure correct format
                audio_flat = indata.flatten()
                # Convert to int16 if needed
                if audio_flat.dtype == np.float32:
                    audio_int16 = (audio_flat * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_flat.astype(np.int16)
                
                self.audio_queue.put(audio_int16)
    
    async def stream_llm_response(self):
        """Process transcripts and stream LLM responses"""
        while True:
            try:
                if not self.transcript_queue.empty():
                    transcript = self.transcript_queue.get()
                    print(f"🤖 Processing: {transcript}")
                    
                    # Stop any current audio playback when new transcript comes in
                    self.stop_current_audio.set()
                    
                    # Wait a moment for audio to stop
                    await asyncio.sleep(0.1)
                    
                    # Clear any remaining audio in queue
                    while not self.audio_playback_queue.empty():
                        try:
                            self.audio_playback_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    # Reset the stop signal
                    self.stop_current_audio.clear()
                    
                    # Reset the streaming handler for new response
                    self.streaming_handler.current_text = ""
                    
                    # Stream LLM response
                    formatted_prompt = self.prompt_template.format_messages(user_input=transcript)
                    response = await asyncio.to_thread(self.chat.invoke, formatted_prompt)
                    
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ LLM error: {e}")
    
    async def stream_tts_audio(self):
        """Convert text chunks to speech and queue audio sequentially"""
        while True:
            try:
                if not self.llm_text_queue.empty():
                    text_chunk = self.llm_text_queue.get()
                    print(f"🔊 TTS: {text_chunk}")
                    
                    # Filter out any remaining thinking tokens
                    if '<think>' in text_chunk or '</think>' in text_chunk:
                        continue
                    
                    # Clean the text
                    clean_text = text_chunk.replace('<think>', '').replace('</think>', '').strip()
                    if not clean_text:
                        continue
                    
                    # If text-only mode, just print and continue
                    if self.text_only_mode:
                        print(f"💬 AUDIO: {clean_text}")
                        continue
                    
                    # Use ElevenLabs streaming endpoint
                    audio_data = await self.synthesize_speech_streaming(clean_text)
                    if audio_data:
                        # Add to sequential playback queue
                        self.audio_playback_queue.put((audio_data, clean_text))
                
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ TTS error: {e}")
    
    async def synthesize_speech_streaming(self, text):
        """Stream TTS from ElevenLabs with better error handling"""
        # Check if ElevenLabs API key is valid
        if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "your_elevenlabs_api_key":
            print("⚠ ElevenLabs API key not configured, using fallback TTS")
            return ("fallback_tts", text)
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.6,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            },
            "output_format": "mp3_22050_32"
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, stream=True, timeout=10)
            if response.status_code == 200:
                audio_chunks = []
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        audio_chunks.append(chunk)
                return b''.join(audio_chunks)
            elif response.status_code == 401:
                print("❌ ElevenLabs API key invalid, using fallback TTS")
                return ("fallback_tts", text)
            else:
                print(f"❌ ElevenLabs error: {response.status_code}")
                return ("fallback_tts", text)
        except Exception as e:
            print(f"❌ TTS synthesis error: {e}")
            return ("fallback_tts", text)
    
    def stop_all_audio(self):
        """Stop all currently playing audio"""
        try:
            # Stop pygame mixer
            pygame.mixer.stop()
            pygame.mixer.music.stop()
            
            # Set stop signal
            self.stop_current_audio.set()
            
            # Terminate current TTS process if running
            if self.current_tts_process and self.current_tts_process.is_alive():
                # For system TTS, we can't easily stop it, but we can ignore it
                pass
                
        except Exception as e:
            print(f"⚠ Error stopping audio: {e}")
    
    def play_fallback_tts(self, text):
        """Play fallback TTS in a controlled manner"""
        def tts_worker():
            try:
                if os.name == 'nt':  # Windows
                    import subprocess
                    ps_command = f'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak("{text}")'
                    
                    # Store the process so we can track it
                    process = subprocess.Popen([
                        'powershell', '-Command', ps_command
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Wait for completion or stop signal
                    while process.poll() is None:
                        if self.stop_current_audio.is_set():
                            process.terminate()
                            break
                        time.sleep(0.1)
                
                else:
                    # Linux/Mac fallback
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.setProperty('volume', 0.9)
                        
                        # Try to set Hindi voice
                        voices = engine.getProperty('voices')
                        for voice in voices:
                            if any(lang in voice.id.lower() for lang in ['hi', 'hindi', 'india']):
                                engine.setProperty('voice', voice.id)
                                break
                        
                        # Check for stop signal before speaking
                        if not self.stop_current_audio.is_set():
                            engine.say(text)
                            engine.runAndWait()
                        
                        engine.stop()
                        
                    except ImportError:
                        print(f"💬 TTS TEXT: {text}")
                        
            except Exception as e:
                print(f"❌ Fallback TTS error: {e}")
                print(f"💬 TTS TEXT: {text}")
        
        # Start TTS in separate thread
        self.current_tts_process = threading.Thread(target=tts_worker, daemon=True)
        self.current_tts_process.start()
        
        return self.current_tts_process
    
    def play_audio_stream(self):
        """Play audio chunks sequentially - FIXED VERSION"""
        while True:
            try:
                if not self.audio_playback_queue.empty():
                    with self.audio_lock:  # Ensure only one audio plays at a time
                        # Stop any currently playing audio first
                        self.stop_all_audio()
                        
                        # Wait a moment for audio to stop
                        time.sleep(0.1)
                        
                        self.is_speaking = True
                        audio_item = self.audio_playback_queue.get()
                        
                        # Handle tuple format (audio_data, text)
                        if isinstance(audio_item, tuple):
                            audio_data, text = audio_item
                        else:
                            audio_data = audio_item
                            text = "Unknown"
                        
                        # Check if we should stop before playing
                        if self.stop_current_audio.is_set():
                            self.is_speaking = False
                            continue
                        
                        # Handle fallback TTS
                        if isinstance(audio_data, tuple) and audio_data[0] == "fallback_tts":
                            print(f"🔊 Playing fallback TTS: {audio_data[1][:50]}...")
                            tts_thread = self.play_fallback_tts(audio_data[1])
                            
                            # Wait for TTS to complete or stop signal
                            while tts_thread.is_alive():
                                if self.stop_current_audio.is_set():
                                    break
                                time.sleep(0.1)
                            
                            # Extra wait for TTS to finish
                            if not self.stop_current_audio.is_set():
                                time.sleep(1)
                            
                            self.is_speaking = False
                            continue
                        
                        # Handle real audio data
                        if len(audio_data) < 100:  # Too small to be real audio
                            print("⚠ Audio data too small, skipping...")
                            self.is_speaking = False
                            continue
                        
                        # Create unique temporary file
                        timestamp = str(int(time.time() * 1000000))  # More unique timestamp
                        
                        # Detect file format and use appropriate extension
                        if audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb'):
                            # MP3 format
                            temp_filename = f"temp_audio_{timestamp}.mp3"
                            file_format = "mp3"
                        elif audio_data.startswith(b'RIFF'):
                            # WAV format
                            temp_filename = f"temp_audio_{timestamp}.wav"
                            file_format = "wav"
                        else:
                            # Default to MP3
                            temp_filename = f"temp_audio_{timestamp}.mp3"
                            file_format = "mp3"
                        
                        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
                        
                        try:
                            # Check if we should stop before writing file
                            if self.stop_current_audio.is_set():
                                self.is_speaking = False
                                continue
                            
                            # Write audio data
                            with open(temp_path, "wb") as f:
                                f.write(audio_data)
                            
                            print(f"🔊 Playing {file_format.upper()}: {text[:50]}...")
                            
                            # Play audio using pygame mixer music (only one can play at a time)
                            pygame.mixer.music.load(temp_path)
                            pygame.mixer.music.play()
                            
                            # Wait for music to finish or stop signal
                            while pygame.mixer.music.get_busy():
                                if self.stop_current_audio.is_set():
                                    pygame.mixer.music.stop()
                                    break
                                time.sleep(0.1)
                            
                            if not self.stop_current_audio.is_set():
                                print("✅ Audio finished playing")
                            else:
                                print("⏹ Audio stopped by user")
                            
                        except Exception as e:
                            print(f"❌ Audio playback error: {e}")
                            print(f"💬 [AUDIO FAILED]: {text}")
                        
                        finally:
                            # Clean up file
                            try:
                                if os.path.exists(temp_path):
                                    # Wait a bit before deleting to ensure file is not in use
                                    time.sleep(0.2)
                                    os.unlink(temp_path)
                            except Exception as e:
                                print(f"⚠ File cleanup error: {e}")
                            
                            self.is_speaking = False
                
                time.sleep(0.1)  # Small delay when no audio to play
                
            except Exception as e:
                print(f"❌ Audio stream error: {e}")
                self.is_speaking = False
                time.sleep(1)
    
    async def fallback_transcription(self):
        """Fallback to batch transcription if streaming fails"""
        print("🔄 Using fallback batch transcription...")
        
        while True:
            try:
                # Collect audio for 3 seconds
                audio_buffer = []
                start_time = time.time()
                
                while time.time() - start_time < 3.0:
                    if not self.audio_queue.empty():
                        audio_buffer.append(self.audio_queue.get())
                    await asyncio.sleep(0.1)
                
                if audio_buffer:
                    # Combine audio chunks
                    combined_audio = np.concatenate(audio_buffer)
                    
                    # Save to temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        with wave.open(temp_file.name, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes((combined_audio * 32767).astype(np.int16).tobytes())
                        
                        # Transcribe using REST API
                        transcript = await self.transcribe_batch(temp_file.name)
                        if transcript:
                            self.transcript_queue.put(transcript)
                        
                        # Clean up
                        os.unlink(temp_file.name)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Fallback transcription error: {e}")
                await asyncio.sleep(1)
    
    async def transcribe_batch(self, filepath):
        """Batch transcription using Deepgram REST API"""
        try:
            url = "https://api.deepgram.com/v1/listen"
            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav"
            }
            params = {
                "model": "nova-2",
                "language": "hi",
                "punctuate": "true",
                "smart_format": "true"
            }
            
            with open(filepath, 'rb') as audio_file:
                response = requests.post(url, headers=headers, params=params, data=audio_file)
                
                if response.status_code == 200:
                    result = response.json()
                    transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
                    if transcript.strip():
                        print(f"🗣 Batch: {transcript}")
                        return transcript.strip()
                else:
                    print(f"❌ Batch transcription error: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Batch transcription error: {e}")
        
        return None
    
    async def start_streaming(self):
        """Start all streaming components"""
        print("🟢 Starting Low-Latency Voice Assistant")
        
        # Start audio playback thread
        audio_thread = threading.Thread(target=self.play_audio_stream, daemon=True)
        audio_thread.start()
        
        # Start audio input stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self.audio_callback,
            blocksize=CHUNK_SIZE
        ):
            print("🎤 Listening... Speak naturally!")
            
            # Run all async components concurrently
            await asyncio.gather(
                self.stream_deepgram_transcription(),
                self.stream_llm_response(),
                self.stream_tts_audio()
            )

# Additional optimizations
class OptimizedAudioBuffer:
    """Circular buffer for low-latency audio processing"""
    def __init__(self, size=SAMPLE_RATE * 2):  # 2 seconds buffer
        self.buffer = np.zeros(size)
        self.write_pos = 0
        self.size = size
    
    def write(self, data):
        data_len = len(data)
        if self.write_pos + data_len <= self.size:
            self.buffer[self.write_pos:self.write_pos + data_len] = data
        else:
            # Wrap around
            first_part = self.size - self.write_pos
            self.buffer[self.write_pos:] = data[:first_part]
            self.buffer[:data_len - first_part] = data[first_part:]
        
        self.write_pos = (self.write_pos + data_len) % self.size
    
    def read_recent(self, samples):
        start_pos = (self.write_pos - samples) % self.size
        if start_pos + samples <= self.size:
            return self.buffer[start_pos:start_pos + samples]
        else:
            part1 = self.buffer[start_pos:]
            part2 = self.buffer[:samples - len(part1)]
            return np.concatenate([part1, part2])

async def main():
    print("🎛 Voice Assistant Setup")
    print("=" * 40)
    
    # Check ElevenLabs API key
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "your_elevenlabs_api_key":
        print("⚠ ElevenLabs API key not configured!")
        print("📝 To get high-quality TTS:")
        print("   1. Go to https://elevenlabs.io")
        print("   2. Sign up for free account")
        print("   3. Get your API key from profile")
        print("   4. Add to .env file: ELEVENLABS_API_KEY=your_key_here")
        print()
        print("💡 For now, we'll use system TTS as fallback")
        print()
    
    print("🎛 Choose mode:")
    print("1. Full voice mode (with TTS)")
    print("2. Text-only mode (no TTS, faster)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    text_only = choice == "2"
    
    if text_only:
        print("📝 Running in text-only mode")
    else:
        print("🔊 Running in full voice mode")
        if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "your_elevenlabs_api_key":
            print("🔄 Will use system TTS fallback")
    
    print("=" * 40)
    
    assistant = LowLatencyVoiceAssistant(text_only_mode=text_only)
    try:
        await assistant.start_streaming()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())