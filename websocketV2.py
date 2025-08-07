import os
import asyncio
import websockets
import json
import queue
import threading
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
import pygame
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import time
import traceback
import collections
import webrtcvad
import logging
import aiohttp

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")  # e.g., "21m00Tcm4TlvDq8ikWAM"

if not ELEVENLABS_API_KEY:
    print("ELEVENLABS_API_KEY not found")
    exit(1)

if not DEEPGRAM_API_KEY:
    print("DEEPGRAM_API_KEY not found")
    exit(1)
if not GROQ_API_KEY:
    print("GROQ_API_KEY not found")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
BUFFER_DURATION = 1.0  # seconds
MAX_BUFFER_FRAMES = int(BUFFER_DURATION * 1000 / FRAME_DURATION)

# Initialize pygame mixer quietly
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.init()

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, text_queue, assistant_ref):
        self.text_queue = text_queue
        self.assistant_ref = assistant_ref
        self.current_text = ""
        self.full_response = ""
        self.is_generating = False
    
    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.is_generating = True
        self.current_text = ""
        self.full_response = ""
        logging.info("🤖 LLM generation started...")
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.assistant_ref.user_interrupted:
            return
        self.current_text += token
        self.full_response += token
    
    def on_llm_end(self, response, **kwargs) -> None:
        self.is_generating = False
        if self.full_response.strip() and not self.assistant_ref.user_interrupted:
            logging.info(f"LLM response complete: {self.full_response[:50]}...")
            self.text_queue.put(self.full_response.strip())
        else:
            logging.info("LLM response cancelled due to interruption")
        self.current_text = ""
        self.full_response = ""

class FreeTTSVoiceAssistant:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.llm_text_queue = queue.Queue()
        
        # Context management
        self.context_stack = collections.deque(maxlen=3)
        
        # VAD setup
        self.vad = webrtcvad.Vad(3) # Less sensitive than 2 or 3
        self.silence_threshold = 20  # Increased silence threshold
        self.audio_buffer = collections.deque(maxlen=MAX_BUFFER_FRAMES)
        self.is_speech_detected = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.consecutive_speech_threshold = 3  # Require more consecutive speech
        
        # Control flags
        self.is_listening = True
        self.is_speaking = False
        self.user_interrupted = False
        self.connection_active = True
        self.llm_processing = False
        self.tts_task = None
        
        # Add interrupt event for immediate stopping
        self.interrupt_event = asyncio.Event()
        self.tts_lock = asyncio.Lock()  # Lock for TTS operations
        
        self.streaming_handler = StreamingCallbackHandler(self.llm_text_queue, self)
        
        self.chat = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            streaming=True,
            callbacks=[self.streaming_handler],
            groq_api_key=GROQ_API_KEY
        )
        
        self.system_prompt = """You are an experienced customer care agent for Vishwakarma Classes. Handle client queries about classes only. Be friendly and respond in Hinglish. Keep responses short (1-2 lines). Use full words instead of abbreviations (Mister instead of MR, Rupees instead of RS). Start with warm greetings initially, dont always say namaste as you said it first"""
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{conversation_context}\n\nCurrent question: {user_input}")
        ])

    def process_audio_frame(self, audio_data):
        """Process audio frame with WebRTC VAD and energy threshold"""
        try:
            # Convert to int16 if needed
            if audio_data.dtype == np.float32:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            # Ensure correct frame size for VAD
            frame_bytes = audio_int16.tobytes()
            if len(frame_bytes) < 2 * FRAME_SIZE:
                return False
            
            # Use only the required frame size
            vad_frame = frame_bytes[:2 * FRAME_SIZE]
            return self.vad.is_speech(vad_frame, SAMPLE_RATE)
        except Exception:
            return False

    def handle_interruption(self):
        """More graceful interruption handling"""
        if not self.is_speaking and not self.streaming_handler.is_generating:
            return
        
        logging.info("🛑 User speaking - interrupting current output")
        self.user_interrupted = True
        
        # Stop any ongoing TTS
        if self.is_speaking:
            try:
                pygame.mixer.music.stop()
                logging.info("🔇 TTS stopped")
            except Exception as e:
                logging.error(f"❌ TTS stop error: {e}")
            finally:
                self.is_speaking = False
        
        # Clear queues more selectively
        self._clear_queue(self.llm_text_queue)
        self.interrupt_event.set()

    def _clear_queue(self, q):
        """Helper to clear a queue"""
        while not q.empty():
            try:
                q.get_nowait()
            except:
                break

    def audio_callback(self, indata, frames, time_info, status):
        """Robust audio processing with noise suppression"""
        if not self.connection_active:
            return
            
        try:
            audio_flat = indata.flatten()
            self.audio_buffer.append(audio_flat.copy())
            
            # More conservative speech detection
            is_speech = self.process_audio_frame(audio_flat)
            audio_int16 = (audio_flat * 32767).astype(np.int16) if audio_flat.dtype == np.float32 else audio_flat.astype(np.int16)
            
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                
                if not self.is_speech_detected and self.speech_frames > self.consecutive_speech_threshold:
                    self.is_speech_detected = True
                    self.user_interrupted = False
                    logging.info("🎤 New speech detected")
                    
                    # Handle interruption if needed
                    if self.is_speaking or self.streaming_handler.is_generating:
                        self.handle_interruption()
                    
                    # Send buffer
                    for buffered in self.audio_buffer:
                        self.audio_queue.put(buffered)
                
                self.audio_queue.put(audio_int16)
                
            elif self.is_speech_detected:
                self.silence_frames += 1
                if self.silence_frames > self.silence_threshold:
                    self.is_speech_detected = False
                    self.speech_frames = 0
                    logging.info("🤫 Speech ended")
                    
        except Exception as e:
            logging.error(f"❌ Audio callback error: {e}")

    async def stream_deepgram_transcription(self):
        """Deepgram connection with better keepalive"""
        uri = "wss://api.deepgram.com/v1/listen"
        extra_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
                
        params = {
            "model": "nova-2-general",
            "language": "hi",
            "punctuate": "true",
            "smart_format": "true", 
            "interim_results": "true",
            "endpointing": "300",
            "vad_events": "true",
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE),
            "channels": str(CHANNELS),
            "keep_alive": "true"
        }
                
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_uri = f"{uri}?{param_string}"
        
        retry_count = 0
        max_retries = 5
        
        while self.connection_active and retry_count < max_retries:
            try:
                logging.info(f"🔄 Connecting to Deepgram... (attempt {retry_count + 1})")
                async with websockets.connect(full_uri, extra_headers=extra_headers, 
                                            ping_interval=20, ping_timeout=10) as websocket:
                    logging.info("✅ Connected to Deepgram")
                    retry_count = 0
                    
                    # Send keepalive periodically
                    last_keepalive = time.time()
                    
                    async def send_audio():
                        nonlocal last_keepalive
                        while self.connection_active:
                            try:
                                current_time = time.time()
                                
                                # Send audio if available
                                if not self.audio_queue.empty():
                                    audio_data = self.audio_queue.get(timeout=0.1)
                                    await websocket.send(audio_data.tobytes())
                                    last_keepalive = current_time
                                
                                # Send keepalive if no audio for too long
                                elif current_time - last_keepalive > 10:
                                    await websocket.send(json.dumps({"type": "KeepAlive"}))
                                    last_keepalive = current_time
                                
                                await asyncio.sleep(0.01)
                                
                            except queue.Empty:
                                continue
                            except Exception as e:
                                logging.error(f"❌ Audio send error: {e}")
                                break
                    
                    async def receive_transcripts():
                        try:
                            async for message in websocket:
                                if not self.connection_active:
                                    break
                                    
                                data = json.loads(message)
                                if data.get('type') == 'Results':
                                    # Handle both response formats
                                    alternatives = None
                                    if 'channel' in data:
                                        alternatives = data['channel'].get('alternatives', [])
                                    elif 'results' in data:
                                        channels = data['results'].get('channels', [])
                                        if channels:
                                            alternatives = channels[0].get('alternatives', [])
                                    
                                    if alternatives and len(alternatives) > 0:
                                        transcript = alternatives[0].get('transcript', '').strip()
                                        is_final = data.get('is_final', False)
                                        
                                        if is_final and transcript and len(transcript) > 2:
                                            logging.info(f"🗣 Final: {transcript}")
                                            self.transcript_queue.put(transcript)
                                        elif transcript:
                                            print(f"⏳ Interim: {transcript}", end='\r')
                        except Exception as e:
                            logging.error(f"❌ Transcript receive error: {e}")
                            return
                    
                    await asyncio.gather(
                        send_audio(),
                        receive_transcripts(),
                        return_exceptions=True
                    )
                    
            except Exception as e:
                retry_count += 1
                logging.error(f"❌ Connection error (attempt {retry_count}): {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(min(retry_count * 2, 10))
                else:
                    logging.error("❌ Max retries reached, giving up")
                    self.connection_active = False

    def build_conversation_context(self):
        """Build context from recent conversations"""
        if not self.context_stack:
            return ""
        
        context_parts = []
        for user_msg, assistant_msg in self.context_stack:
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {assistant_msg}")
        
        return "\n".join(context_parts)

    async def stream_llm_response(self):
        """Process all transcripts, handle interruptions properly"""
        while self.connection_active:
            try:
                if not self.transcript_queue.empty():
                    transcript = self.transcript_queue.get()
                    
                    self.llm_processing = True
                    logging.info(f"🤖 Processing transcript: {transcript}")
                    
                    # Reset interruption flag for new input
                    self.user_interrupted = False
                    self.interrupt_event.clear()
                    
                    context = self.build_conversation_context()
                    
                    # Reset streaming handler
                    self.streaming_handler.current_text = ""
                    self.streaming_handler.full_response = ""
                    
                    try:
                        formatted_prompt = self.prompt_template.format_messages(
                            conversation_context=context,
                            user_input=transcript
                        )
                        
                        # Run LLM in thread
                        response = await asyncio.to_thread(self.chat.invoke, formatted_prompt)
                        
                        # Add to context if not interrupted
                        if not self.user_interrupted and self.streaming_handler.full_response.strip():
                            self.context_stack.append((transcript, self.streaming_handler.full_response.strip()))
                            logging.info(f"💾 Added to context: {len(self.context_stack)} conversations")
                        
                    except Exception as e:
                        logging.error(f"❌ LLM processing error: {e}")
                    finally:
                        self.llm_processing = False
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logging.error(f"❌ LLM loop error: {e}")
                self.llm_processing = False
                await asyncio.sleep(1)

    async def stream_tts_audio(self):
        """Convert complete text responses to speech"""
        while self.connection_active:
            try:
                if not self.llm_text_queue.empty() and not self.is_speaking:
                    text = self.llm_text_queue.get()
                    
                    if self.user_interrupted:
                        logging.info("🚫 TTS skipped due to interruption")
                        continue
                    
                    logging.info(f"🔊 Starting TTS: {text[:50]}...")
                    
                    try:
                        # Run with timeout
                        await asyncio.wait_for(
                            self.synthesize_speech_streaming(text),
                            timeout=30.0
                        )
                        logging.info("✅ TTS completed")
                    except asyncio.TimeoutError:
                        logging.warning("⌛ TTS timed out")
                    except asyncio.CancelledError:
                        logging.info("❌ TTS cancelled")
                    except Exception as e:
                        logging.error(f"❌ TTS error: {e}")
                    finally:
                        self.is_speaking = False
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logging.error(f"❌ TTS loop error: {e}")
                self.is_speaking = False
                await asyncio.sleep(1)

    async def synthesize_speech_streaming(self, text):
        """Stream TTS from ElevenLabs"""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  # Important for Hindi/English
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            async with self.tts_lock:
                self.is_speaking = True
                
                # Use aiohttp for async HTTP requests
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status == 200:
                            audio_stream = BytesIO()
                            async for chunk in response.content.iter_any():
                                if self.user_interrupted:
                                    raise asyncio.CancelledError("User interrupted TTS")
                                audio_stream.write(chunk)
                            
                            audio_stream.seek(0)
                            
                            # Play the audio
                            try:
                                sound = pygame.mixer.Sound(audio_stream)
                                sound.play()
                                
                                # Wait for playback to finish or interruption
                                start_time = time.time()
                                while pygame.mixer.get_busy() and not self.user_interrupted:
                                    if time.time() - start_time > 30:  # Max 30 seconds per TTS
                                        break
                                    await asyncio.sleep(0.1)
                                
                                # Stop if interrupted
                                if self.user_interrupted:
                                    sound.stop()
                                
                            except pygame.error as e:
                                logging.error(f"❌ Pygame sound error: {e}")
                                # Fallback to music player
                                audio_stream.seek(0)
                                pygame.mixer.music.load(audio_stream)
                                pygame.mixer.music.play()
                                while pygame.mixer.music.get_busy() and not self.user_interrupted:
                                    await asyncio.sleep(0.1)
                                if self.user_interrupted:
                                    pygame.mixer.music.stop()
                        else:
                            error_text = await response.text()
                            logging.error(f"❌ ElevenLabs error: {response.status} - {error_text}")
        except asyncio.CancelledError:
            logging.info("TTS interrupted by user")
            raise
        except Exception as e:
            logging.error(f"❌ TTS synthesis error: {e}")
            raise
        finally:
            self.is_speaking = False

    async def start_streaming(self):
        """Start the voice assistant"""
        logging.info("🟢 Voice Assistant Ready")
        logging.info("🎤 Speak naturally - I'll respond in Hinglish!")
        logging.info("💬 Real-time conversation - interrupt anytime!")
        logging.info("🔄 Starting all services...")
        
        try:
            # Start audio stream
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self.audio_callback,
                blocksize=FRAME_SIZE,
                dtype=np.float32
            )
            with stream:
                logging.info("✅ Audio stream started")
                
                # Start all services
                tasks = [
                    asyncio.create_task(self.stream_deepgram_transcription()),
                    asyncio.create_task(self.stream_llm_response()),
                    asyncio.create_task(self.stream_tts_audio()),
                ]
                
                logging.info("✅ All services started - Ready for conversation!")
                logging.info("🎯 Try speaking, then interrupt mid-response to test real-time!")
                
                # Wait for all tasks
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logging.error(f"❌ Stream error: {e}")
            self.connection_active = False

async def main():
    assistant = FreeTTSVoiceAssistant()
    try:
        await assistant.start_streaming()
    except KeyboardInterrupt:
        logging.info("\n🛑 Shutting down gracefully...")
        assistant.connection_active = False
        
        # Cancel all running tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        
        # Wait for tasks to cancel
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logging.info("👋 Goodbye!")
    except Exception as e:
        logging.error(f"❌ Main error: {e}")
        traceback.print_exc()
    finally:
        # Ensure pygame is properly quit
        pygame.mixer.quit()
        logging.info("✅ Pygame mixer quit")

if __name__ == "__main__":
    asyncio.run(main())
