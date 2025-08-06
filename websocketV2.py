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

# TTS Options
import pyttsx3

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DEEPGRAM_API_KEY:
    print("❌ DEEPGRAM_API_KEY not found")
    exit(1)
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found")
    exit(1)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 8000
BUFFER_DURATION = 1.0
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
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
        print("🤖 LLM generation started...")
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Check for interruption but don't stop - let it complete if almost done
        if self.assistant_ref.user_interrupted:
            return  # Just ignore new tokens, don't add them
        
        self.current_text += token
        self.full_response += token
    
    def on_llm_end(self, response, **kwargs) -> None:
        self.is_generating = False
        if self.full_response.strip() and not self.assistant_ref.user_interrupted:
            print(f"✅ LLM response complete: {self.full_response[:50]}...")
            self.text_queue.put(self.full_response.strip())
        else:
            print("❌ LLM response cancelled due to interruption")
        self.current_text = ""
        self.full_response = ""

class FreeTTSVoiceAssistant:
    def __init__(self, tts_method="pyttsx3"):
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.llm_text_queue = queue.Queue()
        self.tts_method = tts_method
        
        # Context management
        self.context_stack = collections.deque(maxlen=3)
        
        # VAD setup
        self.vad = webrtcvad.Vad(2)
        self.audio_buffer = collections.deque(maxlen=MAX_BUFFER_FRAMES)
        self.is_speech_detected = False
        self.silence_frames = 0
        self.speech_frames = 0
        
        # Control flags
        self.is_listening = True
        self.is_speaking = False
        self.user_interrupted = False
        self.connection_active = True
        self.llm_processing = False
        self.tts_task = None
        
        # Add interrupt event for immediate stopping
        self.interrupt_event = asyncio.Event()
        
        self.init_tts_engine()
        self.streaming_handler = StreamingCallbackHandler(self.llm_text_queue, self)
        
        self.chat = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            streaming=True,
            callbacks=[self.streaming_handler],
            groq_api_key=GROQ_API_KEY
        )
        
        self.system_prompt = """You are an experienced customer care agent for Vishwakarma Classes. Handle client queries about classes only. Be friendly and respond in Hinglish. Keep responses short (1-2 lines). Use full words instead of abbreviations (Mister instead of MR, Rupees instead of RS). Start with warm greetings."""
            
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{conversation_context}\n\nCurrent question: {user_input}")
        ])
        
    def init_tts_engine(self):
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 0.9)
            print("✅ TTS engine initialized")
        except Exception as e:
            print(f"❌ TTS init failed: {e}")
            self.tts_engine = None

    def process_audio_frame(self, audio_data):
        """Process audio frame with WebRTC VAD"""
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
        """FIXED: Only stop current operations, don't stop processing pipeline"""
        print("🛑 User speaking - interrupting current output")
        self.user_interrupted = True
        
        # Stop TTS immediately
        if self.is_speaking and self.tts_engine:
            try:
                self.tts_engine.stop()
                print("🔇 TTS stopped")
            except:
                pass
            finally:
                self.is_speaking = False
        
        # Cancel TTS task if running
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            print("❌ TTS task cancelled")
        
        # Clear TTS queue (old responses), but keep transcript processing
        self._clear_queue(self.llm_text_queue)
        
        # Set interrupt event
        self.interrupt_event.set()

    def _clear_queue(self, q):
        """Helper to clear a queue"""
        while not q.empty():
            try:
                q.get_nowait()
            except:
                break

    def audio_callback(self, indata, frames, time_info, status):
        """FIXED: Removed blocking time.sleep() and proper interruption"""
        audio_flat = indata.flatten()
        
        # Always buffer audio for pre-roll
        self.audio_buffer.append(audio_flat.copy())
        
        # Check for speech
        is_speech = self.process_audio_frame(audio_flat)
        
        if is_speech:
            if not self.is_speech_detected:
                # Speech started - interrupt current operations if needed
                if self.is_speaking or self.streaming_handler.is_generating:
                    self.handle_interruption()
                
                # Speech started - send buffer
                self.is_speech_detected = True
                self.user_interrupted = False  # Reset for new input
                self.silence_frames = 0
                
                print("🎤 New speech detected")
                
                # Send pre-roll buffer
                for buffered_audio in self.audio_buffer:
                    if buffered_audio.dtype == np.float32:
                        audio_int16 = (buffered_audio * 32767).astype(np.int16)
                    else:
                        audio_int16 = buffered_audio.astype(np.int16)
                    self.audio_queue.put(audio_int16)
            
            # Continue sending current frame
            if audio_flat.dtype == np.float32:
                audio_int16 = (audio_flat * 32767).astype(np.int16)
            else:
                audio_int16 = audio_flat.astype(np.int16)
            self.audio_queue.put(audio_int16)
            
            self.speech_frames += 1
            self.silence_frames = 0  # Reset silence counter
            
        elif self.is_speech_detected:
            # Potential end of speech
            self.silence_frames += 1
            if self.silence_frames > 8:  # End speech after several silent frames
                print("🤫 Speech ended")
                self.is_speech_detected = False
                self.silence_frames = 0
                self.speech_frames = 0

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
            "keep_alive": "true"  # Keep connection alive
        }
                
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_uri = f"{uri}?{param_string}"
        
        retry_count = 0
        max_retries = 5
        
        while self.connection_active and retry_count < max_retries:
            try:
                print(f"🔄 Connecting to Deepgram... (attempt {retry_count + 1})")
                async with websockets.connect(full_uri, extra_headers=extra_headers, 
                                            ping_interval=20, ping_timeout=10) as websocket:
                    print("✅ Connected to Deepgram")
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
                                print(f"❌ Audio send error: {e}")
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
                                            print(f"🗣 Final: {transcript}")
                                            # Always process new transcripts
                                            self.transcript_queue.put(transcript)
                                        elif transcript:
                                            print(f"⏳ Interim: {transcript}", end='\r')
                        except Exception as e:
                            print(f"❌ Transcript receive error: {e}")
                            return
                    
                    await asyncio.gather(
                        send_audio(),
                        receive_transcripts(),
                        return_exceptions=True
                    )
                    
            except Exception as e:
                retry_count += 1
                print(f"❌ Connection error (attempt {retry_count}): {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(min(retry_count * 2, 10))
                else:
                    print("❌ Max retries reached, giving up")
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
        """FIXED: Process all transcripts, handle interruptions properly"""
        while self.connection_active:
            try:
                if not self.transcript_queue.empty():
                    transcript = self.transcript_queue.get()
                    
                    # Don't check if already processing - handle all inputs
                    self.llm_processing = True
                    
                    print(f"🤖 Processing transcript: {transcript}")
                    
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
                        
                        # Add to context if not interrupted and we got a response
                        if not self.user_interrupted and self.streaming_handler.full_response.strip():
                            self.context_stack.append((transcript, self.streaming_handler.full_response.strip()))
                            print(f"💾 Added to context: {len(self.context_stack)} conversations")
                        
                    except Exception as e:
                        print(f"❌ LLM processing error: {e}")
                    finally:
                        self.llm_processing = False
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                print(f"❌ LLM loop error: {e}")
                self.llm_processing = False
                await asyncio.sleep(1)

    async def stream_tts_audio(self):
        """FIXED: Better TTS handling - always ready for new responses"""
        while self.connection_active:
            try:
                if not self.llm_text_queue.empty() and not self.is_speaking:
                    text = self.llm_text_queue.get()
                    
                    # Skip if interrupted, but continue processing
                    if self.user_interrupted:
                        print("🚫 TTS skipped due to interruption")
                        continue
                    
                    print(f"🔊 Starting TTS: {text[:50]}...")
                    
                    # Create TTS task
                    self.tts_task = asyncio.create_task(self.synthesize_speech_pyttsx3(text))
                    
                    try:
                        await self.tts_task
                        print("✅ TTS completed")
                    except asyncio.CancelledError:
                        print("❌ TTS task was cancelled")
                    except Exception as e:
                        print(f"❌ TTS task error: {e}")
                    finally:
                        self.tts_task = None
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                print(f"❌ TTS loop error: {e}")
                await asyncio.sleep(1)

    async def synthesize_speech_pyttsx3(self, text):
        """TTS with proper interruption support"""
        try:
            if self.tts_engine and not self.user_interrupted:
                self.is_speaking = True
                print(f"🎵 Speaking: {text}")
                
                def speak():
                    try:
                        if not self.user_interrupted:
                            self.tts_engine.say(text)
                            self.tts_engine.runAndWait()
                    except Exception as e:
                        print(f"❌ TTS speak error: {e}")
                
                await asyncio.to_thread(speak)
            
        except Exception as e:
            print(f"❌ TTS synthesis error: {e}")
        finally:
            self.is_speaking = False

    async def start_streaming(self):
        """Start the voice assistant"""
        print("🟢 Voice Assistant Ready")
        print("🎤 Speak naturally - I'll respond in Hinglish!")
        print("💬 Real-time conversation - interrupt anytime!")
        print("🔄 Starting all services...")
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self.audio_callback,
                blocksize=FRAME_SIZE,
                dtype=np.float32
            ):
                print("✅ Audio stream started")
                
                # Start all services
                tasks = [
                    asyncio.create_task(self.stream_deepgram_transcription()),
                    asyncio.create_task(self.stream_llm_response()),
                    asyncio.create_task(self.stream_tts_audio())
                ]
                
                print("✅ All services started - Ready for conversation!")
                print("🎯 Try speaking, then interrupt mid-response to test real-time!")
                
                # Wait for all tasks
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            print(f"❌ Stream error: {e}")
            self.connection_active = False

async def main():
    assistant = FreeTTSVoiceAssistant(tts_method="pyttsx3")
    try:
        await assistant.start_streaming()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        assistant.connection_active = False
    except Exception as e:
        print(f"❌ Main error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
