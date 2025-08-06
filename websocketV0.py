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
import tempfile
import traceback

# TTS Options - choose one by uncommenting
# Option 1: pyttsx3 (offline, cross-platform)
import pyttsx3

# Option 2: gTTS (online, free, no API key needed)
# from gtts import gTTS

# Option 3: Edge TTS (online, free, Microsoft voices)
# import edge_tts

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Debug: Check if environment variables are loaded
if not DEEPGRAM_API_KEY:
    print("❌ DEEPGRAM_API_KEY not found in environment variables")
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found in environment variables")
    print("💡 Make sure you have a .env file with GROQ_API_KEY=your_key_here")
    exit(1)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 8000 # 5 seconds of audio at 16 kHz lower it is less is the latency but higher cpu usage--improve
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0

# Initialize pygame mixer for low-latency audio playback
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.init()

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, text_queue):
        self.text_queue = text_queue
        self.current_text = ""
        self.full_response = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.current_text += token
        self.full_response += token
        print(f"LLM Token: {token}", end='', flush=True)
    
    def on_llm_end(self, response, **kwargs) -> None:
        print("\nLLM Response Complete")
        if self.full_response.strip():
            self.text_queue.put(self.full_response.strip())
        self.current_text = ""
        self.full_response = ""

class FreeTTSVoiceAssistant:
    def __init__(self, tts_method="pyttsx3"):
        self.audio_queue = queue.Queue()   # think unnecessary --improve
        self.transcript_queue = queue.Queue() # think unnecessary --improve
        self.llm_text_queue = queue.Queue()
        self.tts_audio_queue = queue.Queue()
        self.tts_method = tts_method
        
        # Initialize TTS engine based on selected method
        self.init_tts_engine()
        
        # Initialize LLM with streaming
        self.streaming_handler = StreamingCallbackHandler(self.llm_text_queue)
        
        # Ensure GROQ_API_KEY is available
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
            
        self.chat = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            streaming=True,
            callbacks=[self.streaming_handler],
            groq_api_key=GROQ_API_KEY  # Explicitly pass the API key
        )
        
        self.system_prompt = """       
        You are an experienced customer care agent. you are task is to handle the queries of the clients who will be inquiring for classes. Do not get out of context, like if some one is asking the question which are not related to the classes than you are supposed to tell them you cannot answer that question. Be as friendly as possible, and you have to talk to customer in hinglish strictly.
        In some cases like abbrevations like MR, RS, its literally saying the character, instead give complete form of it like mister, rupees, etc for all others. 
        You are strictly supposed to sound friendly and in hinglish.
        You are strictly support to talk in short, which is only 1-2 lines response to any user talk.
        """
            
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{user_input}")
        ])
        
        self.is_listening = False
        self.is_speaking = False
        self.connection_active = True
        
    def init_tts_engine(self):
        """Initialize the selected TTS engine"""
        if self.tts_method == "pyttsx3":
            try:
                self.tts_engine = pyttsx3.init()
                # Configure pyttsx3 settings
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Try to find a female voice or use first available
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                    else:
                        self.tts_engine.setProperty('voice', voices[0].id)
                
                # Set speech rate and volume
                self.tts_engine.setProperty('rate', 180)  # Speed of speech
                self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
                print("✅ pyttsx3 TTS engine initialized")
            except Exception as e:
                print(f"❌ Failed to initialize pyttsx3: {e}")
                self.tts_engine = None

    async def stream_deepgram_transcription(self):
        """Stream audio to Deepgram for real-time transcription"""
        uri = "wss://api.deepgram.com/v1/listen"
        
        extra_headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}"
        }
                
        params = {
            "model": "nova-2-general",
            "language": "hi",
            "punctuate": "true",
            "smart_format": "true", 
            "interim_results": "true",
            "endpointing": "300",
            "vad_events": "true", # toggle false to check --improve
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE),
            "channels": str(CHANNELS)
        }
                
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_uri = f"{uri}?{param_string}"
        
        while self.connection_active:
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
                        try:
                            while self.connection_active:
                                if not self.audio_queue.empty():
                                    audio_data = self.audio_queue.get()
                                    
                                    if audio_data.dtype != np.int16:
                                        audio_int16 = (audio_data * 32767).astype(np.int16)
                                    else:
                                        audio_int16 = audio_data
                                    
                                    await websocket.send(audio_int16.tobytes())
                                
                                await asyncio.sleep(0.02)
                                
                        except websockets.exceptions.ConnectionClosed:
                            print("🔌 Audio send connection closed")
                        except Exception as e:
                            print(f"⚠ Audio send error: {e}")
                    
                    async def receive_transcripts():
                        try:
                            async for message in websocket:
                                try:
                                    data = json.loads(message)
                                    
                                    if 'type' in data:
                                        if data['type'] == 'Results':
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
                        except Exception as e:
                            print(f"⚠ Transcript receive error: {e}")
                    
                    async def keep_alive():
                        while self.connection_active:
                            try:
                                await websocket.ping()
                                await asyncio.sleep(30)
                            except:
                                break
                    
                    await asyncio.gather(
                        send_audio(),
                        receive_transcripts(),
                        keep_alive(),
                        return_exceptions=True
                    )
                    
            except Exception as e:
                print(f"❌ Deepgram connection error: {e}")
                print("🔄 Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
                continue
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio status: {status}")
        
        if not self.is_speaking:              
            volume = np.linalg.norm(indata) * 10
            if volume > SILENCE_THRESHOLD:
                self.is_listening = True                
                audio_flat = indata.flatten()                
                if audio_flat.dtype == np.float32:
                    audio_int16 = (audio_flat * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_flat.astype(np.int16)
                
                self.audio_queue.put(audio_int16)
    
    async def stream_llm_response(self):
        """Process transcripts and stream complete LLM responses"""
        while True:
            try:
                if not self.transcript_queue.empty():
                    transcript = self.transcript_queue.get()
                    print(f"\n🤖 Processing: {transcript}")
                                        
                    self.streaming_handler.current_text = ""
                    self.streaming_handler.full_response = ""
                                        
                    formatted_prompt = self.prompt_template.format_messages(user_input=transcript)
                    print("\nLLM Response: ", end='', flush=True)
                    await asyncio.to_thread(self.chat.invoke, formatted_prompt)
                    
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ LLM error: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)
    
    async def stream_tts_audio(self):
        """Convert complete text responses to speech using selected TTS method"""
        while True:
            try:
                if not self.llm_text_queue.empty():
                    complete_text = self.llm_text_queue.get()
                    print(f"\n🔊 TTS Generating for: {complete_text}")
                    
                    if self.tts_method == "pyttsx3":
                        await self.synthesize_speech_pyttsx3(complete_text)
                    elif self.tts_method == "gtts":
                        audio_data = await self.synthesize_speech_gtts(complete_text)
                        if audio_data:
                            self.tts_audio_queue.put(audio_data)
                    elif self.tts_method == "edge_tts":
                        audio_data = await self.synthesize_speech_edge_tts(complete_text)
                        if audio_data:
                            self.tts_audio_queue.put(audio_data)
                
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ TTS error: {e}")
                await asyncio.sleep(1)
    
    async def synthesize_speech_pyttsx3(self, text):
        """Use pyttsx3 for offline TTS (plays directly)"""
        try:
            if self.tts_engine:
                self.is_speaking = True
                
                # Run TTS in a separate thread to avoid blocking
                def speak():
                    try:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                    finally:
                        self.is_speaking = False
                
                await asyncio.to_thread(speak)
            else:
                print("❌ TTS engine not available")
        except Exception as e:
            print(f"❌ pyttsx3 synthesis error: {e}")
            self.is_speaking = False
    
    async def synthesize_speech_gtts(self, text):
        """Use Google Text-to-Speech (online, free, no API key)"""
        try:
            from gtts import gTTS
            import io
            
            # Create gTTS object
            tts = gTTS(text=text, lang='hi', slow=False)  # 'hi' for Hindi, 'en' for English
            
            # Save to BytesIO buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.getvalue()
            
        except Exception as e:
            print(f"❌ gTTS synthesis error: {e}")
            return None
    
    async def synthesize_speech_edge_tts(self, text):
        """Use Microsoft Edge TTS (online, free)"""
        try:
            import edge_tts
            import io
            
            # Available voices: en-US-AriaNeural, en-US-JennyNeural, hi-IN-SwaraNeural, etc.
            voice = "hi-IN-SwaraNeural"  # Hindi voice, change to "en-US-AriaNeural" for English
            
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Edge TTS synthesis error: {e}")
            return None
    
    def play_audio_stream(self):
        """Play audio chunks as they arrive (for gTTS and Edge TTS)"""
        while True:
            try:
                if not self.tts_audio_queue.empty():
                    self.is_speaking = True
                    audio_data = self.tts_audio_queue.get()
                    
                    # Use BytesIO instead of temp file
                    audio_stream = BytesIO(audio_data)
                    audio_stream.seek(0)
                    
                    try:
                        pygame.mixer.music.load(audio_stream)
                        pygame.mixer.music.play()
                        
                        # Wait for audio to finish
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                            
                    finally:
                        audio_stream.close()
                        self.is_speaking = False
                
                time.sleep(0.05)
            except Exception as e:
                print(f"❌ Audio playback error: {e}")
                self.is_speaking = False
    
    async def start_streaming(self):
        """Start all streaming components"""
        print(f"🟢 Starting Voice Assistant with {self.tts_method.upper()} TTS")
        
        # Start audio playback thread (only needed for gTTS and Edge TTS)
        if self.tts_method in ["gtts", "edge_tts"]:
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

async def main():
    # Choose TTS method: "pyttsx3", "gtts", or "edge_tts"
    tts_method = "pyttsx3"  # Change this to switch TTS methods
    
    assistant = FreeTTSVoiceAssistant(tts_method=tts_method)
    try:
        await assistant.start_streaming()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        assistant.connection_active = False
    except Exception as e:
        print(f"❌ Error: {e}")
        assistant.connection_active = False

if __name__ == "__main__":
    asyncio.run(main())


    
    # async def listen_loop():
    # while True:
    #     user_audio = await record_audio()  # Waits for user to speak
    #     transcript = await transcribe(user_audio)
        
    #     if tts_task and not tts_task.done():
    #         tts_task.cancel()  # Stop current TTS if speaking
        
    #     llm_response = await call_llm(transcript)
    #     tts_task = asyncio.create_task(speak_text_async(llm_response))




# noise filtering as in case if the user in in chaotic environment.

#  Practical Tips
# Use WebRTC VAD, webrtcvad, or pyaudio + RMS threshold to detect speech.

# Implement a light buffering system to:

# Store ~1 second of audio pre-roll

# Send it to Deepgram on reconnect to avoid missed words

# tell me more about it and show the solution

# import collections
# import webrtcvad
# import pyaudio
# import asyncio
# import time
# import numpy as np

# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# FRAME_DURATION = 30  # ms
# FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # samples per frame
# BUFFER_DURATION = 1.0  # seconds
# MAX_FRAMES = int(BUFFER_DURATION * 1000 / FRAME_DURATION)

# vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3
# p = pyaudio.PyAudio()

# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=FRAME_SIZE)

# # Ring buffer to hold pre-roll
# buffer = collections.deque(maxlen=MAX_FRAMES)


# def read_frame():
#     data = stream.read(FRAME_SIZE, exception_on_overflow=False)
#     return data


# def is_speech(frame):
#     return vad.is_speech(frame, RATE)


# async def wait_for_speech_start():
#     print("🎧 Waiting for speech...")
#     while True:
#         frame = read_frame()
#         buffer.append(frame)
#         if is_speech(frame):
#             print("🗣️ Speech detected!")
#             return list(buffer)  # Return pre-roll

# async def stream_to_deepgram(pre_roll_frames):
#     # Simulated streaming logic — replace with real Deepgram code
#     print(f"🚀 Streaming {len(pre_roll_frames)} pre-roll frames to Deepgram...")
#     for frame in pre_roll_frames:
#         await asyncio.sleep(FRAME_DURATION / 1000)  # simulate real-time sending
#         # await websocket.send(frame)
#     print("✅ Done streaming pre-roll. Now sending live frames...")

#     # Now keep streaming live audio
#     while True:
#         frame = read_frame()
#         if not is_speech(frame):
#             print("🛑 Speech ended.")
#             break
#         await asyncio.sleep(FRAME_DURATION / 1000)
#         # await websocket.send(frame)


# async def main_loop():
#     while True:
#         pre_roll = await wait_for_speech_start()
#         await stream_to_deepgram(pre_roll)

# try:
#     asyncio.run(main_loop())
# except KeyboardInterrupt:
#     print("Exiting...")
# finally:
#     stream.stop_stream()
#     stream.close()
#     p.terminate()



# stack up the prompts in case if user fucks up...like just the relevant ones
