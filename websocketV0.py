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
import traceback


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
        self.full_response = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.current_text += token
        self.full_response += token
        print(f"LLM Token: {token}", end='', flush=True)  # Print tokens as they come
    
    def on_llm_end(self, response, **kwargs) -> None:
        print("\nLLM Response Complete")  # New line when response is complete
        if self.full_response.strip():
            self.text_queue.put(self.full_response.strip())
        self.current_text = ""
        self.full_response = ""
        
class LowLatencyVoiceAssistant:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.llm_text_queue = queue.Queue()
        self.tts_audio_queue = queue.Queue()
        
        # Initialize LLM with streaming
        self.streaming_handler = StreamingCallbackHandler(self.llm_text_queue)
        self.chat = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            streaming=True,
            callbacks=[self.streaming_handler]
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
        
    async def stream_deepgram_transcription(self):
        """Stream audio to Deepgram for real-time transcription"""
        uri = "wss://api.deepgram.com/v1/listen"
        
        extra_headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}"
        }
                
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
                
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_uri = f"{uri}?{param_string}"
        
        while self.connection_active:
            try:
                async with websockets.connect(
                    full_uri, 
                    extra_headers=extra_headers,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10 # whenever user interrupts
                ) as websocket:
                    print("🔗 Connected to Deepgram")
                    
                    async def send_audio():
                        try:
                            while self.connection_active:
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
                        except Exception as e:
                            print(f"⚠ Transcript receive error: {e}")
                    
                    # Send keep-alive message
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
        """Convert complete text responses to speech"""
        while True:
            try:
                if not self.llm_text_queue.empty():
                    complete_text = self.llm_text_queue.get()
                    print(f"\n🔊 TTS Generating for: {complete_text}")
                    
                    audio_data = await self.synthesize_speech_streaming(complete_text)
                    if audio_data:
                        self.tts_audio_queue.put(audio_data)
                
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ TTS error: {e}")
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
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            if response.status_code == 200:
                audio_chunks = []
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        audio_chunks.append(chunk)
                return b''.join(audio_chunks)
            else:
                print(f"❌ ElevenLabs error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"❌ TTS synthesis error: {e}")
            return None
    
    def play_audio_stream(self):
        """Play audio chunks as they arrive"""
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

async def main():
    assistant = LowLatencyVoiceAssistant()
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
