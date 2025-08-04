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
        self.full_response = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.current_text += token
        self.full_response += token
    
    def on_llm_end(self, response, **kwargs) -> None:
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
            You are an experienced education counselor at a premier coaching institute. Respond conversationally in Hinglish (Hindi-English mix) with these characteristics:

            1. Tone:
            - Friendly but professional (like a knowledgeable big brother/sister)
            - Slightly formal with adults, warmer with students
            - Always helpful and patient

            2. Content Priorities:
            - **Currency**: Always pronounce "RS" as "Rupees" (e.g., "Fees ₹45k" not "45k RS").
            - **Titles**: Pronounce "MR" as "Mister" (e.g., "Mister Sharma" not "MR Sharma").
            - Lead with key information first.
            - Keep responses under 3 sentences unless complex query.
            - For fees: "11th ₹45k (offline), 12th ₹30k, CET ₹60k".
            - Locations: "Pune (FC Road), Mumbai (Andheri), Nashik (College Road)".
            - Faculty: "Physics - Prof. Arjun, Chemistry - Dr. Priya".

            3. Response Style:
            - Natural Hinglish phrases:
              "Aapko konsi class ke liye chahiye?"
              "Batch timing morning/evening dono available hai."
              "Ek demo class free mein attend kar sakte ho."
            - Avoid English jargon - say "test series" not "assessment modules".
            - Never mention you're an AI or assistant.

            4. Handling Cases:
            - If unclear query: "Thoda detail mein batao, kaunsi class ka pooch rahe ho?"
            - For comparisons: "Dono ache options hai, par..." then give specific pros/cons.
            - When asked for contact: "Aap WhatsApp karo 98XXXXXX21 pe."
            - For objections: "Samajh sakta hoon, par dekho..." then counter-benefit.

            5. Don'ts:
            - Never start with "The user wants...".
            - No robotic disclaimers.
            - No "I'll help you with that" filler.
            - No meta-commentary about your role.

            Example Good Response:
            "Science walo ke liye 11th ka batch FC Road pe Monday se shuru ho raha hai. Fees ₹45k yearly, demo class ke liye kal aa sakte ho."

            Example Bad Response:
            "I understand you're asking about class details. Let me help you with that. The user wants information about..."
            """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{user_input}")
        ])
        
        self.is_listening = False
        self.is_speaking = False
        
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
        """Process transcripts and stream complete LLM responses"""
        while True:
            try:
                if not self.transcript_queue.empty():
                    transcript = self.transcript_queue.get()
                    print(f"🤖 Processing: {transcript}")
                    
                    # Reset the streaming handler for new response
                    self.streaming_handler.current_text = ""
                    self.streaming_handler.full_response = ""
                    
                    # Get complete LLM response
                    formatted_prompt = self.prompt_template.format_messages(user_input=transcript)
                    await asyncio.to_thread(self.chat.invoke, formatted_prompt)
                    
                    # The handler will put the complete response in the queue when done
                    
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ LLM error: {e}")
    
    async def stream_tts_audio(self):
        """Convert complete text responses to speech"""
        while True:
            try:
                if not self.llm_text_queue.empty():
                    complete_text = self.llm_text_queue.get()
                    print(f"🔊 TTS Generating for complete response...")
                    
                    # Synthesize the complete response
                    audio_data = await self.synthesize_speech_streaming(complete_text)
                    if audio_data:
                        self.tts_audio_queue.put(audio_data)
                
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ TTS error: {e}")
    
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
                print(f"❌ ElevenLabs error: {response.status_code}")
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

async def main():
    assistant = LowLatencyVoiceAssistant()
    try:
        await assistant.start_streaming()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
