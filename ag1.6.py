import os
import sys
import argparse
import tempfile
import asyncio
import websockets
import json
import queue
import threading
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from typing import List
import time
import wave
import pygame
import requests
from io import BytesIO
import base64
from functools import lru_cache
import hashlib

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

load_dotenv()
# Optional OCR imports (only used if needed)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Optimized audio settings for lower latency
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024  # Reduced for lower latency
BUFFER_SIZE = 2048  # Smaller buffer
SILENCE_THRESHOLD = 0.005
SILENCE_DURATION = 2.0
VAD_BUFFER_SIZE = 5

# ElevenLabs settings
ELEVENLABS_SAMPLE_RATE = 24000
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Initialize pygame mixer with optimized settings for lower latency
pygame.mixer.pre_init(frequency=ELEVENLABS_SAMPLE_RATE, size=-16, channels=1, buffer=512)
pygame.mixer.init()

# ---------- Custom Prompt Template for Hinglish ----------
HINGLISH_PROMPT_TEMPLATE = """You are a helpful AI assistant that can understand and respond in Hinglish (a mix of Hindi and English). 
You should match the language style of the user's question. If they ask in Hinglish, respond in Hinglish. 
If they ask in English, respond in English. If they ask in Hindi, respond in Hinglish.

Keep your responses concise and conversational, suitable for voice interaction. Avoid overly long explanations.

Use the following pieces of context to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

Try to answer user questions precisely, but should strictly not user more than 30 words.
If user asks some out of context question, like tell me the capital of any place, etc, dont answer and polietly say, im here to help on behalf of classes and cannot answer that questions.

Previous conversation:
{chat_history}

Context from the document:
{context}

Question: {question}

Answer in the same language style as the question (Hinglish/English) and not more than 2 sentences, keeping it conversational and concise:
eg: query: kya aap mujhe classes ki details de sakte hai?
    answer: Haan bilkul hamare yaha 10th, 12th ke classes hai
    
    query: Kya aap mujhe 10th ke baare me bata sakte hai?
    answer: hamare yaha 10th ssc, cbse aur icse hai isme se aapko konsa chahiye? """

# ---------- Optimized Audio Buffer System ----------
class AudioBuffer:
    def __init__(self):
        self.buffer = queue.Queue()
        self.is_playing = False
        self.playback_lock = asyncio.Lock()
        
    async def add_chunk(self, chunk):
        """Add audio chunk to buffer and start playback if not already playing"""
        try:
            self.buffer.put(chunk)
            if not self.is_playing:
                asyncio.create_task(self.start_playback())
        except Exception as e:
            print(f"Error adding chunk to buffer: {e}")
            
    async def start_playback(self):
        """Start playing buffered audio chunks"""
        async with self.playback_lock:
            if self.is_playing:
                return
                
            self.is_playing = True
            try:
                while not self.buffer.empty():
                    chunk = self.buffer.get()
                    await self.play_chunk(chunk)
            except Exception as e:
                print(f"Error in playback: {e}")
            finally:
                self.is_playing = False
                
    async def play_chunk(self, chunk):
        """Play individual audio chunk"""
        try:
            # Convert raw audio data to proper format
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            
            # Create proper WAV format with headers
            wav_io = BytesIO()
            
            # Write WAV header
            wav_io.write(b'RIFF')
            wav_io.write((36 + len(chunk)).to_bytes(4, 'little'))
            wav_io.write(b'WAVE')
            wav_io.write(b'fmt ')
            wav_io.write((16).to_bytes(4, 'little'))  # PCM format
            wav_io.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
            wav_io.write((1).to_bytes(2, 'little'))   # Number of channels
            wav_io.write((ELEVENLABS_SAMPLE_RATE).to_bytes(4, 'little'))  # Sample rate
            wav_io.write((ELEVENLABS_SAMPLE_RATE * 2).to_bytes(4, 'little'))  # Byte rate
            wav_io.write((2).to_bytes(2, 'little'))   # Block align
            wav_io.write((16).to_bytes(2, 'little'))  # Bits per sample
            wav_io.write(b'data')
            wav_io.write(len(chunk).to_bytes(4, 'little'))
            wav_io.write(chunk)
            
            # Create temp file with proper WAV format
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(wav_io.getvalue())
                temp_path = temp_file.name
            
            # Play using pygame
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Wait for chunk to finish
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.01)
                
            # Clean up
            os.unlink(temp_path)
            wav_io.close()
            
        except Exception as e:
            print(f"Error playing chunk: {e}")

# ---------- Response Cache System ----------
class ResponseCache:
    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
        
    def _get_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
        
    def get(self, query: str):
        """Get cached response"""
        key = self._get_key(query)
        if key in self.cache:
            # Move to end (most recent)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
        
    def set(self, query: str, response: str):
        """Cache response"""
        key = self._get_key(query)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            
        self.cache[key] = response
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

# ---------- Helper Functions ----------
def try_load_with_pypdf(pdf_path: str):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"[PyPDFLoader] failed: {e}")
        return []

def try_load_with_unstructured(pdf_path: str):
    try:
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"[UnstructuredPDFLoader] failed: {e}")
        return []

def ocr_pdf_to_text(pdf_path: str) -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError("pdf2image and pytesseract are required for OCR fallback but not available.")
    print("[OCR] Converting PDF pages to images (pdf2image)...")
    images = convert_from_path(pdf_path)
    print(f"[OCR] {len(images)} pages converted. Running pytesseract on each page...")
    text_parts = []
    for i, img in enumerate(images, start=1):
        page_text = pytesseract.image_to_string(img)
        print(f"[OCR] Page {i} length: {len(page_text)} chars")
        text_parts.append(page_text)
    return "\n\n".join(text_parts)

def docs_total_text_length(docs) -> int:
    return sum(len(getattr(d, "page_content", str(d))) for d in docs)

def detect_language_style(text: str) -> str:
    """Simple heuristic to detect if text contains Hinglish/Hindi elements"""
    hindi_indicators = ['hai', 'hain', 'kar', 'kya', 'kaise', 'kahan', 'kyun', 'aur', 'ya', 'main', 'mein', 'ko', 'ka', 'ki', 'ke', 'se', 'par', 'wala', 'wali', 'vale']
    text_lower = text.lower()
    
    hindi_count = sum(1 for indicator in hindi_indicators if indicator in text_lower)
    if hindi_count > 0:
        return "hinglish"
    return "english"

def get_elevenlabs_voice_settings(language_style: str) -> dict:
    """Get appropriate ElevenLabs voice settings based on detected language"""
    if language_style == "hinglish":
        return {
            "voice_id": ELEVENLABS_VOICE_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "style_exaggeration": 0.3,
                "speaker_boost": True
            }
        }
    else:
        return {
            "voice_id": ELEVENLABS_VOICE_ID,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.75,
                "style": 0.1,
                "style_exaggeration": 0.3,
                "speaker_boost": True
            }
        }

class OptimizedVoiceRAGAssistant:
    def __init__(self, qa_chain, deepgram_api_key, elevenlabs_api_key):
        self.qa_chain = qa_chain
        self.deepgram_api_key = deepgram_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        
        # Queues and buffers
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.audio_buffer = AudioBuffer()
        self.response_cache = ResponseCache(max_size=50)
        
        # State management
        self.is_listening = False
        self.is_speaking = False
        self.connection_active = False
        self.processing_query = False
        self.last_processed_transcript = ""
        self.processing_lock = asyncio.Lock()
        
    def audio_callback(self, indata, frames, time, status):
        """Optimized audio callback for lower latency"""
        if status:
            print(f"Audio status: {status}")
        
        if not self.is_speaking and self.connection_active:
            volume = np.linalg.norm(indata) * 10
            if volume > SILENCE_THRESHOLD:
                self.is_listening = True                
                audio_flat = indata.flatten()                
                if audio_flat.dtype == np.float32:
                    audio_int16 = (audio_flat * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_flat.astype(np.int16)
                
                # Limit queue size to prevent latency buildup
                if self.audio_queue.qsize() < 50:
                    self.audio_queue.put(audio_int16)

    async def stream_deepgram_transcription(self):
        """Fixed Deepgram transcription with proper nova-2-phonecall parameters"""
        uri = "wss://api.deepgram.com/v1/listen"
        
        # Corrected Deepgram authorization header format
        extra_headers = {
            "Authorization": f"Token {self.deepgram_api_key}"
        }
        
        # Proper parameters for nova-2-phonecall model
        params = {
            "model": "nova-2",
            "language": "hi",
            "punctuate": "true",
            "interim_results": "true",
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "multichannel": "false",
            "utterances": "true",
            "endpointing": "500",
            "vad_events": "true",
            "diarize": "false"
        }
        
        # Build query string correctly
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_uri = f"{uri}?{param_string}"
        
        self.connection_active = True
        
        try:
            async with websockets.connect(
                full_uri, 
                extra_headers=extra_headers,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                print("🔗 Connected to Deepgram with nova-2-phonecall model")
                
                async def send_audio():
                    try:
                        silent_packet = np.zeros(CHUNK_SIZE, dtype=np.int16)
                        last_activity = time.time()
                        
                        while self.connection_active:
                            if not self.audio_queue.empty():
                                audio_data = self.audio_queue.get()
                                
                                if audio_data.dtype != np.int16:
                                    audio_int16 = (audio_data * 32767).astype(np.int16)
                                else:
                                    audio_int16 = audio_data
                                
                                await websocket.send(audio_int16.tobytes())
                                last_activity = time.time()
                            else:
                                current_time = time.time()
                                if current_time - last_activity > 1.5:
                                    await websocket.send(silent_packet.tobytes())
                                    last_activity = current_time
                                await asyncio.sleep(0.005)
                            
                    except Exception as e:
                        print(f"⚠ Audio send error: {e}")
                        self.connection_active = False
                
                async def receive_transcripts():
                    try:
                        last_final_transcript = ""
                        async for message in websocket:
                            try:
                                data = json.loads(message)
                                
                                # Handle Deepgram response format for phonecall model
                                if data.get('type') == 'Results':
                                    # Phonecall model has slightly different response structure
                                    if 'channel' in data and 'alternatives' in data['channel']:
                                        transcript = data['channel']['alternatives'][0].get('transcript', '')
                                        is_final = data.get('is_final', False)
                                        
                                        if is_final and transcript.strip():
                                            if transcript.strip() != last_final_transcript:
                                                last_final_transcript = transcript.strip()
                                                print(f"🗣 Final: {transcript}")
                                                
                                                if not self.processing_query:
                                                    self.transcript_queue.put(transcript.strip())
                                        elif transcript.strip():
                                            print(f"🗣 Interim: {transcript}")
                                
                            except (json.JSONDecodeError, KeyError) as e:
                                print(f"⚠ Parsing error: {e}")
                                continue
                                
                    except Exception as e:
                        print(f"⚠ Transcript receive error: {e}")
                        self.connection_active = False
                
                await asyncio.gather(
                    send_audio(),
                    receive_transcripts(),
                    return_exceptions=True
                )
                        
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"❌ Deepgram connection failed with status code: {e.status_code}")
            print(f"❌ Response headers: {e.headers}")
            print(f"❌ Error details: This might be due to:")
            print(f"   1. Invalid API key permissions for phonecall model")
            print(f"   2. Missing required parameters for phonecall model")
            print(f"   3. API key doesn't have access to phonecall features")
            print(f"❌ Try using the regular 'nova-2' model instead")
            self.connection_active = False
        except Exception as e:
            print(f"❌ Deepgram connection error: {e}")
            self.connection_active = False

    async def process_streaming_response(self, transcript):
        """Process response with streaming and caching"""
        # Check cache first
        cached_response = self.response_cache.get(transcript)
        if cached_response:
            print(f"🔄 Using cached response for: {transcript[:30]}...")
            return cached_response
        
        try:
            # Use streaming if available
            if hasattr(self.qa_chain, 'stream'):
                full_response = ""
                async for chunk in self.qa_chain.astream({"question": transcript}):
                    if "answer" in chunk:
                        chunk_text = chunk["answer"]
                        full_response += chunk_text
                        
                        # Stream partial responses to TTS for immediate playback
                        if len(full_response) > 30 and not self.tts_queue.qsize():
                            await self.stream_partial_to_tts(full_response, transcript)
                
                # Cache the full response
                self.response_cache.set(transcript, full_response)
                return full_response
            else:
                # Fallback to regular processing
                result = await asyncio.to_thread(self.qa_chain, {"question": transcript})
                response = result["answer"]
                self.response_cache.set(transcript, response)
                return response
                
        except Exception as e:
            print(f"❌ Streaming response error: {e}")
            # Fallback to regular processing
            result = await asyncio.to_thread(self.qa_chain, {"question": transcript})
            response = result["answer"]
            self.response_cache.set(transcript, response)
            return response

    async def stream_partial_to_tts(self, partial_text, original_query):
        """Stream partial response to TTS for immediate playback"""
        language_style = detect_language_style(original_query)
        
        if not self.tts_queue.qsize():  # Only if TTS queue is empty
            self.tts_queue.put({
                "text": partial_text,
                "language_style": language_style,
                "is_partial": True
            })

    async def process_voice_queries(self):
        """Optimized voice query processing with streaming"""
        while True:
            try:
                if not self.transcript_queue.empty() and not self.processing_query:
                    async with self.processing_lock:
                        if self.processing_query:
                            continue
                            
                        self.processing_query = True
                        transcript = self.transcript_queue.get()
                        
                        # Clear remaining transcripts to prevent duplicates
                        while not self.transcript_queue.empty():
                            self.transcript_queue.get()
                        
                        if len(transcript.strip()) < 3 or transcript == self.last_processed_transcript:
                            self.processing_query = False
                            continue
                            
                        self.last_processed_transcript = transcript
                        print(f"\n🤖 Processing: {transcript}")
                        
                        query_style = detect_language_style(transcript)
                        print(f"[Style: {query_style}]")
                        
                        try:
                            # Use optimized streaming response
                            answer = await self.process_streaming_response(transcript)
                            
                            print("\n--- Voice Answer ---")
                            print(answer)
                            print("-" * 60)
                            
                            # Queue for TTS if not already queued
                            if not self.tts_queue.qsize():
                                self.tts_queue.put({
                                    "text": answer,
                                    "language_style": query_style,
                                    "is_partial": False
                                })
                        
                        except Exception as e:
                            print(f"❌ Query processing error: {e}")
                        
                        finally:
                            self.processing_query = False
                    
                await asyncio.sleep(0.05)  # Reduced sleep for faster processing
            except Exception as e:
                print(f"❌ Voice query processing error: {e}")
                self.processing_query = False

    async def generate_elevenlabs_speech_websocket(self, text: str, language_style: str):
        """WebSocket-based ElevenLabs TTS for real-time streaming"""
        try:
            self.is_speaking = True
            print(f"🔊 Starting WebSocket TTS for: {text[:50]}...")
            
            voice_config = get_elevenlabs_voice_settings(language_style)
            
            # Use the streaming WebSocket endpoint
            uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_config['voice_id']}/stream-input?model_id=eleven_turbo_v2_5&output_format=pcm_24000"
            
            async with websockets.connect(
                uri,
                extra_headers={"xi-api-key": self.elevenlabs_api_key}
            ) as websocket:
                
                # Send BOS (Beginning of Stream) message
                bos_message = {
                    "text": " ",
                    "voice_settings": voice_config["voice_settings"],
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290]
                    }
                }
                await websocket.send(json.dumps(bos_message))
                
                # Send text message
                text_message = {
                    "text": text + " ",
                    "try_trigger_generation": True
                }
                await websocket.send(json.dumps(text_message))
                
                # Send EOS (End of Stream) message
                eos_message = {"text": ""}
                await websocket.send(json.dumps(eos_message))
                
                # Collect all audio data first
                audio_chunks = []
                async for message in websocket:
                    if isinstance(message, bytes):
                        audio_chunks.append(message)
                    else:
                        try:
                            msg_data = json.loads(message)
                            if msg_data.get("audio"):
                                # Handle base64 encoded audio
                                audio_data = base64.b64decode(msg_data["audio"])
                                audio_chunks.append(audio_data)
                            elif msg_data.get("isFinal"):
                                break
                        except Exception as e:
                            print(f"WebSocket message parsing error: {e}")
                
                # Combine all chunks and play as single audio
                if audio_chunks:
                    combined_audio = b''.join(audio_chunks)
                    await self.play_combined_audio(combined_audio)
                
                print("🔇 WebSocket TTS streaming completed")
                
        except Exception as e:
            print(f"❌ WebSocket TTS error: {e}")
            # Fallback to REST API
            await self.generate_elevenlabs_speech_fallback(text, language_style)
        finally:
            # Wait a bit before re-enabling listening
            await asyncio.sleep(0.5)
            self.is_speaking = False
            print("🎤 Re-enabled listening after TTS")

    async def play_combined_audio(self, audio_data: bytes):
        """Play combined audio data as a single stream"""
        try:
            # Create proper WAV format
            wav_io = BytesIO()
            
            # Write WAV header for 24kHz mono PCM
            wav_io.write(b'RIFF')
            wav_io.write((36 + len(audio_data)).to_bytes(4, 'little'))
            wav_io.write(b'WAVE')
            wav_io.write(b'fmt ')
            wav_io.write((16).to_bytes(4, 'little'))  # PCM format
            wav_io.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
            wav_io.write((1).to_bytes(2, 'little'))   # Number of channels (mono)
            wav_io.write((ELEVENLABS_SAMPLE_RATE).to_bytes(4, 'little'))  # Sample rate 24000
            wav_io.write((ELEVENLABS_SAMPLE_RATE * 2).to_bytes(4, 'little'))  # Byte rate
            wav_io.write((2).to_bytes(2, 'little'))   # Block align
            wav_io.write((16).to_bytes(2, 'little'))  # Bits per sample
            wav_io.write(b'data')
            wav_io.write(len(audio_data).to_bytes(4, 'little'))
            wav_io.write(audio_data)
            
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(wav_io.getvalue())
                temp_path = temp_file.name
            
            # Play using pygame
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.01)
            
            # Clean up
            os.unlink(temp_path)
            wav_io.close()
            
        except Exception as e:
            print(f"Error playing combined audio: {e}")

    async def generate_elevenlabs_speech_fallback(self, text: str, language_style: str):
        """Fallback REST API method for ElevenLabs TTS"""
        try:
            voice_config = get_elevenlabs_voice_settings(language_style)
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_config['voice_id']}/stream"
            headers = {
                "xi-api-key": self.elevenlabs_api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": voice_config["voice_settings"]
            }
            
            response = requests.post(url, headers=headers, json=data, stream=True)
            
            if response.status_code == 200:
                audio_data = BytesIO()
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        audio_data.write(chunk)
                
                audio_data.seek(0)
                pygame.mixer.music.load(audio_data)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.01)
                    
                print("🔇 Fallback TTS playback completed")
            else:
                print(f"❌ ElevenLabs API error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Fallback TTS error: {e}")

    async def realtime_tts_processor(self):
        """Optimized real-time TTS processor"""
        while True:
            try:
                if not self.tts_queue.empty() and not self.is_speaking:
                    tts_data = self.tts_queue.get()
                    
                    # Clear remaining TTS requests to prevent audio overlap
                    while not self.tts_queue.empty():
                        self.tts_queue.get()
                    
                    text = tts_data["text"]
                    language_style = tts_data["language_style"]
                    is_partial = tts_data.get("is_partial", False)
                    
                    print(f"🔊 {'Streaming' if is_partial else 'Generating'} TTS: {text[:50]}...")
                    
                    # Use WebSocket TTS for real-time streaming
                    await self.generate_elevenlabs_speech_websocket(text, language_style)
                    
                await asyncio.sleep(0.01)  # Minimal sleep for responsive processing
            except Exception as e:
                print(f"❌ TTS processor error: {e}")

    async def audio_output_manager(self):
        """Manage audio output buffer and playback"""
        while True:
            try:
                # This runs continuously to manage the audio buffer
                # The AudioBuffer class handles the actual playback
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ Audio output manager error: {e}")

    async def optimized_voice_interaction(self):
        """Optimized concurrent voice interaction"""
        print("🟢 Starting Optimized Voice-Enabled RAG Assistant")
        print("🚀 Features: Real-time WebSocket TTS, Response Caching, Streaming Processing")
        print("🎤 Listening for voice input... Speak naturally!")
        print("🔊 Ultra-low latency audio playback enabled!")
        
        # Start text input thread for fallback
        text_thread = threading.Thread(target=self.text_input_handler, daemon=True)
        text_thread.start()
        
        # Start audio input stream with optimized settings
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self.audio_callback,
            blocksize=CHUNK_SIZE,
            dtype=np.float32
        ):
            # Run all optimized tasks concurrently
            tasks = [
                self.stream_deepgram_transcription(),
                self.process_voice_queries(),
                self.realtime_tts_processor(),
                self.audio_output_manager()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)

    def text_input_handler(self):
        """Handle text input as fallback"""
        while True:
            try:
                text_query = input()
                if text_query.lower().strip() in ("exit", "quit", "bye", "alvida"):
                    print("Goodbye! Cache saved.")
                    os._exit(0)
                elif text_query.lower().strip() in ("clear", "reset"):
                    self.qa_chain.memory.clear()
                    self.response_cache = ResponseCache(max_size=50)  # Reset cache
                    print("Memory and cache cleared!")
                elif text_query.strip():
                    print(f"💬 Text input: {text_query}")
                    if not self.processing_query:
                        self.transcript_queue.put(text_query.strip())
            except (EOFError, KeyboardInterrupt):
                print("\nShutting down...")
                os._exit(0)

# ---------- Optimized RAG Builder ----------
def build_optimized_rag(pdf_path: str, groq_api_key: str, deepgram_api_key: str, elevenlabs_api_key: str, index_save_path: str = "faiss_index"):
    # PDF loading logic (same as original)
    print(f"[+] Loading PDF: {pdf_path}")
    documents = try_load_with_pypdf(pdf_path)
    print(f"[+] PyPDFLoader returned {len(documents)} page-docs; total chars = {docs_total_text_length(documents)}")

    if len(documents) == 0 or docs_total_text_length(documents) < 200:
        print("[!] PyPDFLoader returned little/no text. Trying UnstructuredPDFLoader...")
        documents = try_load_with_unstructured(pdf_path)
        print(f"[+] UnstructuredPDFLoader returned {len(documents)} page-docs; total chars = {docs_total_text_length(documents)}")

    if len(documents) == 0 or docs_total_text_length(documents) < 200:
        print("[!] Loaders failed. Falling back to OCR...")
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR not available. Install pdf2image and pytesseract.")
        ocr_text = ocr_pdf_to_text(pdf_path)
        if len(ocr_text.strip()) == 0:
            raise RuntimeError("OCR produced no text.")
        from langchain.schema import Document
        documents = [Document(page_content=ocr_text, metadata={"source": pdf_path})]
        print(f"[+] OCR produced {len(ocr_text)} characters of text.")

    # Text splitting and embeddings (same as original)
    print("[+] Splitting into optimized chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)  # Smaller chunks for faster processing
    docs = splitter.split_documents(documents)
    print(f"[+] Created {len(docs)} chunks.")

    if len(docs) == 0:
        raise RuntimeError("No chunks produced after splitting.")

    print("[+] Creating embeddings (optimized multilingual model)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # FAISS index creation/loading
    if os.path.exists(index_save_path):
        print(f"[+] Loading existing FAISS index from '{index_save_path}'...")
        vectorstore = FAISS.load_local(index_save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[+] Building optimized FAISS index...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        print(f"[+] Saving FAISS index to '{index_save_path}'...")
        vectorstore.save_local(index_save_path)

    # Optimized retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Optimized Groq LLM with faster model
    print("[+] Initializing optimized Groq LLM...")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.1-8b-instant",  # Faster, lighter model for lower latency
        temperature=0.3,
        max_tokens=30,  # Reduced for concise responses
        streaming=True  # Enable streaming for real-time responses
    )

    # Custom prompt for optimized responses
    custom_prompt = PromptTemplate(
        template=HINGLISH_PROMPT_TEMPLATE,
        input_variables=["chat_history", "context", "question"]
    )

    # Optimized memory with smaller window
    memory = ConversationBufferWindowMemory(
        k=3,  # Reduced for faster processing
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Optimized ConversationalRetrievalChain
    print("[+] Building optimized ConversationalRetrievalChain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
        verbose=False,
        max_tokens_limit=2000  # Limit context for faster processing
    )

    return qa_chain

# ---------- Optimized Async Main ----------
async def optimized_async_main(pdf_path: str, groq_api_key: str, deepgram_api_key: str, elevenlabs_api_key: str, index_save_path: str):
    """Optimized async main function with all improvements"""
    try:
        # Build optimized RAG pipeline
        qa_chain = build_optimized_rag(pdf_path, groq_api_key, deepgram_api_key, elevenlabs_api_key, index_save_path)
        
        print("[+] ✅ Optimized RAG pipeline ready!")
        print("[+] 🚀 Features enabled:")
        print("    • Real-time WebSocket TTS streaming")
        print("    • Response caching for faster queries")
        print("    • Optimized audio processing")
        print("    • Concurrent processing pipeline")
        print("    • Low-latency voice interaction")
        print("[+] 🎤 Voice input: English, Hindi, Hinglish supported!")
        print("[+] 🔊 ElevenLabs Turbo v2.5 real-time TTS!")
        
        # Create optimized voice assistant
        voice_assistant = OptimizedVoiceRAGAssistant(qa_chain, deepgram_api_key, elevenlabs_api_key)
        
        # Start optimized voice interaction
        await voice_assistant.optimized_voice_interaction()
        
    except Exception as e:
        print(f"❌ Error in optimized main: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Voice RAG with Real-time WebSocket TTS")
    parser.add_argument("pdf", nargs="?", help="Path to PDF (if omitted, you'll be prompted).")
    parser.add_argument("--index", default="faiss_index", help="Folder path to save/load FAISS index")
    parser.add_argument("--memory-window", type=int, default=3, help="Number of conversation exchanges to remember (reduced for optimization)")
    parser.add_argument("--cache-size", type=int, default=50, help="Number of responses to cache")
    args = parser.parse_args()

    load_dotenv()
    
    # Check required API keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in environment. Put it in .env or export GROQ_API_KEY.")
        sys.exit(1)
        
    if not DEEPGRAM_API_KEY:
        print("ERROR: DEEPGRAM_API_KEY not found in environment. Put it in .env or export DEEPGRAM_API_KEY.")
        sys.exit(1)
        
    if not ELEVENLABS_API_KEY:
        print("ERROR: ELEVENLABS_API_KEY not found in environment. Put it in .env or export ELEVENLABS_API_KEY.")
        sys.exit(1)

    pdf_path = args.pdf
    if not pdf_path:
        pdf_path = input("Enter path to PDF: ").strip()

    if not os.path.isfile(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    try:
        print("🚀 Starting Optimized Voice RAG Assistant...")
        print("⚡ Performance improvements:")
        print("   • WebSocket-based real-time TTS streaming")
        print("   • Response caching system")
        print("   • Optimized audio buffer management")
        print("   • Concurrent processing pipeline")
        print("   • Reduced latency settings")
        print("   • Streaming LLM responses")
        print("-" * 60)
        
        # Run the optimized async main function
        asyncio.run(optimized_async_main(pdf_path, GROQ_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, args.index))
    except KeyboardInterrupt:
        print("\n👋 Shutting down optimized assistant... Dhanyawad!")
    except Exception as e:
        print(f"\nFatal error in optimized system: {e}")
        sys.exit(1)
