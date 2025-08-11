import os
import asyncio
import json
import queue
import threading
import numpy as np
from dotenv import load_dotenv
import pygame
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import time
import logging
import collections
import webrtcvad
import aiohttp
import websockets
import base64
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Validate required environment variables
required_keys = [DEEPGRAM_API_KEY, GROQ_API_KEY, ELEVENLABS_API_KEY]
if not all(required_keys):
    print("❌ Missing required environment variables")
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

# Initialize pygame mixer for TTS playback
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.init()

# FastAPI app setup
app = FastAPI(title="Exotel Voice Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket, assistant_ref):
        self.websocket = websocket
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
            logging.info(f"✅ LLM response complete: {self.full_response[:50]}...")
            # Queue the complete response for TTS
            asyncio.create_task(
                self.assistant_ref.handle_tts_response(self.full_response.strip())
            )
        else:
            logging.info("LLM response cancelled due to interruption")
        self.current_text = ""
        self.full_response = ""

class ExotelVoiceAssistant:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.audio_queue = asyncio.Queue()
        self.transcript_queue = asyncio.Queue()
        
        # Context management
        self.context_stack = collections.deque(maxlen=3)
        
        # VAD setup
        self.vad = webrtcvad.Vad(2)
        self.silence_threshold = 20
        self.is_speech_detected = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.consecutive_speech_threshold = 3
        
        # Control flags
        self.is_listening = True
        self.is_speaking = False
        self.user_interrupted = False
        self.connection_active = True
        self.deepgram_ws = None
        
        # Initialize LLM with streaming
        self.streaming_handler = StreamingCallbackHandler(websocket, self)
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

    async def initialize_deepgram_connection(self):
        """Initialize WebSocket connection to Deepgram"""
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
        
        try:
            self.deepgram_ws = await websockets.connect(
                full_uri, 
                extra_headers=extra_headers,
                ping_interval=20, 
                ping_timeout=10
            )
            logging.info("✅ Connected to Deepgram")
            
            # Start listening for transcripts
            asyncio.create_task(self.listen_deepgram_transcripts())
            
        except Exception as e:
            logging.error(f"❌ Deepgram connection error: {e}")
            raise

    async def listen_deepgram_transcripts(self):
        """Listen for transcripts from Deepgram"""
        try:
            async for message in self.deepgram_ws:
                if not self.connection_active:
                    break
                
                data = json.loads(message)
                if data.get('type') == 'Results':
                    # Handle transcript
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
                        
                        if transcript and len(transcript) > 2:
                            if is_final:
                                logging.info(f"🗣 Final: {transcript}")
                                await self.handle_transcript(transcript)
                            else:
                                logging.info(f"💬 Interim: {transcript}")
                                
        except Exception as e:
            logging.error(f"❌ Deepgram transcript error: {e}")

    async def process_audio_data(self, audio_data: bytes):
        """Process incoming audio data from Exotel"""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Process with VAD
            is_speech = self.process_audio_frame(audio_array)
            
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                
                if not self.is_speech_detected and self.speech_frames > self.consecutive_speech_threshold:
                    self.is_speech_detected = True
                    logging.info("🎤 Speech detected")
                    
                    # Handle interruption if assistant is speaking
                    if self.is_speaking:
                        await self.handle_interruption()
                
                # Send audio to Deepgram
                if self.deepgram_ws and not self.deepgram_ws.closed:
                    await self.deepgram_ws.send(audio_data)
                    
            elif self.is_speech_detected:
                self.silence_frames += 1
                if self.silence_frames > self.silence_threshold:
                    self.is_speech_detected = False
                    self.speech_frames = 0
                    logging.info("🤫 Speech ended")
                    
        except Exception as e:
            logging.error(f"❌ Audio processing error: {e}")

    def process_audio_frame(self, audio_data):
        """Process audio frame with WebRTC VAD"""
        try:
            # Ensure correct frame size for VAD
            frame_bytes = audio_data.tobytes()
            if len(frame_bytes) < 2 * FRAME_SIZE:
                return False
            
            # Use only the required frame size
            vad_frame = frame_bytes[:2 * FRAME_SIZE]
            return self.vad.is_speech(vad_frame, SAMPLE_RATE)
        except Exception:
            return False

    async def handle_interruption(self):
        """Handle user interruption"""
        if not self.is_speaking:
            return
        
        logging.info("🛑 User interrupted - stopping TTS")
        self.user_interrupted = True
        self.is_speaking = False
        
        # Stop pygame mixer
        try:
            pygame.mixer.music.stop()
            pygame.mixer.stop()
        except Exception as e:
            logging.error(f"❌ TTS stop error: {e}")

    async def handle_transcript(self, transcript: str):
        """Handle final transcript from Deepgram"""
        if self.user_interrupted:
            self.user_interrupted = False  # Reset for new input
        
        logging.info(f"🤖 Processing transcript: {transcript}")
        
        try:
            context = self.build_conversation_context()
            formatted_prompt = self.prompt_template.format_messages(
                conversation_context=context,
                user_input=transcript
            )
            
            # Stream LLM response (will trigger callback)
            await self.chat.ainvoke(formatted_prompt)
            
            # Add to context if successful
            if not self.user_interrupted and self.streaming_handler.full_response.strip():
                self.context_stack.append((transcript, self.streaming_handler.full_response.strip()))
                logging.info(f"💾 Context updated: {len(self.context_stack)} messages")
                
        except Exception as e:
            logging.error(f"❌ LLM processing error: {e}")

    def build_conversation_context(self):
        """Build conversation context"""
        if not self.context_stack:
            return ""
        
        context_parts = []
        for user_msg, assistant_msg in self.context_stack:
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {assistant_msg}")
        
        return "\n".join(context_parts)

    async def handle_tts_response(self, text: str):
        """Generate and send TTS audio back to Exotel"""
        if self.user_interrupted:
            logging.info("🚫 TTS skipped due to interruption")
            return
        
        logging.info(f"🔊 Generating TTS: {text[:50]}...")
        
        try:
            self.is_speaking = True
            
            # Generate TTS audio
            audio_data = await self.synthesize_speech(text)
            
            if audio_data and not self.user_interrupted:
                # Send audio back to Exotel via WebSocket
                await self.send_audio_to_exotel(audio_data)
                logging.info("✅ TTS sent to Exotel")
            
        except Exception as e:
            logging.error(f"❌ TTS error: {e}")
        finally:
            self.is_speaking = False

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Generate speech using ElevenLabs"""
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
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        return audio_data
                    else:
                        error_text = await response.text()
                        logging.error(f"❌ ElevenLabs error: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logging.error(f"❌ TTS synthesis error: {e}")
            return None

    async def send_audio_to_exotel(self, audio_data: bytes):
        """Send audio data back to Exotel via WebSocket"""
        try:
            # Encode audio as base64 for WebSocket transmission
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            message = {
                "type": "audio_response",
                "data": audio_b64,
                "format": "mp3",  # or wav depending on ElevenLabs output
                "sample_rate": 22050
            }
            
            await self.websocket.send_text(json.dumps(message))
            
        except Exception as e:
            logging.error(f"❌ Error sending audio to Exotel: {e}")

    async def cleanup(self):
        """Clean up resources"""
        self.connection_active = False
        
        if self.deepgram_ws and not self.deepgram_ws.closed:
            await self.deepgram_ws.close()
            
        pygame.mixer.quit()
        logging.info("✅ Assistant cleaned up")

# Global storage for active connections
active_connections: Dict[str, ExotelVoiceAssistant] = {}

@app.get("/")
async def root():
    return {"message": "Exotel Voice Assistant API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_connections": len(active_connections)}

@app.websocket("/ws/voice/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    """Main WebSocket endpoint for Exotel integration"""
    await websocket.accept()
    logging.info(f"🔗 New connection from Exotel: {call_id}")
    
    # Create assistant instance for this call
    assistant = ExotelVoiceAssistant(websocket)
    active_connections[call_id] = assistant
    
    try:
        # Initialize Deepgram connection
        await assistant.initialize_deepgram_connection()
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": "Voice Assistant connected - ready to receive audio"
        }))
        
        # Listen for messages from Exotel
        while assistant.connection_active:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "audio_data":
                    # Decode base64 audio and process
                    audio_data = base64.b64decode(message["data"])
                    await assistant.process_audio_data(audio_data)
                    
                elif message["type"] == "call_ended":
                    logging.info(f"📞 Call ended: {call_id}")
                    break
                    
                elif message["type"] == "ping":
                    # Respond to ping
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError:
                logging.error("❌ Invalid JSON received")
                continue
                
    except WebSocketDisconnect:
        logging.info(f"🔌 WebSocket disconnected: {call_id}")
    except Exception as e:
        logging.error(f"❌ WebSocket error for {call_id}: {e}")
    finally:
        # Cleanup
        if call_id in active_connections:
            await active_connections[call_id].cleanup()
            del active_connections[call_id]
        
        logging.info(f"✅ Connection {call_id} cleaned up")

@app.post("/webhook/exotel")
async def exotel_webhook(request_data: dict):
    """Webhook endpoint for Exotel call events"""
    logging.info(f"📲 Exotel webhook: {request_data}")
    
    # Handle different Exotel events
    event_type = request_data.get("EventType")
    call_sid = request_data.get("CallSid")
    
    if event_type == "call-initiated":
        logging.info(f"📞 Call initiated: {call_sid}")
    elif event_type == "call-answered":
        logging.info(f"✅ Call answered: {call_sid}")
    elif event_type == "call-completed":
        logging.info(f"📞 Call completed: {call_sid}")
        # Clean up if connection still exists
        if call_sid in active_connections:
            await active_connections[call_sid].cleanup()
            del active_connections[call_sid]
    
    return {"status": "success"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",  # Change this to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )