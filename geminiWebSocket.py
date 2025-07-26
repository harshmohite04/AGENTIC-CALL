import os
import queue
import threading
import sounddevice as sd
import numpy as np
import asyncio
import websockets
import json
import base64
import time

from dotenv import load_dotenv
# We only need these specific imports from deepgram
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# --- Configuration ---
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "zT03pEAEi0VHKciJODfn")
ELEVENLABS_MODEL_ID = "eleven_turbo_v2" 

# Audio settings
SAMPLERATE = 16000
CHANNELS = 1

# --- System Prompt ---
SYSTEM_PROMPT = """
You are a friendly and empathetic academic counselor for EduSpark Academy.
Your goal is to help students and parents by providing clear and helpful information.
Speak in a natural, conversational Hinglish tone. Be warm and reassuring.

Here is the information you have:
- Course Fees:
  - Class 11 (Offline): ₹45,000
  - Class 12: ₹30,000
  - MH-CET Crash Course: ₹60,000
- Center Locations:
  - Pune: FC Road
  - Mumbai: Andheri
  - Nashik: College Road
- Key Faculty:
  - Physics: Prof. Arjun Deshmukh
  - Chemistry: Dr. Priya Kulkarni
  - Maths: Mrs. Sneha Patil
  - Biology: Dr. Rohan Mehta

Keep your answers concise and directly address the user's questions.
Start the conversation by introducing yourself and asking how you can help.
"""

# --- Initialization ---
transcript_queue = queue.Queue()
llm_response_queue = queue.Queue()
audio_out_queue = queue.Queue()
assistant_is_speaking = threading.Event()

# Initialize APIs
dg_client = DeepgramClient(DEEPGRAM_API_KEY)
chat = ChatGroq(model="llama3-70b-8192", temperature=0.7, groq_api_key=GROQ_API_KEY)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_input}")
])
chain = prompt_template | chat

# --- Thread 1: Coordinated Audio Input and Transcription ---
def audio_input_and_transcription_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Using 'asynclive' is the most documented method, despite any warnings.
        dg_connection = dg_client.listen.asynclive.v("1")

        async def on_message(self, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if len(transcript) > 0 and result.is_final:
                print(f"🗣️ User said: {transcript}")
                transcript_queue.put(transcript)

        async def on_error(self, error, **kwargs):
            print(f"❌ Deepgram Error: {error}")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # FINAL CHANGE: This is the most explicit and complete set of options
        # required by the v4.x SDK to avoid a 400 Bad Request error.
        options = LiveOptions(
            model="nova-2",
            language="hi-IN",
            encoding="linear16",
            channels=CHANNELS,
            sample_rate=SAMPLERATE,
            smart_format=True,
            punctuate=True,
            endpointing=300, # Helps detect end of speech
            interim_results=False
        )

        def mic_callback(indata, frames, time, status):
            if not assistant_is_speaking.is_set():
                if dg_connection:
                    asyncio.run_coroutine_threadsafe(dg_connection.send(indata.tobytes()), loop)

        def start_microphone():
            print("🎤 Listening for your question...")
            with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype='int16', callback=mic_callback):
                while True:
                    time.sleep(1)

        mic_thread = threading.Thread(target=start_microphone, daemon=True)
        mic_thread.start()

        print("✅ Attempting to connect to Deepgram with final configuration...")
        # The 'start' method takes the LiveOptions object directly.
        loop.run_until_complete(dg_connection.start(options))

    except Exception as e:
        print(f"❌ An error occurred in the audio thread: {e}")

# --- Thread 2: Language Model Processing ---
def llm_thread_main():
    while True:
        try:
            transcript = transcript_queue.get(block=True)
            if not transcript.strip():
                continue

            print("🧠 Thinking...")
            assistant_is_speaking.set()
            llm_stream = chain.stream({"user_input": transcript})
            
            sentence_buffer = ""
            for chunk in llm_stream:
                content = chunk.content
                sentence_buffer += content
                if any(p in sentence_buffer for p in ".?!"):
                    llm_response_queue.put(sentence_buffer)
                    print(f"🤖 AI says: {sentence_buffer.strip()}")
                    sentence_buffer = ""
            if sentence_buffer:
                llm_response_queue.put(sentence_buffer)
                print(f"🤖 AI says: {sentence_buffer.strip()}")
            llm_response_queue.put(None)
            
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            assistant_is_speaking.clear()

# --- Thread 3: Text-to-Speech and Audio Playback ---
def tts_playback_thread_main():
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input?model_id={ELEVENLABS_MODEL_ID}"
    def playback_callback(outdata, frames, time, status):
        if status: print(f"Playback Error: {status}")
        try:
            data = audio_out_queue.get_nowait()
            outdata[:] = data.reshape(outdata.shape)
        except queue.Empty:
            outdata.fill(0)

    with sd.OutputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype='int16', callback=playback_callback):
        while True:
            first_chunk = llm_response_queue.get(block=True)
            if first_chunk is None: continue
            print("🔊 Synthesizing and playing audio...")
            async def tts_stream():
                async with websockets.connect(uri) as websocket:
                    await websocket.send(json.dumps({
                        "text": " ", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}, "xi_api_key": ELEVENLABS_API_KEY,
                    }))
                    async def send_text():
                        await websocket.send(json.dumps({"text": first_chunk, "try_trigger_generation": True}))
                        while True:
                            text_chunk = llm_response_queue.get(block=True)
                            if text_chunk is None: break
                            await websocket.send(json.dumps({"text": text_chunk, "try_trigger_generation": True}))
                        await websocket.send(json.dumps({"text": ""}))
                    async def receive_audio():
                        while True:
                            try:
                                message_str = await websocket.recv()
                                message = json.loads(message_str)
                                if message.get("audio"):
                                    audio_data = np.frombuffer(base64.b64decode(message["audio"]), dtype=np.int16)
                                    audio_out_queue.put(audio_data)
                                if message.get('isFinal'): break
                            except websockets.exceptions.ConnectionClosed: break
                        assistant_is_speaking.clear()
                        print("✅ Playback complete.")
                    await asyncio.gather(send_text(), receive_audio())
            try:
                asyncio.run(tts_stream())
            except Exception as e:
                print(f"❌ TTS/Playback Error: {e}")
                assistant_is_speaking.clear()

# --- Main Function ---
def main():
    print("🟢 Agentic Call Center (WebSocket Streaming Version)")
    print("Initializing...")
    threading.Thread(target=audio_input_and_transcription_thread, daemon=True).start()
    threading.Thread(target=llm_thread_main, daemon=True).start()
    threading.Thread(target=tts_playback_thread_main, daemon=True).start()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()