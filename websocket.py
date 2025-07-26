import os
import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json
from dotenv import load_dotenv
import requests
import threading
from playsound import playsound
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Load API Keys
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "zT03pEAEi0VHKciJODfn")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq Chat
chat = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0
)
system_prompt = """
You are a friendly academic counselor for EduSpark Academy.
Speak Hinglish in a natural, empathetic tone to help students pick the right course.
"""
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{user_input}")
])

# Audio Settings
SAMPLERATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# Send Audio to Deepgram
async def stream_audio(websocket):
    def callback(indata, frames, time, status):
        if status:
            print("Mic Error:", status)
        websocket.send(indata.tobytes())

    with sd.RawInputStream(samplerate=SAMPLERATE, blocksize=CHUNK_SIZE,
                           dtype="int16", channels=CHANNELS, callback=callback):
        print("🎤 Listening (Ctrl+C to stop)...")
        while True:
            await asyncio.sleep(0.1)

# Process Transcripts
async def process_transcripts(websocket):
    async for message in websocket:
        data = json.loads(message)
        if "channel" in data and "alternatives" in data["channel"]:
            transcript = data["channel"]["alternatives"][0]["transcript"]
            if transcript.strip() != "":
                print(f"🗣 User: {transcript}")
                asyncio.create_task(handle_response(transcript))

# Handle LLM and TTS
async def handle_response(user_input):
    # Get LLM response
    formatted_prompt = prompt_template.format_messages(user_input=user_input)
    response = chat.invoke(formatted_prompt)
    print(f"🤖 AI: {response.content}")

    # Synthesize speech
    audio_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": response.content,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    res = requests.post(audio_url, headers=headers, json=payload)
    if res.status_code == 200:
        audio_file = "response.mp3"
        with open(audio_file, "wb") as f:
            f.write(res.content)
        threading.Thread(target=playsound, args=(audio_file,), daemon=True).start()
    else:
        print(f"❌ ElevenLabs Error: {res.status_code}")

# Main
async def main():
    uri = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={SAMPLERATE}&channels={CHANNELS}&language=hi"
    async with websockets.connect(uri, extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}) as websocket:
        await asyncio.gather(
            stream_audio(websocket),
            process_transcripts(websocket)
        )

if __name__ == "__main__":
    asyncio.run(main())
