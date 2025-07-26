import os
import queue
import threading
import sounddevice as sd
import numpy as np
import wave
from dotenv import load_dotenv
from deepgram import Deepgram
import requests
from playsound import playsound
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import time
# Setup paths for ffmpeg
from pydub import AudioSegment
AudioSegment.converter = r"D:\Download\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"D:\Download\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe"

# Load API keys
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "zT03pEAEi0VHKciJODfn")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize APIs
dg_client = Deepgram(DEEPGRAM_API_KEY)

chat = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

# Define system prompt
system_prompt = """
.
- Fees: Class 11 ₹45,000 (offline), Class 12 ₹30,000, MH-CET ₹60,000.
- Centers: Pune (FC Road), Mumbai (Andheri), Nashik (College Road).
- Faculty: Prof. Arjun Deshmukh (Physics), Dr. Priya Kulkarni (Chemistry), etc.
Speak Hinglish in a natural, empathetic tone to help students pick the right course.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{user_input}")
])

# Audio settings
samplerate = 16000
channels = 1
input_device_index = 1
RECORD_DIR = "recordings"
OUTPUT_DIR = "output_audio"
os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper: save audio
def save_wav(audio_data, filename):
    filepath = os.path.join(RECORD_DIR, filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    return filepath

# Transcribe
def transcribe_audio(filepath):
    try:
        with open(filepath, 'rb') as audio_file:
            source = {'buffer': audio_file, 'mimetype': 'audio/wav'}
            response = dg_client.transcription.sync_prerecorded(source, {
                'model': 'nova-2',
                'language': 'hi',
                'punctuate': True,
                'smart_format': True
            })
            transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
            print(f"🗣 User said: {transcript}")
            return transcript.strip()
    except Exception as e:
        print("⚠ Deepgram Error:", e)
        return ""

# LLM response
def get_llm_response(user_text):
    try:
        formatted_prompt = prompt_template.format_messages(user_input=user_text)
        response = chat.invoke(formatted_prompt)
        print(f"🤖 AI Response: {response.content}")
        return response.content.strip()
    except Exception as e:
        print("⚠ LLM Error:", e)
        return "Sorry, kuch error aaya hai."

# Text to Speech
def synthesize_speech(text, filename="response.mp3"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    else:
        print(f"❌ ElevenLabs Error: {response.status_code} - {response.text}")
        return None

# Play Audio
def play_audio(filepath):
    try:
        playsound(filepath)
    except Exception as e:
        print(f"⚠ Audio Play Error: {e}")

# Silence detection
def detect_silence(indata, threshold=0.01):
    volume_norm = np.linalg.norm(indata) * 10
    return volume_norm < threshold

# Continuous listening
def continuous_listen():
    print("🎤 Listening... Speak anytime.")
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        audio_buffer = []
        silent_chunks = 0
        while True:
            data = q.get()
            audio_buffer.append(data)

            if detect_silence(data):
                silent_chunks += 1
            else:
                silent_chunks = 0

            # If silent for a while, consider end of speech
            if silent_chunks > 20:  # adjust sensitivity
                print("⏸ Detected silence, processing...")
                break

        audio_data = np.concatenate(audio_buffer, axis=0)
        return audio_data

# Main loop
def main():
    print("🟢 Agentic Call Center (Continuous Listening)")
    while True:
        # Listen
        audio = continuous_listen()
        timestamp = str(int(time.time()))
        wav_file = save_wav(audio, f"{timestamp}.wav")

        # Transcribe
        transcript = transcribe_audio(wav_file)
        if not transcript:
            print("❌ Couldn’t understand. Try again.")
            continue

        # LLM response
        llm_reply = get_llm_response(transcript)

        # Synthesize & play
        audio_file = synthesize_speech(llm_reply, f"response_{timestamp}.mp3")
        if audio_file:
            play_audio(audio_file)


if __name__ == "__main__":
    main()
