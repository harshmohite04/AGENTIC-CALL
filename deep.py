from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
import asyncio
import sounddevice as sd
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 8000

async def main():
    try:
        # Create Deepgram client
        dg_client = DeepgramClient(DEEPGRAM_API_KEY)

        # Set transcription options
        options = LiveOptions(
            model="general",
            language="hi",  # Hindi
            punctuate=True,
            interim_results=True,
        )

        # Create a websocket connection
        dg_connection = await dg_client.listen.websocket(options)

        # Callback for transcription events
        async def on_transcript(event):
            transcript = event.channel.alternatives[0].transcript
            if transcript.strip():
                print(f"🗣 You: {transcript}")
            if event.is_final:
                print("✅ Final Transcript:", transcript)
                await dg_connection.finish()  # Stop the stream

        # Register event handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        dg_connection.on(LiveTranscriptionEvents.Error, lambda e: print("❌ Error:", e))
        dg_connection.on(LiveTranscriptionEvents.Close, lambda: print("🔴 Connection closed"))

        print("🎤 Listening... Speak now.")

        # Start recording and send audio chunks
        def callback(indata, frames, time, status):
            if status:
                print("⚠️ Mic Status:", status)
            dg_connection.send(indata)

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                               dtype='int16', channels=CHANNELS, callback=callback):
            await dg_connection.finished  # Keep streaming until closed

    except Exception as e:
        print("❌ Fatal Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
