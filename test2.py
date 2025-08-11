import asyncio
import websockets
import json
import base64
import sounddevice as sd
import numpy as np
import time
import logging
import queue
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VoiceAssistantTestClient:
    def __init__(self, server_url="ws://localhost:8000/ws/voice/test-call-1"):
        self.server_url = server_url
        self.websocket = None
        self.is_recording = False
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.int16
        self.audio_queue = queue.Queue()  # Thread-safe queue for audio data
        
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            logging.info(f"✅ Connected to server: {self.server_url}")
            return True
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            return False
    
    async def send_test_message(self):
        """Send a test message to check connection"""
        test_message = {
            "type": "ping"
        }
        await self.websocket.send(json.dumps(test_message))
        logging.info("📤 Sent ping message")
    
    async def simulate_audio_stream(self, duration=5):
        """Simulate audio streaming for testing (without actual microphone)"""
        logging.info(f"🎵 Simulating audio stream for {duration} seconds...")
        
        # Generate some test audio data (silence with a beep pattern)
        frames_per_second = self.sample_rate // 1024  # Send chunks of 1024 samples
        total_frames = duration * frames_per_second
        
        for i in range(total_frames):
            # Generate test audio (mostly silence with occasional beep)
            samples_per_frame = 1024
            if i % (frames_per_second * 2) == 0:  # Beep every 2 seconds
                # Generate a simple beep
                t = np.linspace(0, samples_per_frame / self.sample_rate, samples_per_frame)
                audio_data = (np.sin(2 * np.pi * 440 * t) * 0.1 * 32767).astype(np.int16)
            else:
                # Generate silence
                audio_data = np.zeros(samples_per_frame, dtype=np.int16)
            
            # Encode as base64 and send
            audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            message = {
                "type": "audio_data",
                "data": audio_b64,
                "format": "pcm",
                "sample_rate": self.sample_rate
            }
            
            await self.websocket.send(json.dumps(message))
            await asyncio.sleep(1024 / self.sample_rate)  # Wait for frame duration
        
        logging.info("🔚 Audio simulation completed")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Real microphone callback - runs in audio thread"""
        if status:
            logging.warning(f"⚠️ Audio status: {status}")
        
        if self.is_recording:
            # Convert float32 to int16
            audio_int16 = (indata.flatten() * 32767).astype(np.int16)
            
            # Put audio data in queue (thread-safe)
            try:
                self.audio_queue.put_nowait(audio_int16)
            except queue.Full:
                logging.warning("⚠️ Audio queue full, dropping frame")
    
    async def send_audio_chunk(self, audio_data):
        """Send audio chunk to server"""
        try:
            audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            message = {
                "type": "audio_data",
                "data": audio_b64,
                "format": "pcm",
                "sample_rate": self.sample_rate
            }
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logging.error(f"❌ Error sending audio: {e}")
    
    async def start_real_microphone(self, duration=30):
        """Start recording from real microphone"""
        logging.info(f"🎤 Starting real microphone recording for {duration} seconds...")
        logging.info("💬 Speak now - your voice will be processed!")
        
        self.is_recording = True
        
        try:
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=1024,
                dtype='float32'
            ):
                await asyncio.sleep(duration)
        except Exception as e:
            logging.error(f"❌ Microphone error: {e}")
        finally:
            self.is_recording = False
            logging.info("🛑 Microphone recording stopped")
    
    async def listen_for_responses(self):
        """Listen for responses from the server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "status":
                    logging.info(f"📋 Status: {data.get('message')}")
                
                elif msg_type == "audio_response":
                    logging.info("🔊 Received audio response from server")
                    # You could decode and play the audio here if needed
                    audio_data = base64.b64decode(data.get('data'))
                    logging.info(f"🎵 Audio response size: {len(audio_data)} bytes")
                
                elif msg_type == "pong":
                    logging.info("🏓 Received pong response")
                
                else:
                    logging.info(f"📥 Received: {data}")
                    
        except websockets.exceptions.ConnectionClosed:
            logging.info("🔌 Server connection closed")
        except Exception as e:
            logging.error(f"❌ Error listening for responses: {e}")
    
    async def send_call_end(self):
        """Send call end signal"""
        message = {"type": "call_ended"}
        await self.websocket.send(json.dumps(message))
        logging.info("📞 Sent call end signal")
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logging.info("🔌 WebSocket connection closed")

async def main():
    print("🎯 Voice Assistant WebSocket Test Client")
    print("=" * 50)
    
    client = VoiceAssistantTestClient()
    
    # Connect to server
    if not await client.connect():
        return
    
    try:
        # Start listening for responses in background
        response_task = asyncio.create_task(client.listen_for_responses())
        
        # Test basic connection
        await client.send_test_message()
        await asyncio.sleep(1)
        
        print("\n🤖 Choose test mode:")
        print("1️⃣  Simulated audio (safe test - no microphone needed)")
        print("2️⃣  Real microphone (speak and test voice processing)")
        print("3️⃣  Quick connection test only")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            print("\n🎵 Running simulated audio test...")
            await client.simulate_audio_stream(duration=10)
            
        elif choice == "2":
            print("\n🎤 Starting real microphone test...")
            print("💡 Make sure your microphone is working!")
            print("💬 Speak clearly in Hindi/English - the assistant will respond!")
            input("Press Enter when ready...")
            
            await client.start_real_microphone(duration=30)
            
        elif choice == "3":
            print("\n⚡ Quick connection test...")
            await asyncio.sleep(2)
            
        else:
            print("❌ Invalid choice")
        
        # Send call end
        await client.send_call_end()
        await asyncio.sleep(1)
        
        # Cancel response listening task
        response_task.cancel()
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        logging.error(f"❌ Test error: {e}")
    finally:
        await client.close()
        print("✅ Test completed!")

if __name__ == "__main__":
    print("🚀 Starting WebSocket test client...")
    print("📡 Make sure your FastAPI server is running on localhost:8000")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")