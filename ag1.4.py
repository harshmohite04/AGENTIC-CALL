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
from io import BytesIO

# Edge TTS imports
import edge_tts

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Optional OCR imports (only used if needed)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Audio settings for Deepgram - IMPROVED SETTINGS
SAMPLE_RATE = 16000  # Changed from 20000 to standard 16kHz
CHANNELS = 1
CHUNK_SIZE = 4800  # Reduced chunk size for better responsiveness
SILENCE_THRESHOLD = 0.005  # Reduced threshold
SILENCE_DURATION = 2.0  # Increased silence duration
VAD_BUFFER_SIZE = 5  # Voice activity detection buffer

# Initialize pygame mixer for audio playback
pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.init()

# ---------- Custom Prompt Template for Hinglish ----------
HINGLISH_PROMPT_TEMPLATE = """You are a helpful AI assistant that can understand and respond in Hinglish (a mix of Hindi and English). 
You should match the language style of the user's question. If they ask in Hinglish, respond in Hinglish. 
If they ask in English, respond in English. If they ask in Hindi, respond in Hinglish.

Keep your responses concise and conversational, suitable for voice interaction. Avoid overly long explanations.

Use the following pieces of context to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

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

# ---------- Helpers ----------
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

def get_tts_voice(language_style: str) -> str:
    """Get appropriate Edge TTS voice based on detected language"""
    if language_style == "hinglish":
        return "hi-IN-MadhurNeural"  # Madhur voice for Hindi/Hinglish
    else:
        return "hi-IN-MadhurNeural"  # English voice for English queries

# ---------- Voice RAG Assistant with TTS ----------
class VoiceRAGAssistant:
    def __init__(self, qa_chain, deepgram_api_key):
        self.qa_chain = qa_chain
        self.deepgram_api_key = deepgram_api_key
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.is_listening = False
        self.is_speaking = False
        self.connection_active = False
        self.processing_query = False  # NEW: Prevent duplicate processing
        self.last_processed_transcript = ""  # NEW: Track last processed transcript
        self.processing_lock = asyncio.Lock()  # NEW: Async lock for processing
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio status: {status}")
        
        # Only listen when not speaking to avoid feedback and connection is active
        if not self.is_speaking and self.connection_active:
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
                
                # Only add to queue if there's space (prevent queue buildup)
                if self.audio_queue.qsize() < 100:  # Limit queue size
                    self.audio_queue.put(audio_int16)
        elif self.is_speaking:
            # During speech, we can still collect some audio but at reduced rate
            if self.audio_queue.qsize() < 10:
                volume = np.linalg.norm(indata) * 10
                if volume > SILENCE_THRESHOLD * 2:  # Higher threshold during speech
                    audio_flat = indata.flatten()
                    if audio_flat.dtype == np.float32:
                        audio_int16 = (audio_flat * 32767).astype(np.int16)
                    else:
                        audio_int16 = audio_flat.astype(np.int16)
                    self.audio_queue.put(audio_int16)

    async def stream_deepgram_transcription(self):
        """Stream audio to Deepgram for real-time transcription - IMPROVED"""
        uri = "wss://api.deepgram.com/v1/listen"
        
        extra_headers = {
            "Authorization": f"Token {self.deepgram_api_key}"
        }
        
        # IMPROVED Connection parameters
        params = {
            "model": "nova-2",
            "language": "hi",
            "punctuate": "true",
            "smart_format": "true", 
            "interim_results": "true",  # Enable interim results for better responsiveness
            "endpointing": "300",  # Reduced endpointing for faster response
            "vad_events": "true",  # Enable voice activity detection
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE),
            "channels": str(CHANNELS),
            "multichannel": "false",
            "alternatives": "1"
        }
        
        # Add parameters to URI
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
                print("🔗 Connected to Deepgram with improved settings")
                
                async def send_audio():
                    try:
                        silent_packet = np.zeros(CHUNK_SIZE, dtype=np.int16)
                        last_activity = time.time()
                        
                        while self.connection_active:
                            if not self.audio_queue.empty():
                                audio_data = self.audio_queue.get()
                                
                                # Ensure audio is in correct format
                                if audio_data.dtype != np.int16:
                                    audio_int16 = (audio_data * 32767).astype(np.int16)
                                else:
                                    audio_int16 = audio_data
                                
                                # Send audio bytes
                                await websocket.send(audio_int16.tobytes())
                                last_activity = time.time()
                            else:
                                # Send keepalive audio to prevent connection timeout
                                current_time = time.time()
                                if current_time - last_activity > 2.0:  # Send keepalive every 2 seconds
                                    await websocket.send(silent_packet.tobytes())
                                    last_activity = current_time
                                await asyncio.sleep(0.01)  # Reduced sleep time
                            
                    except websockets.exceptions.ConnectionClosed:
                        print("🔌 WebSocket connection closed in send_audio")
                        self.connection_active = False
                    except Exception as e:
                        print(f"⚠ Audio send error: {e}")
                        self.connection_active = False
                
                async def receive_transcripts():
                    try:
                        last_final_transcript = ""
                        async for message in websocket:
                            try:
                                data = json.loads(message)
                                
                                if 'type' in data and data['type'] == 'Results':
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
                                            # IMPROVED: Only process if different from last
                                            if transcript.strip() != last_final_transcript:
                                                last_final_transcript = transcript.strip()
                                                print(f"🗣 Final: {transcript}")
                                                
                                                # Add to queue only if not already processing
                                                if not self.processing_query:
                                                    self.transcript_queue.put(transcript.strip())
                                        elif transcript.strip():
                                            print(f"🗣 Interim: {transcript}")
                                
                            except (json.JSONDecodeError, KeyError) as e:
                                print(f"⚠ Parsing error: {e}")
                                continue
                                
                    except websockets.exceptions.ConnectionClosed:
                        print("🔌 Transcript receiver connection closed")
                        self.connection_active = False
                    except Exception as e:
                        print(f"⚠ Transcript receive error: {e}")
                        self.connection_active = False
                
                # Run tasks concurrently with proper error handling
                tasks = await asyncio.gather(
                    send_audio(),
                    receive_transcripts(),
                    return_exceptions=True
                )
                
                # Check if any task failed
                for i, task_result in enumerate(tasks):
                    if isinstance(task_result, Exception):
                        print(f"Task {i} failed with error: {task_result}")
                        
        except Exception as e:
            print(f"❌ Deepgram connection error: {e}")
            self.connection_active = False
            
        # Try to reconnect if connection was lost
        if self.connection_active:
            print("🔄 Connection lost, attempting to reconnect in 3 seconds...")
            await asyncio.sleep(3)
            if self.connection_active:  # Check if we haven't been told to stop
                await self.stream_deepgram_transcription()  # Recursive reconnect

    async def process_voice_queries(self):
        """Process transcribed voice queries through RAG system and queue for TTS - IMPROVED"""
        while True:
            try:
                if not self.transcript_queue.empty() and not self.processing_query:
                    async with self.processing_lock:  # Use async lock
                        if self.processing_query:  # Double-check
                            continue
                            
                        self.processing_query = True
                        transcript = self.transcript_queue.get()
                        
                        # Clear any remaining transcripts in queue to prevent duplicates
                        while not self.transcript_queue.empty():
                            self.transcript_queue.get()
                        
                        # Skip very short transcripts or duplicates
                        if len(transcript.strip()) < 3 or transcript == self.last_processed_transcript:
                            self.processing_query = False
                            continue
                            
                        self.last_processed_transcript = transcript
                        print(f"\n🤖 Processing voice query: {transcript}")
                        
                        # Detect query language style
                        query_style = detect_language_style(transcript)
                        print(f"[Detected style: {query_style}]")
                        
                        try:
                            # Process through RAG chain
                            result = await asyncio.to_thread(self.qa_chain, {"question": transcript})
                            answer = result["answer"]
                            
                            print("\n--- Voice Answer ---")
                            print(answer)
                            
                            # Show source documents if available (but don't speak them)
                            if "source_documents" in result and result["source_documents"]:
                                print("\n--- Sources ---")
                                for i, doc in enumerate(result["source_documents"][:2]):
                                    print(f"Source {i+1}: {doc.page_content[:150]}...")
                            print("-" * 60)
                            
                            # Queue answer for TTS (only once)
                            if not self.tts_queue.qsize() > 0:  # Don't queue if already processing TTS
                                self.tts_queue.put({
                                    "text": answer,
                                    "language_style": query_style
                                })
                        
                        except Exception as e:
                            print(f"❌ RAG processing error: {e}")
                        
                        finally:
                            self.processing_query = False
                    
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ Query processing error: {e}")
                self.processing_query = False

    async def text_to_speech_processor(self):
        """Process TTS queue and generate speech using Edge TTS - IMPROVED"""
        while True:
            try:
                if not self.tts_queue.empty() and not self.is_speaking:
                    tts_data = self.tts_queue.get()
                    
                    # Clear any remaining TTS requests to prevent duplicates
                    while not self.tts_queue.empty():
                        self.tts_queue.get()
                    
                    text = tts_data["text"]
                    language_style = tts_data["language_style"]
                    
                    print(f"🔊 Generating speech for: {text[:50]}...")
                    
                    # Get appropriate voice
                    voice = get_tts_voice(language_style)
                    print(f"🎙️ Using voice: {voice}")
                    
                    # Generate speech using Edge TTS
                    await self.generate_and_play_speech(text, voice)
                    
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ TTS processing error: {e}")

    async def generate_and_play_speech(self, text: str, voice: str):
        """Generate speech using Edge TTS and play it - IMPROVED"""
        try:
            self.is_speaking = True
            print(f"🔊 Starting speech generation and playback...")
            
            # Don't clear the audio queue completely - just pause audio collection temporarily
            # with self.audio_queue.mutex:
            #     self.audio_queue.queue.clear()
            
            # Create Edge TTS communicate object
            communicate = edge_tts.Communicate(text, voice)
            
            # Generate speech and save to BytesIO
            audio_data = BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])
            
            # Reset BytesIO position
            audio_data.seek(0)
            
            # Play audio using pygame
            try:
                pygame.mixer.music.load(audio_data)
                pygame.mixer.music.play()
                
                # Wait for audio to finish playing
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                    
                print("🔇 Speech playback completed")
                    
            except Exception as play_error:
                print(f"❌ Audio playback error: {play_error}")
            finally:
                audio_data.close()
                # Add small delay before re-enabling listening
                await asyncio.sleep(1.0)  # Increased delay
                self.is_speaking = False
                print("🎤 Re-enabled listening after speech")
                
        except Exception as e:
            print(f"❌ Speech generation error: {e}")
            self.is_speaking = False

    async def start_voice_interaction(self):
        """Start complete voice interaction with RAG system and TTS - IMPROVED"""
        print("🟢 Starting Voice-Enabled RAG Assistant with TTS")
        print("🎤 Listening for voice input... Speak naturally!")
        print("🔊 I will speak back to you using Edge TTS Madhur voice!")
        print("📝 You can also type 'exit' to quit or use Ctrl+C")
        print("🔧 Improved settings: 16kHz sampling, better VAD, duplicate prevention")
        
        # Start text input thread for fallback
        text_thread = threading.Thread(target=self.text_input_handler, daemon=True)
        text_thread.start()
        
        # Start audio input stream with improved settings
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self.audio_callback,
            blocksize=CHUNK_SIZE,
            dtype=np.float32  # Explicitly set dtype
        ):
            # Run all voice processing tasks concurrently
            await asyncio.gather(
                self.stream_deepgram_transcription(),
                self.process_voice_queries(),
                self.text_to_speech_processor(),
                return_exceptions=True
            )

    def text_input_handler(self):
        """Handle text input as fallback"""
        while True:
            try:
                text_query = input()
                if text_query.lower().strip() in ("exit", "quit", "bye", "alvida"):
                    print("Goodbye")
                    os._exit(0)
                elif text_query.lower().strip() in ("clear", "reset"):
                    self.qa_chain.memory.clear()
                    print("Memory cleared! Conversation history cleared.")
                elif text_query.strip():
                    # Add text query to transcript queue for processing
                    print(f"💬 Text input: {text_query}")
                    if not self.processing_query:
                        self.transcript_queue.put(text_query.strip())
            except (EOFError, KeyboardInterrupt):
                print("\nShutting down...")
                os._exit(0)

# ---------- Main RAG Builder ----------
def build_rag(pdf_path: str, groq_api_key: str, deepgram_api_key: str, index_save_path: str = "faiss_index"):
    # 1) Try loaders
    print(f"[+] Loading PDF: {pdf_path}")
    documents = try_load_with_pypdf(pdf_path)
    print(f"[+] PyPDFLoader returned {len(documents)} page-docs; total chars = {docs_total_text_length(documents)}")

    if len(documents) == 0 or docs_total_text_length(documents) < 200:
        print("[!] PyPDFLoader returned little/no text. Trying UnstructuredPDFLoader...")
        documents = try_load_with_unstructured(pdf_path)
        print(f"[+] UnstructuredPDFLoader returned {len(documents)} page-docs; total chars = {docs_total_text_length(documents)}")

    if len(documents) == 0 or docs_total_text_length(documents) < 200:
        print("[!] Loaders failed or returned too little text. Falling back to OCR (this requires poppler & tesseract installed).")
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR not available. Install pdf2image and pytesseract and system deps (poppler, tesseract).")
        ocr_text = ocr_pdf_to_text(pdf_path)
        if len(ocr_text.strip()) == 0:
            raise RuntimeError("OCR produced no text. Double-check PDF and OCR installations.")
        from langchain.schema import Document
        documents = [Document(page_content=ocr_text, metadata={"source": pdf_path})]
        print(f"[+] OCR produced {len(ocr_text)} characters of text.")

    # 2) Split text into chunks
    print("[+] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    print(f"[+] Created {len(docs)} chunks. Example chunk length: {len(docs[0].page_content) if docs else 0}")

    if len(docs) == 0:
        raise RuntimeError("No chunks produced after splitting. Abort.")

    # 3) Create embeddings (HuggingFace)
    print("[+] Creating embeddings (HuggingFace: sentence-transformers/all-MiniLM-L6-v2). This may download model on first run.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Quick sanity: embed the first chunk to ensure embeddings work
    sample_embedding = embeddings.embed_documents([docs[0].page_content])
    if not sample_embedding or len(sample_embedding[0]) == 0:
        raise RuntimeError("Embeddings returned empty vectors. Check HuggingFace embeddings installation.")

    # 4) Build / load FAISS index
    fs_prefix = index_save_path
    if os.path.exists(fs_prefix):
        print(f"[+] Found existing FAISS index at '{fs_prefix}'. Loading it...")
        vectorstore = FAISS.load_local(fs_prefix, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[+] Building FAISS index from documents (this may take a bit)...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        print(f"[+] Saving FAISS index to '{fs_prefix}' for reuse.")
        vectorstore.save_local(fs_prefix)

    # 5) Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 6) Groq LLM with enhanced model for multilingual support
    print("[+] Initializing Groq ChatGroq LLM...")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama3-70b-8192",
        temperature=0.3  # Reduced temperature for more consistent responses
    )

    # 7) Create custom prompt
    custom_prompt = PromptTemplate(
        template=HINGLISH_PROMPT_TEMPLATE,
        input_variables=["chat_history", "context", "question"]
    )

    # 8) Memory for conversation context
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # 9) ConversationalRetrievalChain with custom prompt and memory
    print("[+] Building ConversationalRetrievalChain with memory...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
        verbose=False  # Set to False for cleaner voice output
    )

    return qa_chain

# ---------- Async Main ----------
async def async_main(pdf_path: str, groq_api_key: str, deepgram_api_key: str, index_save_path: str):
    """Async main function to handle voice RAG with TTS"""
    try:
        # Build RAG pipeline
        qa_chain = build_rag(pdf_path, groq_api_key, deepgram_api_key, index_save_path)
        
        print("[+] RAG pipeline ready with Hinglish support and conversation memory!")
        print("[+] Voice input enabled - speak in English, Hindi, or Hinglish!")
        print("[+] Voice output enabled with Edge TTS Madhur voice!")
        print("[+] Improved duplicate prevention and word capture!")
        
        # Create voice assistant with TTS
        voice_assistant = VoiceRAGAssistant(qa_chain, deepgram_api_key)
        
        # Start voice interaction with TTS
        await voice_assistant.start_voice_interaction()
        
    except Exception as e:
        print(f"❌ Error in async main: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice-enabled RAG with TTS using Groq + Deepgram + Edge TTS")
    parser.add_argument("pdf", nargs="?", help="Path to PDF (if omitted, you'll be prompted).")
    parser.add_argument("--index", default="faiss_index", help="Folder path to save/load FAISS index")
    parser.add_argument("--memory-window", type=int, default=5, help="Number of conversation exchanges to remember")
    args = parser.parse_args()

    load_dotenv()
    
    # Check required API keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in environment. Put it in .env or export GROQ_API_KEY.")
        sys.exit(1)
        
    if not DEEPGRAM_API_KEY:
        print("ERROR: DEEPGRAM_API_KEY not found in environment. Put it in .env or export DEEPGRAM_API_KEY.")
        sys.exit(1)

    pdf_path = args.pdf
    if not pdf_path:
        pdf_path = input("Enter path to PDF: ").strip()

    if not os.path.isfile(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    try:
        # Run the async main function
        asyncio.run(async_main(pdf_path, GROQ_API_KEY, DEEPGRAM_API_KEY, args.index))
    except KeyboardInterrupt:
        print("\n👋 Shutting down... Dhanyawad!")
    except Exception as e:
        print(f"\nFatal error : {e}")
        sys.exit(1)
