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

# Audio settings for Deepgram
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 8000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0

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
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio status: {status}")
        
        # Only listen when not speaking to avoid feedback
        if not self.is_speaking:
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

    async def stream_deepgram_transcription(self):
        """Stream audio to Deepgram for real-time transcription"""
        uri = "wss://api.deepgram.com/v1/listen"
        
        extra_headers = {
            "Authorization": f"Token {self.deepgram_api_key}"
        }
        
        # Connection parameters
        params = {
            "model": "nova-2",
            "language": "hi",
            "punctuate": "true",
            "smart_format": "true",
            "interim_results": "false",
            "endpointing": "1000",
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE),
            "channels": str(CHANNELS)
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
                print("🔗 Connected to Deepgram")
                
                async def send_audio():
                    try:
                        silent_packet = np.zeros(CHUNK_SIZE, dtype=np.int16)
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
                            else:
                              await websocket.send(silent_packet.tobytes())
                              await asyncio.sleep(5)
                            
                    except websockets.exceptions.ConnectionClosed:
                        print("🔌 WebSocket connection closed")
                        self.connection_active = False
                    except Exception as e:
                        print(f"⚠ Audio send error: {e}")
                        self.connection_active = False
                
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
                        self.connection_active = False
                    except Exception as e:
                        print(f"⚠ Transcript receive error: {e}")
                        self.connection_active = False
                
                # Send keep-alive message
                async def keep_alive():
                    while self.connection_active:
                        try:
                            await websocket.ping()
                            await asyncio.sleep(30)
                        except:
                            self.connection_active = False
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
        
    async def process_voice_queries(self):
        """Process transcribed voice queries through RAG system and queue for TTS"""
        last_transcript = None
        while True:
            try:
                if not self.transcript_queue.empty():
                    transcript = self.transcript_queue.get()
                    
                    # Skip very short transcripts
                    if len(transcript.strip()) < 3 or transcript == last_transcript:
                        continue
                    await asyncio.sleep(0.3)
                    if not self.transcript_queue.empty() and self.transcript_queue.queue[0] == transcript:
                        continue
                    last_transcript=transcript
                    print(f"\n🤖 Processing voice query: {transcript}")
                    
                    # Detect query language style
                    query_style = detect_language_style(transcript)
                    print(f"[Detected style: {query_style}]")
                    
                    # Process through RAG chain
                    result = await asyncio.to_thread(self.qa_chain, {"question": transcript})
                    answer = result["answer"]
                    
                    print("\n--- Voice Answer ")
                    print(answer)
                    
                    # Show source documents if available (but don't speak them)
                    if "source_documents" in result and result["source_documents"]:
                        print("\n--- Sources ")
                        for i, doc in enumerate(result["source_documents"][:2]):
                            print(f"Source {i+1}: {doc.page_content[:150]}...")
                    print("-" * 60)
                    
                    # Queue answer for TTS
                    self.tts_queue.put({
                        "text": answer,
                        "language_style": query_style
                    })
                    
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"❌ Query processing error: {e}")

    async def text_to_speech_processor(self):
        """Process TTS queue and generate speech using Edge TTS"""
        while True:
            try:
                if not self.tts_queue.empty():
                    tts_data = self.tts_queue.get()
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
        """Generate speech using Edge TTS and play it"""
        try:
            self.is_speaking = True
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
            
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
                    
            except Exception as play_error:
                print(f"❌ Audio playback error: {play_error}")
            finally:
                audio_data.close()
                self.is_speaking = False
                
        except Exception as e:
            print(f"❌ Speech generation error: {e}")
            self.is_speaking = False

    async def fallback_transcription(self):
        """Fallback to batch transcription if streaming fails"""
        print("🔄 Using fallback batch transcription...")
        import requests
        
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
        import requests
        try:
            url = "https://api.deepgram.com/v1/listen"
            headers = {
                "Authorization": f"Token {self.deepgram_api_key}",
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

    async def start_voice_interaction(self):
        """Start complete voice interaction with RAG system and TTS"""
        print("🟢 Starting Voice-Enabled RAG Assistant with TTS")
        print("🎤 Listening for voice input... Speak naturally!")
        print("🔊 I will speak back to you using Edge TTS Madhur voice!")
        print("📝 You can also type 'exit' to quit or use Ctrl+C")
        
        # Start text input thread for fallback
        text_thread = threading.Thread(target=self.text_input_handler, daemon=True)
        text_thread.start()
        
        # Start audio input stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self.audio_callback,
            blocksize=CHUNK_SIZE
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
                elif text_query.lower().strip() in ("clear", "reset", ):
                    self.qa_chain.memory.clear()
                    print("Memory cleared! Conversation history cleared.")
                elif text_query.strip():
                    # Add text query to transcript queue for processing
                    print(f"💬 Text input: {text_query}")
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
        temperature=1.2
        
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

    
    print("[+] Building ConversationalRetrievalChain with memory...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
        verbose=False
    )

    return qa_chain

async def async_main(pdf_path: str, groq_api_key: str, deepgram_api_key: str, index_save_path: str):
    """Async main function to handle voice RAG with TTS"""
    try:
        # Build RAG pipeline
        qa_chain = build_rag(pdf_path, groq_api_key, deepgram_api_key, index_save_path)
        
        print("[+] RAG pipeline ready with Hinglish support and conversation memory!")
        print("[+] Voice input enabled - speak in English, Hindi, or Hinglish!")
        print("[+] Voice output enabled with Edge TTS Madhur voice!")
                
        voice_assistant = VoiceRAGAssistant(qa_chain, deepgram_api_key)
        
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
        asyncio.run(async_main(pdf_path, GROQ_API_KEY, DEEPGRAM_API_KEY, args.index))
    except KeyboardInterrupt:
        print("\n👋 Shutting down... Dhanyawad!")
    except Exception as e:
        print(f"\nFatal error : {e}")
        sys.exit(1)
