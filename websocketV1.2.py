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
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Optional OCR imports (only used if needed)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

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

# RAG settings
RAG_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 200

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

class DocumentProcessor:
    """Handles PDF processing with OCR fallback"""
    @staticmethod
    def try_load_with_pypdf(pdf_path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(pdf_path)
            return loader.load()
        except Exception:
            return []

    @staticmethod
    def try_load_with_unstructured(pdf_path: str) -> List[Document]:
        try:
            loader = UnstructuredPDFLoader(pdf_path)
            return loader.load()
        except Exception:
            return []

    @staticmethod
    def ocr_pdf_to_text(pdf_path: str) -> str:
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR dependencies not available")
        images = convert_from_path(pdf_path)
        text_parts = []
        for img in images:
            text_parts.append(pytesseract.image_to_string(img))
        return "\n\n".join(text_parts)

    @staticmethod
    def load_pdf(pdf_path: str) -> List[Document]:
        """Load PDF with fallback to OCR if needed"""
        documents = DocumentProcessor.try_load_with_pypdf(pdf_path)
        if not documents or sum(len(d.page_content) for d in documents) < 200:
            documents = DocumentProcessor.try_load_with_unstructured(pdf_path)
        
        if not documents or sum(len(d.page_content) for d in documents) < 200:
            if not OCR_AVAILABLE:
                raise RuntimeError("PDF parsing failed and OCR not available")
            ocr_text = DocumentProcessor.ocr_pdf_to_text(pdf_path)
            if not ocr_text.strip():
                raise RuntimeError("OCR produced no text")
            documents = [Document(page_content=ocr_text, metadata={"source": pdf_path})]
        
        return documents

class RAGManager:
    """Manages RAG operations without hardcoded translations"""
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.embeddings = None
        self.index_path = RAG_INDEX_PATH
        self.translation_chain = None  # For dynamic translations
    
    def initialize_embeddings(self):
        """Initialize embeddings model"""
        if not self.embeddings:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
    
    def initialize_translation_chain(self, llm):
        """Initialize a simple translation prompt"""
        translation_prompt = ChatPromptTemplate.from_template(
            "Translate this Hinglish question to English while keeping "
            "all educational/technical terms unchanged:\n\n{question}\n\n"
            "English translation:"
        )
        self.translation_chain = translation_prompt | llm
    
    def build_index(self, documents: List[Document]):
        """Build or load FAISS index"""
        self.initialize_embeddings()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        docs = splitter.split_documents(documents)
        
        if os.path.exists(self.index_path):
            self.vectorstore = FAISS.load_local(self.index_path, self.embeddings)
        else:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.vectorstore.save_local(self.index_path)
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def initialize_qa_chain(self, llm):
        """Initialize QA chain with the given LLM"""
        if not self.retriever:
            raise RuntimeError("Retriever not initialized")
        
        # Initialize translation chain
        self.initialize_translation_chain(llm)
        
        template = """You are a helpful assistant for Takalkar Classes. 
        Answer the question in Hinglish based ONLY on the following context.
        If you don't know the answer, say "Mujhe is baare mein abhi exact info nahi hai".
        
        Context: {context}
        
        Question: {question}
        
        Answer in simple Hinglish:"""
        
        QA_PROMPT = ChatPromptTemplate.from_template(template)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
    
    async def translate_query(self, question: str) -> str:
        """Dynamically translate Hinglish to English using LLM"""
        if not self.translation_chain:
            return question
            
        try:
            # Get English keywords while preserving educational terms
            translation = await self.translation_chain.ainvoke({"question": question})
            translated = translation.content.strip()
            
            # Combine original and translated for better retrieval
            enhanced = f"{question} {translated}" if translated != question else question
            print(f"🔍 Enhanced query: {enhanced}")
            return enhanced
        except Exception as e:
            print(f"⚠ Translation failed, using original query: {e}")
            return question
    
    async def query(self, question: str) -> str:
        """Query the RAG system with automatic translation"""
        if not self.qa_chain:
            raise RuntimeError("QA chain not initialized")
        
        try:
            # Enhance the question dynamically
            enhanced_question = await self.translate_query(question)
            
            result = await self.qa_chain.ainvoke({"query": enhanced_question})
            
            # Debug retrieved documents
            print("\n📄 Retrieved documents:")
            for i, doc in enumerate(result['source_documents'][:3], 1):
                print(f"{i}. {doc.page_content[:100]}...")
            
            return result['result']
        except Exception as e:
            print(f"⚠ RAG query error: {e}")
            return None    
class LowLatencyVoiceAssistant:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.llm_text_queue = queue.Queue()
        self.tts_audio_queue = queue.Queue()
        
        # Initialize RAG
        self.rag_manager = RAGManager()
        self.rag_initialized = False
        
        # Initialize LLM with streaming
        self.streaming_handler = StreamingCallbackHandler(self.llm_text_queue)
        self.chat = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            streaming=True,
            callbacks=[self.streaming_handler]
        )
        
        self.system_prompt = """
        You are an AI calling assistant speaking on behalf of Takalkar Classes. 
You must always respond in Hinglish (Hindi+English mix) that's easy to understand.

Key rules:
1. First check if the question is about Takalkar Classes - if not, politely decline
2. For Takalkar Classes questions, first try to answer from the provided knowledge
3. If information isn't available, say "Mujhe is baare mein abhi exact info nahi hai"
4. Keep responses short (1-2 sentences max) and conversational
5. Always use simple Hinglish like:
   - "Takalkar Classes mein aapko XYZ course milta hai"
   - "Fees around ₹10,000 per month hai"
   - "Classes morning 8 baje se start hoti hain"

Example responses:
- "Takalkar Classes Kota ka famous coaching hai JEE and NEET ke liye"
- "Admission ke liye aapko test dena hoga, uske baad counseling hogi"
- "Mujhe is course ke exact fees ka pata nahi, main confirm karke bataunga"

Never:
- Give information not in the knowledge base
- Respond to non-educational queries
- Use complex English words
        """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{user_input}")
        ])
        
        self.is_listening = False
        self.is_speaking = False
    
    def initialize_rag(self, pdf_path: str):
        """Initialize RAG system with the given PDF"""
        try:
            # Load and process PDF
            documents = DocumentProcessor.load_pdf(pdf_path)
            
            # Build RAG index
            self.rag_manager.build_index(documents)
            self.rag_manager.initialize_qa_chain(self.chat)
            self.rag_initialized = True
            print("✅ RAG system initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize RAG: {e}")
            return False
    
    async def stream_deepgram_transcription(self):
        """Stream audio to Deepgram for real-time transcription"""
        uri = "wss://api.deepgram.com/v1/listen"
        extra_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        
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
                                if audio_data.dtype != np.int16:
                                    audio_int16 = (audio_data * 32767).astype(np.int16)
                                else:
                                    audio_int16 = audio_data
                                await websocket.send(audio_int16.tobytes())
                            await asyncio.sleep(0.02)
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
                                if 'type' in data:
                                    if data['type'] == 'Results':
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
                
                await asyncio.gather(
                    send_audio(),
                    receive_transcripts(),
                    return_exceptions=True
                )
                
        except Exception as e:
            print(f"❌ Deepgram connection error: {e}")
            await self.fallback_transcription()
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio status: {status}")
        
        if not self.is_speaking:
            volume = np.linalg.norm(indata) * 10
            if volume > SILENCE_THRESHOLD:
                self.is_listening = True
                audio_flat = indata.flatten()
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
                    
                    self.streaming_handler.current_text = ""
                    self.streaming_handler.full_response = ""
                    
                    if self.rag_initialized:
                        try:
                            rag_response = await self.rag_manager.query(transcript)
                            if rag_response:
                                print(f"🔍 RAG response: {rag_response}")
                                self.llm_text_queue.put(rag_response)
                                continue
                        except Exception as e:
                            print(f"⚠ RAG query failed: {e}")
                    
                    formatted_prompt = self.prompt_template.format_messages(user_input=transcript)
                    await asyncio.to_thread(self.chat.invoke, formatted_prompt)
                    
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
                    
                    audio_stream = BytesIO(audio_data)
                    audio_stream.seek(0)
                    
                    try:
                        pygame.mixer.music.load(audio_stream)
                        pygame.mixer.music.play()
                        
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                            
                    finally:
                        audio_stream.close()
                        self.is_speaking = False
                
                time.sleep(0.05)
            except Exception as e:
                print(f"❌ Audio playback error: {e}")
                self.is_speaking = False
    
    async def fallback_transcription(self):
        """Fallback to batch transcription if streaming fails"""
        print("🔄 Using fallback batch transcription...")
        
        while True:
            try:
                audio_buffer = []
                start_time = time.time()
                
                while time.time() - start_time < 3.0:
                    if not self.audio_queue.empty():
                        audio_buffer.append(self.audio_queue.get())
                    await asyncio.sleep(0.1)
                
                if audio_buffer:
                    combined_audio = np.concatenate(audio_buffer)
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        with wave.open(temp_file.name, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes((combined_audio * 32767).astype(np.int16).tobytes())
                        
                        transcript = await self.transcribe_batch(temp_file.name)
                        if transcript:
                            self.transcript_queue.put(transcript)
                        
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
    
    async def start_streaming(self, pdf_path: Optional[str] = None):
        """Start all streaming components"""
        print("🟢 Starting Low-Latency Voice Assistant")
        
        # Initialize RAG if PDF provided
        if pdf_path:
            self.initialize_rag(pdf_path)
        
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

async def main():
    assistant = LowLatencyVoiceAssistant()
    
    # Get PDF path from user or environment
    pdf_path = os.getenv("RAG_PDF_PATH")
    if not pdf_path:
        pdf_path = r"C:\Users\aryan\Downloads\IMP Broucher 24-25.pdf"
        if not pdf_path:
            pdf_path = None
    
    try:
        await assistant.start_streaming(pdf_path)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
