import os
import sys
import argparse
import tempfile
import asyncio
import json
import queue
import threading
import time
import hashlib
from dotenv import load_dotenv
from typing import List, Dict, Any
import requests
from io import BytesIO
import base64

# Twilio imports
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from flask import Flask, request, Response
from flask_cors import CORS

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

# ---------- Twilio Configuration ----------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")  # Your Twilio number
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------- Custom Prompt Template for Phone Calls ----------
PHONE_CALL_PROMPT_TEMPLATE = """You are a professional AI call agent making outbound phone calls.

Primary Goal: Use the provided context to answer the person's questions precisely and have natural conversations. If the answer is not in the context, politely state that you don't know. Do not invent answers.

Language & Style:
- Respond in the same language the person uses (e.g., English question = English response)
- Keep responses conversational and clear, suitable for a phone call (max 2-3 sentences)
- Speak like a human - be natural and friendly
- Remember you are an AI call agent speaking to someone on the phone
- Be concise but informative - people don't like long phone conversations
- Be friendly, professional, and helpful

Phone Call Behavior:
- Greet people naturally when they answer
- Ask if they have a moment to talk
- Handle interruptions gracefully
- Be patient and clear
- End calls politely when appropriate

Important: Keep responses SHORT and CONVERSATIONAL for phone calls. Long responses will lose the listener's attention.

Previous conversation:
{chat_history}

Context from the document:
{context}

Question: {question}

Answer (keep it short and conversational):"""

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
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
        
    def set(self, query: str, response: str):
        """Cache response"""
        key = self._get_key(query)
        
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

# ---------- Twilio AI Call Agent ----------
class TwilioAICallAgent:
    def __init__(self, qa_chain, twilio_client, elevenlabs_api_key):
        self.qa_chain = qa_chain
        self.twilio_client = twilio_client
        self.elevenlabs_api_key = elevenlabs_api_key
        self.response_cache = ResponseCache(max_size=50)
        
        # Call state management
        self.active_calls = {}  # call_sid -> call_info
        self.call_conversations = {}  # call_sid -> conversation_history
        
    def make_outbound_call(self, to_number: str, context: str = "general", custom_message: str = None):
        """Make an outbound call to a phone number - SYNCHRONOUS"""
        try:
            print(f"📞 Making outbound call to: {to_number}")
            print(f"📋 Context: {context}")
            
            # Create the webhook URL for initial call handling
            webhook_url = f"{request.url_root}initial_call"
            
            # Make the call - Twilio will call our webhook
            call = self.twilio_client.calls.create(
                url=webhook_url,
                to=to_number,
                from_=TWILIO_PHONE_NUMBER,
                method='POST',
                record=True,
                status_callback=f"{request.url_root}call_status",
                status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
                status_callback_method='POST'
            )
            
            # Store call information
            self.active_calls[call.sid] = {
                'to_number': to_number,
                'context': context,
                'status': 'initiated',
                'start_time': time.time(),
                'custom_message': custom_message
            }
            
            self.call_conversations[call.sid] = []
            
            print(f"✅ Call initiated! Call SID: {call.sid}")
            return call.sid
            
        except Exception as e:
            print(f"❌ Error making outbound call: {e}")
            return None
    
    def process_speech_input(self, call_sid: str, speech_result: str, confidence: float = 0.0):
        """Process speech input from the person on the call - SYNCHRONOUS"""
        try:
            if call_sid not in self.active_calls:
                print(f"⚠️ Unknown call SID: {call_sid}")
                return None
            
            print(f"🗣️ Call {call_sid}: Person said: {speech_result}")
            print(f"🎯 Confidence: {confidence}")
            
            # Add to conversation history
            self.call_conversations[call_sid].append({
                'speaker': 'person',
                'text': speech_result,
                'timestamp': time.time()
            })
            
            # Process the query
            response = self.get_ai_response(speech_result, call_sid)
            
            # Add AI response to conversation history
            self.call_conversations[call_sid].append({
                'speaker': 'ai',
                'text': response,
                'timestamp': time.time()
            })
            
            return response
            
        except Exception as e:
            print(f"❌ Error processing speech input: {e}")
            return "I'm sorry, I'm having trouble processing that. Could you please repeat?"
    
    def get_ai_response(self, query: str, call_sid: str = None):
        """Get AI response using the RAG system - SYNCHRONOUS"""
        try:
            # Check cache first
            cached_response = self.response_cache.get(query)
            if cached_response:
                print(f"🔄 Using cached response for: {query[:30]}...")
                return cached_response
            
            # Get response from QA chain
            print(f"🤖 Processing query: {query[:50]}...")
            result = self.qa_chain({"question": query})
            response = result["answer"]
            
            # Clean up the response for phone conversation
            response = response.strip()
            if len(response) > 200:  # Keep responses short for phone calls
                sentences = response.split('. ')
                response = '. '.join(sentences[:2])
                if not response.endswith('.'):
                    response += '.'
            
            # Cache the response
            self.response_cache.set(query, response)
            print(f"✅ AI response generated: {response}")
            return response
                
        except Exception as e:
            print(f"❌ Error getting AI response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. Could you please try again?"
    
    def get_call_info(self, call_sid: str):
        """Get information about a specific call"""
        return self.active_calls.get(call_sid, {})
    
    def get_call_conversation(self, call_sid: str):
        """Get conversation history for a specific call"""
        return self.call_conversations.get(call_sid, [])
    
    def end_call(self, call_sid: str):
        """End a specific call"""
        try:
            if call_sid in self.active_calls:
                call = self.twilio_client.calls(call_sid).update(status='completed')
                print(f"📞 Call {call_sid} ended")
                
                # Clean up call data
                if call_sid in self.active_calls:
                    del self.active_calls[call_sid]
                if call_sid in self.call_conversations:
                    del self.call_conversations[call_sid]
                
                return True
        except Exception as e:
            print(f"❌ Error ending call: {e}")
        return False

# ---------- RAG Builder ----------
def build_rag_pipeline(pdf_path: str, groq_api_key: str, index_save_path: str = "faiss_index"):
    """Build the RAG pipeline for the AI call agent"""
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

    # Text splitting and embeddings
    print("[+] Splitting into optimized chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    print(f"[+] Created {len(docs)} chunks.")

    if len(docs) == 0:
        raise RuntimeError("No chunks produced after splitting.")

    print("[+] Creating embeddings (multilingual model)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # FAISS index creation/loading
    if os.path.exists(index_save_path):
        print(f"[+] Loading existing FAISS index from '{index_save_path}'...")
        vectorstore = FAISS.load_local(index_save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[+] Building FAISS index...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        print(f"[+] Saving FAISS index to '{index_save_path}'...")
        vectorstore.save_local(index_save_path)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Groq LLM
    print("[+] Initializing Groq LLM...")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.3,
        streaming=False  # Disable streaming for simpler synchronous handling
    )

    # Custom prompt for phone calls
    custom_prompt = PromptTemplate(
        template=PHONE_CALL_PROMPT_TEMPLATE,
        input_variables=["chat_history", "context", "question"]
    )

    # Memory
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # ConversationalRetrievalChain
    print("[+] Building ConversationalRetrievalChain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
        verbose=False,
        max_tokens_limit=2000
    )

    return qa_chain

# ---------- Flask Web App for Twilio Webhooks ----------
app = Flask(__name__)
CORS(app)

# Global AI agent instance
ai_agent = None

@app.route('/make_call', methods=['POST'])
def make_call():
    """Endpoint to initiate an outbound call"""
    try:
        data = request.get_json()
        to_number = data.get('to_number')
        context = data.get('context', 'general')
        custom_message = data.get('custom_message')
        
        if not to_number:
            return {'error': 'to_number is required'}, 400
        
        # Make the call
        call_sid = ai_agent.make_outbound_call(to_number, context, custom_message)
        
        if call_sid:
            return {'success': True, 'call_sid': call_sid}
        else:
            return {'error': 'Failed to initiate call'}, 500
            
    except Exception as e:
        print(f"❌ Error in make_call endpoint: {e}")
        return {'error': str(e)}, 500

@app.route('/initial_call', methods=['POST'])
def initial_call():
    """Handle initial call connection - this is called when the call is answered"""
    try:
        call_sid = request.form.get('CallSid')
        call_status = request.form.get('CallStatus')
        
        print(f"📞 Initial call handler - Call SID: {call_sid}, Status: {call_status}")
        
        # Create TwiML response with initial greeting
        twiml = VoiceResponse()
        
        # Get custom message if available
        call_info = ai_agent.get_call_info(call_sid)
        custom_message = call_info.get('custom_message') if call_info else None
        
        if custom_message:
            twiml.say(custom_message, voice='alice', language='en-US')
        else:
            twiml.say("Hello! This is an AI assistant. I have some information to share with you. How can I help you today?", 
                      voice='alice', language='en-US')
        
        # Start the conversation loop
        gather = twiml.gather(
            input='speech',
            action='/handle_speech',
            method='POST',
            speech_timeout='auto',
            timeout=10,
            language='en-US',
            enhanced=True,
            speech_model='experimental_conversations'
        )
        
        gather.say("Please go ahead and speak.", voice='alice', language='en-US')
        
        # Fallback if no speech detected
        twiml.say("I didn't hear anything. Let me try again.", voice='alice', language='en-US')
        twiml.redirect('/initial_call')
        
        return Response(str(twiml), mimetype='text/xml')
        
    except Exception as e:
        print(f"❌ Error in initial_call: {e}")
        twiml = VoiceResponse()
        twiml.say("I'm sorry, there was an error. Please try calling again.", voice='alice', language='en-US')
        twiml.hangup()
        return Response(str(twiml), mimetype='text/xml')

@app.route('/call_status', methods=['POST'])
def call_status():
    """Handle Twilio call status callbacks"""
    try:
        call_sid = request.form.get('CallSid')
        call_status = request.form.get('CallStatus')
        
        print(f"📞 Call {call_sid} status: {call_status}")
        
        if call_sid in ai_agent.active_calls:
            ai_agent.active_calls[call_sid]['status'] = call_status
            
            if call_status == 'completed':
                # Clean up completed call
                if call_sid in ai_agent.active_calls:
                    del ai_agent.active_calls[call_sid]
                if call_sid in ai_agent.call_conversations:
                    del ai_agent.call_conversations[call_sid]
        
        return Response(status=200)
        
    except Exception as e:
        print(f"❌ Error in call_status endpoint: {e}")
        return Response(status=500)

@app.route('/handle_speech', methods=['POST'])
def handle_speech():
    """Handle speech input from the call - STT -> LLM -> TTS workflow"""
    try:
        call_sid = request.form.get('CallSid')
        speech_result = request.form.get('SpeechResult', '')
        confidence = float(request.form.get('Confidence', 0.0))
        
        print(f"🗣️ Speech received for call {call_sid}: {speech_result} (confidence: {confidence})")
        
        # Check if we have valid speech input
        if not speech_result.strip() or confidence < 0.3:
            print(f"⚠️ Low confidence or empty speech: {confidence}")
            twiml = VoiceResponse()
            twiml.say("I didn't quite catch that. Could you please repeat?", voice='alice', language='en-US')
            
            # Continue listening
            gather = twiml.gather(
                input='speech',
                action='/handle_speech',
                method='POST',
                speech_timeout='auto',
                timeout=10,
                language='en-US',
                enhanced=True,
                speech_model='experimental_conversations'
            )
            
            twiml.say("I'm still here. Please go ahead.", voice='alice', language='en-US')
            return Response(str(twiml), mimetype='text/xml')
        
        # Process the speech with AI (STT -> LLM workflow)
        ai_response = ai_agent.process_speech_input(call_sid, speech_result, confidence)
        
        if ai_response:
            print(f"🤖 AI Response: {ai_response}")
            
            # Create TwiML response (TTS part of workflow)
            twiml = VoiceResponse()
            
            # Speak the AI response
            twiml.say(ai_response, voice='alice', language='en-US')
            
            # Check if this is a goodbye/ending phrase
            ending_phrases = ['goodbye', 'bye', 'thank you', 'thanks', 'that\'s all', 'hang up']
            if any(phrase in ai_response.lower() for phrase in ending_phrases):
                twiml.say("Have a great day! Goodbye!", voice='alice', language='en-US')
                twiml.hangup()
                return Response(str(twiml), mimetype='text/xml')
            
            # Continue the conversation - gather more speech
            gather = twiml.gather(
                input='speech',
                action='/handle_speech',
                method='POST',
                speech_timeout='auto',
                timeout=15,  # Wait up to 15 seconds for response
                language='en-US',
                enhanced=True,
                speech_model='experimental_conversations'
            )
            
            gather.say("Is there anything else I can help you with?", voice='alice', language='en-US')
            
            # Fallback if no response
            twiml.say("Thank you for calling. Have a great day!", voice='alice', language='en-US')
            twiml.hangup()
            
            return Response(str(twiml), mimetype='text/xml')
        else:
            # Error in AI processing
            print("❌ No AI response generated")
            twiml = VoiceResponse()
            twiml.say("I'm sorry, I'm having trouble processing that right now. Could you please try again?", 
                      voice='alice', language='en-US')
            
            # Try to continue conversation
            gather = twiml.gather(
                input='speech',
                action='/handle_speech',
                method='POST',
                speech_timeout='auto',
                timeout=10,
                language='en-US',
                enhanced=True
            )
            
            twiml.say("Thank you for your patience.", voice='alice', language='en-US')
            return Response(str(twiml), mimetype='text/xml')
            
    except Exception as e:
        print(f"❌ Error in handle_speech endpoint: {e}")
        twiml = VoiceResponse()
        twiml.say("I'm sorry, there was a technical error. Please try calling again later.", 
                  voice='alice', language='en-US')
        twiml.hangup()
        return Response(str(twiml), mimetype='text/xml')

@app.route('/call_info/<call_sid>', methods=['GET'])
def get_call_info(call_sid):
    """Get information about a specific call"""
    try:
        call_info = ai_agent.get_call_info(call_sid)
        conversation = ai_agent.get_call_conversation(call_sid)
        
        return {
            'call_info': call_info,
            'conversation': conversation
        }
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/end_call/<call_sid>', methods=['POST'])
def end_call(call_sid):
    """End a specific call"""
    try:
        success = ai_agent.end_call(call_sid)
        return {'success': success}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/active_calls', methods=['GET'])
def get_active_calls():
    """Get all active calls"""
    try:
        return {'active_calls': ai_agent.active_calls}
    except Exception as e:
        return {'error': str(e)}, 500

# ---------- Main Function ----------
def main():
    """Main function to initialize the AI call agent"""
    global ai_agent
    
    try:
        print("🚀 Starting Twilio AI Call Agent...")
        
        # Check required environment variables
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, GROQ_API_KEY]):
            print("❌ Missing required environment variables!")
            print("Required: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, GROQ_API_KEY")
            print("Optional: ELEVENLABS_API_KEY (for advanced TTS)")
            sys.exit(1)
        
        # Initialize Twilio client
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("✅ Twilio client initialized")
        
        # Build RAG pipeline
        pdf_path = "RAGpdf.pdf"  # Default PDF path
        if not os.path.exists(pdf_path):
            pdf_path = input("Enter path to PDF document: ").strip()
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            sys.exit(1)
        
        qa_chain = build_rag_pipeline(pdf_path, GROQ_API_KEY)
        print("✅ RAG pipeline ready")
        
        # Initialize AI agent
        ai_agent = TwilioAICallAgent(qa_chain, twilio_client, ELEVENLABS_API_KEY)
        print("✅ AI Call Agent initialized")
        
        print("\n🎯 AI Call Agent Ready!")
        print("📞 Workflow: STT (Speech-to-Text) -> LLM (AI Processing) -> TTS (Text-to-Speech)")
        print("🔄 Features:")
        print("   • Make outbound calls to any phone number")
        print("   • Natural AI conversations using RAG knowledge")
        print("   • Real-time speech processing with Twilio")
        print("   • Conversation memory and context")
        print("   • Call recording and monitoring")
        
        print("\n🌐 Web endpoints:")
        print("   • POST /make_call - Initiate a call")
        print("   • GET /active_calls - View active calls")
        print("   • GET /call_info/<call_sid> - Get call details")
        print("   • POST /end_call/<call_sid> - End a call")
        
        print("\n📱 To make a call, send POST to /make_call with:")
        print("   {")
        print('     "to_number": "+1234567890",')
        print('     "context": "general",')
        print('     "custom_message": "Hello! I have information to share..."')
        print("   }")
        
        # Start Flask app
        print("\n🚀 Starting Flask web server on port 5000...")
        print("🔗 Make sure your Twilio webhook URL is set to: http://your-domain.com:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)  # Disable debug for production
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
