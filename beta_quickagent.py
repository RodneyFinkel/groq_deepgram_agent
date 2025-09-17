import asyncio
from dotenv import load_dotenv
import os
import pyaudio
import logging
import re

from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents
from deepgram import Microphone
from alpha_DocumentContextManager import DocumentContextManager
from chunk_config import CHUNK_SIZE_LLM, CHUNK_OVERLAP_LLM
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class LanguageModelProcessor:
    def __init__(self, context_manager=None):
        self.llm = ChatGroq(
            temperature=0,
            model_name="deepseek-r1-distill-llama-70b",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            streaming=True,
            max_retries=3,
        )
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.context_manager = context_manager
        self.max_history_exchanges = 10

        with open('system_prompt2.txt', 'r') as file:
            system_prompt = file.read().strip()
                    
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )
        
        self.list_docs_pattern = re.compile(r"\b(list documents|what documents|available documents|show documents|documents in context)\b", re.IGNORECASE)
    
    def chunk_text_by_tokens(self, text, chunk_size=1000, overlap=200):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            i += chunk_size - overlap
        return chunks

    def process(self, text, context=""):
        self.memory.chat_memory.add_user_message(text)
        max_total_tokens = CHUNK_SIZE_LLM

        if self.context_manager and self.list_docs_pattern.search(text):
            all_data = self.context_manager.collection.get(include=['documents', 'metadatas'])
            doc_list = []
            for doc_id, metadata in zip(all_data['ids'], all_data['metadatas']):
                filename = metadata.get('filename', 'Unknown')
                summary = metadata.get('summary', 'No summary available')
                doc_list.append(f"Document ID: {doc_id}, Filename: {filename}, Summary: {summary}")
            context = "Available documents:\n" + "\n".join(doc_list) if doc_list else "No documents available."
            context_tokens = len(self.tokenizer.encode(context))
            if context_tokens > max_total_tokens:
                tokens = self.tokenizer.encode(context, add_special_tokens=False)[:max_total_tokens]
                context = self.tokenizer.decode(tokens, skip_special_tokens=True)
                logging.warning(f"Document list context truncated to {max_total_tokens} tokens")
            self.memory.save_context({'input': text}, {'output': context})
            logging.info(f"Document list context added with {len(doc_list)} documents. context: {context[:100]}...")
        elif self.context_manager and context:
            chunk_size = CHUNK_SIZE_LLM // 3
            overlap = CHUNK_OVERLAP_LLM
            chunks = self.chunk_text_by_tokens(context, chunk_size=chunk_size, overlap=overlap)
            selected_chunks = []
            total_tokens = 0
            for chunk in chunks:
                chunk_tokens = len(self.tokenizer.encode(chunk, add_special_tokens=False))
                if total_tokens + chunk_tokens > max_total_tokens:
                    break
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            context_for_llm = " ".join(selected_chunks)
            self.last_chunking_info = {
                "num_chunks": len(selected_chunks),
                "chunk_size": chunk_size,
                "chunks": selected_chunks[:5]
            }
            logging.info(f"Chunking Info: {self.last_chunking_info}")
            context = context_for_llm
            system_message = f"Reference Document Context:\n{context}"
            self.memory.save_context({'input': text}, {'output': system_message})
            logging.info(f"System Message: {system_message[:50]}...Added")

        if len(self.memory.chat_memory.messages) > self.max_history_exchanges * 2:
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[-self.max_history_exchanges*2:]

        prompt_messages = self.memory.chat_memory.messages
        history_text = " ".join([msg.content for msg in prompt_messages])
        history_tokens = len(self.tokenizer.encode(history_text))
        context_tokens = len(self.tokenizer.encode(context))
        input_tokens = len(self.tokenizer.encode(text))
        total_prompt_tokens = history_tokens + context_tokens + input_tokens

        while total_prompt_tokens > max_total_tokens and len(prompt_messages) > 2:
            prompt_messages = prompt_messages[2:]
            history_text = " ".join([msg.content for msg in prompt_messages])
            history_tokens = len(self.tokenizer.encode(history_text))
            total_prompt_tokens = history_tokens + context_tokens + input_tokens
        self.memory.chat_memory.messages = prompt_messages 

        response = self.conversation.invoke({"text": text})
        self.memory.chat_memory.add_ai_message(response['text'])
        logging.info(f"LLM Response: {response['text'][:100]}...")
        return response['text']

def check_microphone():
    p = pyaudio.PyAudio()
    try:
        p.get_default_input_device_info()
        logging.info("Microphone detected")
        return True
    except Exception as e:
        logging.error(f"No microphone available: {e}")
        return False
    finally:
        p.terminate()

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm_response = ""
        self.context_manager = DocumentContextManager()
        self.llm = LanguageModelProcessor(context_manager=self.context_manager)
        self.transcription_active = False
        self.session_state = {"transcript": "", "response": ""}

    async def run_voice_agent(self):
        if not check_microphone():
            logging.error("No microphone available")
            return

        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)
        connection = deepgram.listen.asynclive.v("1")

        async def on_open(self, open, **kwargs):
            logging.info("Voice Agent WebSocket opened")

        async def on_message(self, result, **kwargs):
            if "transcript" in result and result["transcript"]:
                self.transcription_response = result["transcript"]
                self.session_state["transcript"] = self.transcription_response
                logging.info(f"Received transcript: {self.transcription_response}")

                # Inject RAG context
                context = ""
                if self.context_manager:
                    similar_docs = self.context_manager.get_similar_documents(self.transcription_response, top_k=5, similarity_threshold=0.7)
                    if similar_docs:
                        context_parts = []
                        for doc in similar_docs:
                            filename = doc['metadata'].get('filename', 'Unknown')
                            chunk_text = doc['document']
                            context_parts.append(f"From {filename}:\n{chunk_text}")
                        context = "\n\n".join(context_parts)
                        logging.info(f"RAG context retrieved: {context[:100]}...")

                # Process with LLM
                self.llm_response = self.llm.process(self.transcription_response, context)
                self.session_state["response"] = self.llm_response
                logging.info(f"LLM response: {self.llm_response[:100]}...")

                # API handles TTS; no explicit send needed
                await connection.send({"type": "Speak", "text": self.llm_response})

        async def on_error(self, error, **kwargs):
            logging.error(f"Voice Agent error: {error}")
            self.transcription_active = False
            self.session_state["transcript"] = ""
            self.session_state["response"] = f"Error: {error}"

        async def on_close(self, close, **kwargs):
            logging.info("Voice Agent WebSocket closed")
            self.transcription_active = False

        connection.on(LiveTranscriptionEvents.Open, on_open)
        connection.on(LiveTranscriptionEvents.Transcript, on_message)
        connection.on(LiveTranscriptionEvents.Error, on_error)
        connection.on(LiveTranscriptionEvents.Close, on_close)

        options = {
            "model": "nova-3-general",
            "llm": "custom",  # BYOM for Groq (handled via LLMChain)
            "tts_model": "aura-luna-en",
            "interruption_sensitivity": 0.8,  # Adjust for barge-in responsiveness
            "endpointing": 150,  # Matches your original setting
            "language": "en-US",
            "encoding": "linear16",
            "sample_rate": 16000,
            "smart_format": True
        }

        await connection.start(options)
        microphone = Microphone(connection.send)
        microphone.start()
        logging.info("Microphone started")

        try:
            while self.transcription_active:
                await asyncio.sleep(0.1)  # Keep loop alive
        finally:
            microphone.finish()
            await connection.finish()
            logging.info("Voice Agent session finished")

    def run_transcription(self):
        self.transcription_active = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_voice_agent())
        finally:
            loop.close()
            self.transcription_active = False
            logging.info("Transcription event loop closed")

    def stop_transcription(self):
        self.transcription_active = False
        self.session_state["transcript"] = ""
        self.session_state["response"] = ""
        logging.info("Transcription stopped")

if __name__ == "__main__":
    manager = ConversationManager()
    manager.run_transcription()