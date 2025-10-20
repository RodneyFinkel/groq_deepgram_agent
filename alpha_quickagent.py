import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import os
import pyaudio

from chunk_config import CHUNK_SIZE_LLM, CHUNK_OVERLAP_LLM
from transformers import AutoTokenizer

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from ddgs import DDGS
import logging # New
import re
from bs4 import BeautifulSoup

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

# NEW: Setup logging to catcg errors and debug info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class LanguageModelProcessor:
    def __init__(self, context_manager):
        self.llm = ChatGroq(temperature=0, 
                            model_name="llama-3.3-70b-versatile", # this is a new valid model 
                            groq_api_key=os.getenv("GROQ_API_KEY"), 
                            streaming=True,
                            max_retries=3,
                            ) 
        
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # NEW
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.context_manager = context_manager
        logging.info(f"LanguageModelProcessor using context_manager instance ID: {self.context_manager.id}")
        self.max_history_exchanges = 10

        # Load the system prompt from a file
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
        self.web_search_pattern = re.compile(r"\b(web search|online research|current information|latest information|duckduckgo|search web|search online)\b", re.IGNORECASE)  # NEWx
        self.online_research_enabled = True # Default to true
        self.browse_page_instructions = "Extract the main content, key facts, and relevant details from the page. Focus on body text, ignore navigation, ads, and scripts. Limit to 400 words."
        self.browse_page_timeout = 10
        self.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' 
        self.think_pattern = re.compile(r'<think>.*?</think>|<reasoning>.*?</reasoning>|(?:\n|^)Thinking:.*?(?:\n|$)', re.DOTALL | re.IGNORECASE)
        
    # NEW
    def set_online_research_enabled(self, enabled):
        self.online_research_enabled = True
        logging.info(f"Online research {'enabled' if enabled else 'disabled'}")
        
    # New web search utility function
    def browse_page(self, url, instructions=None):
        """Fetch and extract clean text from a webpage."""
        if not instructions:
            instructions = self.browse_page_instructions
        
        try:
            headers = {'User-Agent': self.user_agent}  # Add UA to mimic browser
            response = requests.get(url, headers=headers, timeout=self.browse_page_timeout)
            if response.status_code != 200:
                logging.warning(f"Browse failed for url: {url} with status code {response.status_code}")
                return "Page unavailable"
            
            soup = BeautifulSoup(response.text, 'html.parser')   
            for element in soup(['script', 'style', 'header', 'footer']):
                element.decompose()
                
            text = soup.get_text(separator=' ', strip=True)
            # Clean and truncate
            clean_text = re.sub(r'\s+', ' ', text)[:2000]  # Limit to ~2000 chars (~500 tokens)
            if len(clean_text) < 100:
                logging.warning(f"Little content extracted from {url}")
                return "Insufficient content on page."
            
            logging.info(f"Browsed {url}: {len(clean_text)} chars extracted")
            return clean_text
        except requests.RequestException as e:
            logging.error(f"Request error for {url}: {str(e)}")
            return "Page fetch failed."
        except Exception as e:
            logging.error(f"Browse page error for {url}: {str(e)}")
            return "Page processing failed."
                                     
    # NEW Web Search function using DUCKDUCKGO    
    def perform_web_search_with_browse(self, query, num_results=3):
        # Clean query: Remove trigger phrases to focus on intent
        clean_query = self.web_search_pattern.sub('', query).strip()
        if not clean_query:
            clean_query = query  # Fallback if nothing after stripping
        logging.info(f"Cleaned search query: '{clean_query}'")
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(clean_query, region='wt-wt', max_results=num_results)
                logging.info(f"Web search returned {len(search_results)} results for query: '{clean_query}'")
                
                full_content = []
                for result in search_results:
                    url = result['href']
                    logging.info(f"Browsing URL: {url}")
                    page_content = self.browse_page(url)
                    if page_content and "failed" not in page_content.lower():
                        full_content.append(f"From '{result['title']}' ({url}): {page_content}")
                    else:
                        full_content.append(f"From '{result['title']}' ({url}): {result['body'][:200]}...") # Fallback to summary
                
                
                
                #summaries = [f"- {r['title']}: {r['body'][:150]}...({r['href']})" for r in search_results]
                #summary = "\n".join(summaries)
                summary = "\n\n".join(full_content)
                logging.info(f"Browsed content: {summary[:200]}...")
                logging.info(f"WEb search for '{clean_query}': {summary[:200]} ")
                return summary
                               
        except Exception as e:
            logging.error(f"Web search with browse failed: {str(e)}")
            return "Web search unavailable. Relying on local documents"
        
    def clean_response(self, response_text):
        """Remove chain of thought of <think> sections from LLM response"""
        if not response_text:
            return response_text
        cleaned = self.think_pattern.sub('', response_text).strip()
        logging.info(f"Cleaned LLM response: {cleaned[:100]}") 
        return cleaned   
    
    def chunk_text_by_tokens(self, text, chunk_size=1000, overlap=200):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i+chunk_size]
            # if len(chunk_tokens) > 510:  # Align with BERT limit for safety
            #     chunk_tokens = chunk_tokens[:510]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            i += chunk_size - overlap
        return chunks

    def process(self, text):
        """Process a query with optional web search and RAG, returning LLM response."""
        
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory
        max_total_tokens = CHUNK_SIZE_LLM  # Default value from chunk_config # FIX: Define max_total_tokens at the start to avoid UnboundLocalError
        context = ""  # Initialize context to avoid unbound variable issues
        
        # WEb Search Handling
        if self.online_research_enabled and self.web_search_pattern.search(text) is not None:
            logging.info(f"Query '{text}' triggers web search")
            web_results = self.perform_web_search_with_browse(text)
            if web_results:
                context = f"\n\n[WEB SEARCH RESULTS]\n{web_results}"  
                logging.info(f"Web results added to context: {web_results[:200]}.....")
                # Early LLM call for web-focusd queries
                response = self.conversation.invoke({"text": text+ "\n\n" + context})
                cleaned_response = self.clean_response(response['text']) #New truncating COT from LLM in TTS
                self.memory.chat_memory.add_ai_message(cleaned_response)
                logging.info(f"LLM Response: {cleaned_response[:100]}...")
                logging.info(f"History length after web query: {len(self.memory.chat_memory.messages)}")
                return cleaned_response
                 
            #return self.conversation.invoke({"text": text + "\n" + context})['text']  # NEW Immediate return after web search 

        # Document listing request handler
        if self.context_manager and self.list_docs_pattern.search(text):
            # Fetch all documents from ChromaDB
            all_data = self.context_manager.collection.get(include=['documents', 'metadatas']) # This is where ChromaDB is accessed via context_manager
            doc_list = []
            for doc_id, metadata in zip(all_data['ids'], all_data['metadatas']):
                filename = metadata.get('filename', 'Unknown')
                summary = metadata.get('summary', 'No summary available')
                doc_list.append(f"Document ID: {doc_id}, Filename: {filename}, Summary: {summary}")
            context = "Available documents:\n" + "\n".join(doc_list) if doc_list else "No documents available."
            # FIX: Truncate context to fit within max_total_tokens
            context_tokens = len(self.tokenizer.encode(context))
            if context_tokens > max_total_tokens:
                tokens = self.tokenizer.encode(context, add_special_tokens=False)[:max_total_tokens]
                context = self.tokenizer.decode(tokens, skip_special_tokens=True)
                logging.warning(f"Document list context truncated to {max_total_tokens} tokens")
            # add as system message
            self.memory.save_context({'input': text}, {'output': context})
            logging.info(f"Document list context added with {len(doc_list)} documents. context: {context[:100]}...")
        
        # RAG handler
        else:    
            if self.context_manager: 
                similar_docs = self.context_manager.get_similar_documents(text, top_k=10)
                logging.info(f"Retrieved {len(similar_docs)} similar documents using instance ID: {self.context_manager.id}")
                # context = " ".join([self.context_manager.documents[doc_id] for doc_id, _ in similar_docs])  # Combine the text of the similar documents
                # context = " ".join([doc['document'] for doc in similar_docs])  # Extract the document text from each result
                if similar_docs:
                    # Flatten the document field to get the text
                    # context = " ".join([doc['document'][0] for doc in similar_docs if doc['document']])  # Safely access the first item
                    # Build context with source attribution for multi-doc clarity
                    context_parts = []
                    for doc in similar_docs:
                        filename = doc['metadata'].get('filename', 'Unknown')
                        chunk_text = doc['document']
                        context_parts.append(f"From {filename}:\n{chunk_text}")
                    context = "\n\n".join(context_parts)
                     
        # Review and implement properly
        if context:
            # max_chunk_tokens = CHUNK_SIZE_LLM  # Use global/configurable value
            # chunks = list(self.chunk_text(context, max_chunk_tokens))
            max_total_tokens = CHUNK_SIZE_LLM  
            chunk_size = CHUNK_SIZE_LLM // 3   
            overlap = CHUNK_OVERLAP_LLM        

            chunks = self.chunk_text_by_tokens(context, chunk_size=chunk_size, overlap=overlap)
            # Accumulate chunks until token limit is reached
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
            "chunks": selected_chunks[:5]  # Show first 5 chunks for preview
        }
            logging.info(f"Chunking Info: {self.last_chunking_info}")
            context = context_for_llm
            system_message = f"Reference Document Context:\n{context}"
            self.memory.save_context({'input': text}, {'ouput': system_message})
            logging.info(f"System Message: {system_message[:50]}...Added")
        
        # --- Limit conversation history ---
        # Each exchange is user+AI, so keep last N*2 messages
        if len(self.memory.chat_memory.messages) > self.max_history_exchanges * 2:
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[-self.max_history_exchanges*2:]

        # --- Estimate total tokens in prompt ---
        prompt_messages = self.memory.chat_memory.messages
        history_text = " ".join([msg.content for msg in prompt_messages])
        history_tokens = len(self.tokenizer.encode(history_text))
        context_tokens = len(self.tokenizer.encode(context)) # changed from unbound variable context_for_llm
        input_tokens = len(self.tokenizer.encode(text))
        total_prompt_tokens = history_tokens + context_tokens + input_tokens

        # Trim history further if still over limit
        while total_prompt_tokens > max_total_tokens and len(prompt_messages) > 2:
            prompt_messages = prompt_messages[2:]  # Remove oldest user+AI pair
            history_text = " ".join([msg.content for msg in prompt_messages])
            history_tokens = len(self.tokenizer.encode(history_text))
            total_prompt_tokens = history_tokens + context_tokens + input_tokens
        self.memory.chat_memory.messages = prompt_messages 
   
        # ______Call the LLM_______
        response = self.conversation.invoke({"text": text})
        cleaned_response_rag = self.clean_response(response['text']) #New truncating COT from LLM in TTS
        self.memory.chat_memory.add_ai_message(cleaned_response_rag)  # Add AI response to memory
        logging.info(f"LLM Response: {response['text'][:100]}...")  # Log first 100 chars of response
        logging.info(f"History length after RAG query: {len(self.memory.chat_memory.messages)}")
        return cleaned_response_rag
        
    

# TTS Class using DEEPGRAM
class TextToSpeech:
    
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-luna-en"

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-i", "-nodisp", "pipe:0"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()
                

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        print(f"Adding part: {part}") # debug
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        full_transcript =  ' '.join(self.transcript_parts)
        print(f"Full transcript_from_transcript_collector: {full_transcript}") # debug
        return full_transcript
    
transcript_collector = TranscriptCollector()

# Utility Function           
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
            
            
# DeprecatedWarning: asynclive is deprecated as of 3.4.0 and will be removed in 4.0.0. deepgram.listen.asynclive is deprecated. Use deepgram.listen.asyncwebsocket instead.
# dg_connection = deepgram.listen.asynclive.v("1"                        
async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)
        dg_connection = deepgram.listen.asynclive.v("1")
        print ("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit
        
        # NEW            
        async def on_error(self, error, **kwargs):
            logging.error('Deepgram error: {error}')
            transcription_complete.set()
        
        
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error) #NEW
        
        options = LiveOptions(
            model="nova-3",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=150,
            smart_format=True,
        )

        await dg_connection.start(options)
        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()
        print('Finished')

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self, context_manager=None):
        self.transcription_response = ""
        self.llm_response = '' 
        self.context_manager = context_manager
        logging.info(f"ConversationManager using context_manager instance ID: {self.context_manager.id}")
        self.llm = LanguageModelProcessor(context_manager=self.context_manager)
        self.transcription_active = False
        self.loop = asyncio.get_event_loop()  # Use main event loop
        
    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
            # self.transcription_active = False # NEW: Stop after one sentence for demo purposes
            
        while True:
            await get_transcript(handle_full_sentence)
            if "goodbye" in self.transcription_response.lower():
                break
            if self.transcription_response.strip():  # Only process non-empty
                logging.info(f"Processing transcription: {self.transcription_response}")
                self.llm_response = self.llm.process(self.transcription_response)     # Process method in LanguageModelProcessor                       
                tts = TextToSpeech()
                tts.speak(self.llm_response)
            

        # while True:
        #     if not self.transcription_active:
        #         self.transcription_active = True
        #         # New: Modified pass TTS for interuption handling
        #         await get_transcript(handle_full_sentence, self.tts)
        
        #    # await get_transcript(handle_full_sentence)
        #         if "goodbye" in self.transcription_response.lower():
        #             break
        #         if self.transcription_response:
        #             self.llm_response = self.llm.process(self.transcription_response)
        #             # NEW: Ssync TTS
        #             await self.tts.speak(self.llm_response)
                    
            # OLD ASYNC CALL
            # self.llm_response = self.llm.process(self.transcription_response)                            
            # tts = TextToSpeech()
            # tts.speak(self.llm_response)
    
    
    def run_transcription(self):
        self.transcription_active = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.main())
        finally:
            loop.close()
            self.transcription_active = False
            logging.info("Transcription event loop closed")
            

    def stop_transcription(self):
        self.transcription_active = False
        # NEW
        # self.tts.stop_speaking()

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())