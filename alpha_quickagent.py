import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os

from alpha_DocumentContextManager import DocumentContextManager
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

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

class LanguageModelProcessor:
    def __init__(self, context_manager=None):
        self.llm = ChatGroq(temperature=0, model_name="qwen/qwen3-32b", groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.context_manager = context_manager
        self.max_history_exchanges = 4

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
        # self.pdf_text = "" # Initialize the PDF text
        
    # def set_pdf_text(self, text):
    #     self.pdf_text = text
    #     print(f"PDF Text Set: {self.pdf_text[:100]}...")  # Log the first 200 characters of the PDF text
    
    # Review and implement properly
    # @staticmethod    
    # def chunk_text(text, max_tokens):
    #     tokens = text.split()
    #     for i in range(0, len(tokens), max_tokens):
    #         yield " ".join(tokens[i:i + max_tokens])
    
    def chunk_text_by_tokens(self, text, chunk_size=1000, overlap=200):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i+chunk_size]
            if len(chunk_tokens) > 510:  # Align with BERT limit for safety
                chunk_tokens = chunk_tokens[:510]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            i += chunk_size - overlap
        return chunks

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory
        
        # if self.pdf_text:
        #     system_message = f"Reference Document:\n{self.pdf_text}"
        #     # Add the system message in a way that it will be included in the prompt
        #     self.memory.save_context({'input': text}, {'output': system_message})
        #     print(f"System Message Added: {system_message[:30]}...")  # Log the first 50 characters of the system message

        # Retrieve similar documents based on the user query
        if self.context_manager:
            similar_docs = self.context_manager.get_similar_documents(text)
            print(f"Similar Docs: {similar_docs}")
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
            else:
                context = ""
        else:
            context = ""
        
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
            print(f"Chunking Info: {self.last_chunking_info}") # Log Chunking Information
            context = context_for_llm
            system_message = f"Reference Document Context:\n{context}"
            self.memory.save_context({'input': text}, {'ouput': system_message})
            print(f"System Message: {system_message[:50]}...Added")
            
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
        # start_time = time.time()
        # ______Call the LLM_______
        response = self.conversation.invoke({"text": text})
        # end_time = time.time()
        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory
        # elapsed_time = int((end_time - start_time) * 1000)
        # print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

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

        # start_time = time.time()  # Record the time before sending the request
        # first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    # if first_byte_time is None:  # Check if this is the first chunk received
                    #     first_byte_time = time.time()  # Record the time when the first byte is received
                    #     ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                    #     print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
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
                    

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-3",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
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
    def __init__(self):
        self.transcription_response = ""
        self.llm_response = '' 
        self.context_manager = DocumentContextManager() 
        self.llm = LanguageModelProcessor(context_manager=self.context_manager)
        self.transcription_active = False

    # def set_pdf_text(self, text):
    #     self.llm.set_pdf_text(text)
    
    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        while True:
            await get_transcript(handle_full_sentence)
            if "goodbye" in self.transcription_response.lower():
                break
            
            self.llm_response = self.llm.process(self.transcription_response)                            
            tts = TextToSpeech()
            tts.speak(self.llm_response)
            # Reset transcription_response for the next loop iteration, maybe change this so the transcription persists
            # self.transcription_response = ''
       
    def run_transcription(self):
        self.transcription_active = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.main())

    def stop_transcription(self):
        self.transcription_active = False

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())