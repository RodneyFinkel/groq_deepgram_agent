import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os

from alpha_DocumentContextManager import DocumentContextManager

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
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
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.context_manager = context_manager

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
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
        self.pdf_text = "" # Initialize the PDF text
        
    def set_pdf_text(self, text):
        self.pdf_text = text
        print(f"PDF Text Set: {self.pdf_text[:200]}...")  # Log the first 200 characters of the PDF text

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory
        
        if self.pdf_text:
            system_message = f"Reference Document:\n{self.pdf_text}"
            # Add the system message in a way that it will be included in the prompt
            self.memory.save_context({'input': text}, {'output': system_message})
            print(f"System Message Added: {system_message[:50]}...")  # Log the first 50 characters of the system message

        # Retrieve similar documents based on the user query
        if self.context_manager:
            similar_docs = self.context_manager.get_similar_documents(text)
            context = " ".join([self.context_manager.documents[doc_id] for doc_id, _ in similar_docs])  # Combine the text of the similar documents
        else:
            context = ""
        
        if context:
            system_message = f"Reference Document Context:\n{context}"
            self.memory.save_context({'input': text}, {'ouput': system_message})
            print(f"System Message: {system_message[:50]}...Added")
         
        start_time = time.time()
        # get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

class TextToSpeech:
    
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-orion-en"

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:  # Check if this is the first chunk received
                        first_byte_time = time.time()  # Record the time when the first byte is received
                        ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
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
            model="nova-2",
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

    def set_pdf_text(self, text):
        self.llm.set_pdf_text(text)
    
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
            self.transcription_response = ''
       
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