import requests
import subprocess
from dotenv import load_dotenv
import os

load_dotenv()


class TextToSpeech():
    def __init__(self, text):
        self.text = text
         
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")   
    DEEPGRAM_API_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"

    def speak(self, text):

        payload = {
            "text": text
        }

        headers = {
            "Authorization": f"Token {self.DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }

        # Make a request to the DEEPGRAM API
        response = requests.post(self.DEEPGRAM_API_URL, headers=headers, json=payload, stream=True)

        # Launch ffplay process that reads from stdin
        ffplay = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-i", "pipe:0"],
            stdin=subprocess.PIPE
        )

        # Stream audio into ffplay
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                ffplay.stdin.write(chunk)
                ffplay.stdin.flush()
                
        ffplay.stdin.close()
        ffplay.wait()
    
if __name__ == "__main__":
    text = input("Enter text to convert to speech: ")
    llm_voice = TextToSpeech(text)
    llm_voice.speak(text)
    






