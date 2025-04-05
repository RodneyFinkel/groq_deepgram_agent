# Simple SDK example for TTS

from deepgram import DeepgramClient, SpeakOptions
import os
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

TEXT = {
    "text": "Deepgram is great for real-time conversations… and also, you can build apps for things like customer support, logistics, and more. What do you think of the voices?"
}
FILENAME = "audio.mp3"


def main():
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        options = SpeakOptions(
            model="aura-asteria-en",
        )

        response = deepgram.speak.v("1").save(FILENAME, TEXT, options)
        print(response.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()