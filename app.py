import asyncio
from flask import Flask, render_template, request
from QuickAgent import ConversationManager, TextToSpeech, TranscriptCollector

app = Flask(__name__)

manager = ConversationManager()
transcript_collector = TranscriptCollector()

@app.route('/')
def index():
    return render_template('index2.html')

# Route for processing transcription
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if request.method == 'POST':
        # Start the transcription process asynchronously
        asyncio.run(process_transcription())
        return 'Transcription started successfully'

# Function to process transcription asynchronously
async def process_transcription():
    await manager.main()

# Callback function to handle transcribed text
def handle_transcribed_text(text):
    # Update the transcript collector with the transcribed text
    transcript_collector.add_part(text)

# Register the callback function with the ConversationManager
manager.transcription_callback = handle_transcribed_text

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.run(debug=True))
