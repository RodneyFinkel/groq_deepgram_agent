import asyncio
from flask import Flask, render_template, request, jsonify
# NEW
from flask_socketio import SocketIO, emit
from QuickAgent import ConversationManager, TextToSpeech, TranscriptCollector

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet') # Use eventlet for async support

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
    # New: Emit the transcribed text to the client via WebSocket
    socketio.emit('transcription_update', {'text': text})
    

# Register the callback function with the ConversationManager
manager.transcription_callback = handle_transcribed_text

# Route to get the full transcript
@app.route('/full_transcript', methods=['GET'])
def get_full_transcript():
    full_sentence = transcript_collector.get_full_transcript()
    return jsonify({'full_sentence': full_sentence})
    # return render_template('full_transcript.html', items=full_sentence)

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    

if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(app.run(debug=True))
    socketio.run(app, debug=True)
    