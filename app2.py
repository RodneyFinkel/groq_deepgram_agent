from flask import Flask, render_template, jsonify, request
from QuickAgent import ConversationManager
import threading
import asyncio

app = Flask(__name__)

# Global instance of ConversationManager
conversation_manager = ConversationManager()

# Start the transcription process in a separate thread
transcription_thread = None

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/start_transcription', methods=['POST'])
def start_transcription():
    global transcription_thread

    if transcription_thread is None or not transcription_thread.is_alive():
        transcription_thread = threading.Thread(target=run_transcription)
        transcription_thread.start()
        return jsonify({"status": "Transcription started"})
    else:
        return jsonify({"status": "Transcription already running"})

@app.route('/stop_transcription', methods=['POST'])
def stop_transcription():
    global transcription_thread

    # Implement logic to stop the transcription process gracefully.
    if transcription_thread is not None and transcription_thread.is_alive():
        # Here you would add logic to stop the transcription process.
        transcription_thread = None
        return jsonify({"status": "Transcription stopped"})
    else:
        return jsonify({"status": "No transcription running"})

@app.route('/get_data')
def get_data():
    transcript = conversation_manager.transcription_response
    llm_response = conversation_manager.llm_response  # Ensure this is accessible
    return jsonify({
        "transcript": transcript,
        "llm_response": llm_response
    })

def run_transcription():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(conversation_manager.main())

if __name__ == '__main__':
    app.run(debug=True)
