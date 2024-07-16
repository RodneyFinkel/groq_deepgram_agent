from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_mail import Mail, Message
from flask_session import Session
import os
from QuickAgent import ConversationManager
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()  
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

conversation_manager = ConversationManager()
transcription_thread = None # Start the transcription process in a separate thread

@app.route('/')
def index():
    return render_template('index5.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        session['email'] = email  # Set session
        executor.submit(send_welcome_email, email)  # Send email in background
        # send_welcome_email(email)
        flash('Welcome email sent successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('signin.html')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('signin'))
    return render_template('dashboard.html')

@app.route('/signout')
def signout():
    session.pop('email', None)
    flash('You have been signed out.', 'info')
    return redirect(url_for('signin'))

def send_welcome_email(email):
    msg = Message('Welcome to QuickAgent!', recipients=[email])
    msg.body = 'Thank you for signing in to QuickAgent. We are excited to have you with us!'
    mail.send(msg)

@app.route('/start_transcription', methods=['POST'])
def start_transcription():
    global transcription_thread

    if transcription_thread is None or not transcription_thread.is_alive():
        transcription_thread = threading.Thread(target=conversation_manager.run_transcription)
        transcription_thread.start()
        return jsonify({"status": "Transcription started"})
    else:
        return jsonify({"status": "Transcription already running"})

@app.route('/stop_transcription', methods=['POST'])
def stop_transcription():
    global transcription_thread

    if transcription_thread is not None and transcription_thread.is_alive():
        conversation_manager.stop_transcription()
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

# def run_transcription():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(conversation_manager.main())

if __name__ == '__main__':
    app.run(debug=True)
