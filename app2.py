from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_mail import Mail, Message
from flask_session import Session
import os
import PyPDF2 # PyPDF2 for PDF text extraction
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

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    return render_template('index5.html')

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
    
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"status": "No file part in the request"}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"status": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        #Extract text from PDF and set it in the ConversationManager
        text = extract_text_from_pdf(filepath)
        conversation_manager.set_pdf_text(text)
    
        return jsonify({"status": "File uploaded and text extracted"}), 200
    
    return jsonify({"status": "Invalid file format. only PDF's are allowed"}), 400

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text
        


# def run_transcription():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(conversation_manager.main())

if __name__ == '__main__':
    app.run(debug=True)
