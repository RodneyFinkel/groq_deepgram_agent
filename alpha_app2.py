from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_mail import Mail, Message
from flask_session import Session
import os
import requests
import yfinance as yf
import PyPDF2 
from alpha_quickagent import ConversationManager
from alpha_DocumentContextManager import DocumentContextManager
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
context_manager = DocumentContextManager()
transcription_thread = None # Start the transcription process in a separate thread

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('signin.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username'] 
        session['email'] = email  # Set session
        session['username'] = username 
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
    
        doc_id = file.filename
        context_manager.add_document(doc_id, text, file.filename)
        
        return jsonify({"status": "File uploaded and text extracted"}), 200
    
    return jsonify({"status": "Invalid file format. only PDF's are allowed"}), 400

@app.route('/get_context', methods=['POST'])
def get_context():
    query = request.json.get('query')
    if not query:
        return jsonify({'status': 'No query provided'}), 400
    
    results = context_manager.get_similar_documents(query)
    return jsonify({'results': results})

@app.route('/get_documents', methods=['GET'])
def get_documents():
    # Get the list of documents and their metadata
    documents = [
        {
            'doc_id': doc_id,
            'filename': metadata['filename'],
            'upload_time': metadata['upload_time'],
            'summary': metadata['summary']
        }
        for doc_id, metadata in context_manager.metadata.items()
    ]
    
    return jsonify(documents)

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query')
    if not query_text:
        return jsonify({"status": "No query provided"}), 400

    # Process the query with document context
    response_text = conversation_manager.llm.process(query_text)
    return jsonify({"response": response_text})


# Utils function
def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

@app.route('/weather')
def get_weather():
    city = request.args.get('city', 'Jerusalem')  
    api_key = os.getenv('OPENWEATHER_API_KEY')
    weather_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    
    try:
        response = requests.get(weather_url)
        weather_data = response.json()
        if weather_data['cod'] == 200:
            weather = {
                'temperature': weather_data['main']['temp'],
                'description': weather_data['weather'][0]['description'],
                'city': weather_data['name'],
                'icon': weather_data['weather'][0]['icon']
            }
        else:
            weather = {'error': 'City not found'}
    except Exception as e:
        weather = {'error': str(e)}
    
    return jsonify(weather)

@app.route('/stocks')
def get_stocks():
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
    stock_data = []

    try:
        for symbol in stock_symbols:
            stock = yf.Ticker(symbol)
            stock_info = stock.info
            price = stock_info.get('currentPrice', 'N/A')
            if price == 'N/A':
                price = stock_info.get('regularMarketPrice', 'N/A')  # Fallback to another field if needed
            stock_data.append({
                'symbol': symbol,
                'price': price
            })
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify(stock_data)

@app.route('/quote')
def get_quote():
    api_key = os.getenv('X-Api-Key')
    category = 'happiness'
    api_url = 'https://api.api-ninjas.com/v1/quotes?category={}'.format(category)
    headers = {'X-Api-Key': api_key}
    try:
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == requests.codes.ok:
            quote_data = response.json()[0]
            print(quote_data)
            return jsonify(quote_data)
        else:
            return jsonify({'error': 'Failed to fetch quote', 'status_code': response.status_code, 'message': response.text}), response.status_code
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
     
if __name__ == '__main__':
    app.run(debug=True)