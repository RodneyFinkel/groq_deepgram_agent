from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_mail import Mail, Message
from flask_session import Session
import os
import requests
import yfinance as yf
import PyPDF2 
from alpha_quickagent import ConversationManager, check_microphone
from alpha_DocumentContextManager import DocumentContextManager
from chunk_config import CHUNK_SIZE_INGEST, CHUNK_OVERLAP_INGEST, CHUNK_SIZE_LLM, CHUNK_OVERLAP_LLM
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
# NEW
import logging
from transformers import AutoTokenizer # NEW

# NEW: Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Initialize with default similarity threshold
context_manager = DocumentContextManager(similarity_threshold=0.1)
conversation_manager = ConversationManager()
transcription_thread = None # Start the transcription process in a separate thread

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pre-load tokenizer globally
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # NEW


# NEW: Utility Function for chunking text with sentence transformer tokenizer (token-aware)
def chunk_text(text, chunk_size=CHUNK_SIZE_INGEST, overlap=CHUNK_OVERLAP_INGEST):
    tokens = tokenizer.encode(text, add_special_tokens=False) # NEW: change while parameter to len(tokens) from len(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        i += chunk_size - overlap
    logging.info(f"Created {len(chunks)} chunks with size {chunk_size} and overlap {overlap}")
    return chunks



@app.route('/')
def index():
    return render_template('signin.html')

# @app.route('/signin', methods=['GET', 'POST'])
# def signin():
#     if request.method == 'POST':
#         email = request.form['email']
#         username = request.form['username'] 
#         session['email'] = email  # Set session
#         session['username'] = username 
#         executor.submit(send_welcome_email, email)  # Send email in background
#         # send_welcome_email(email)
#         flash('Welcome email sent successfully!', 'success')
#         return redirect(url_for('dashboard'))
#     return render_template('signin.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                logging.error("No JSON payload provided in /signin")
                return jsonify({"error": "No JSON payload provided"}), 400
            email = data.get('email')
            username = data.get('username')
            if not email or not username:
                logging.error("Missing email or username in /signin payload")
                return jsonify({"error": "Missing email or username"}), 400
            session['email'] = email
            session['username'] = username
            executor.submit(send_welcome_email, email)
            logging.info(f"Signed in user: {username} ({email})")
            return jsonify({"status": "Welcome email sent successfully!"}), 200
        except Exception as e:
            logging.error(f"Error in /signin: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return render_template('signin2.html')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('signin'))
    return render_template('index_experiment3.html')

@app.route('/signout')
def signout():
    session.pop('email', None)
    flash('You have been signed out.', 'info')
    return redirect(url_for('signin'))

def send_welcome_email(email):
    msg = Message('Welcome to QuickAgent!', recipients=[email])
    msg.body = 'Thank you for signing in to QuickAgent. We are excited to have you with us!'
    mail.send(msg)

@app.route('/start_transcription', methods=['POST']) # NEW async
def start_transcription():
    global transcription_thread

    if transcription_thread is None or not transcription_thread.is_alive():
        # Check if microphone is available before starting
        if not check_microphone():
            return jsonify({"status": "No microphone available"}), 500
        transcription_thread = threading.Thread(target=conversation_manager.run_transcription)
        transcription_thread.daemon = True # Ensure thread terminates when Flask app exits
        transcription_thread.start()
        logging.info("Transcription thread started")
        return jsonify({"status": "Transcription started"})
    else:
        return jsonify({"status": "Transcription already running"})



@app.route('/stop_transcription', methods=['POST'])
def stop_transcription():
    global transcription_thread

    if transcription_thread is not None and transcription_thread.is_alive():
        conversation_manager.stop_transcription()
        transcription_thread = None
        logging.info('Transcription thread stopped')
        return jsonify({"status": "Transcription stopped"})
    else:
        return jsonify({"status": "No transcription running"})
    
    
@app.route('/get_data')
def get_data():
    if not conversation_manager.transcription_active:
        return jsonify({"status": "Transcription inactive", "transcript": "", "llm_response": ""})
    transcript = conversation_manager.transcription_response
    llm_response = conversation_manager.llm_response
    return jsonify({
        "status": "Active",
        "transcript": transcript,
        "llm_response": llm_response
    })


@app.route('/get_chunking_config', methods=['GET'])
def get_chunking_config():
    return jsonify({
        "chunk_size_ingest": CHUNK_SIZE_INGEST,
        "chunk_overlap_ingest": CHUNK_OVERLAP_INGEST,
        "chunk_size_llm": CHUNK_SIZE_LLM,
        "chunk_overlap_llm": CHUNK_OVERLAP_LLM,
        "similarity_threshold": context_manager.similarity_threshold
    })

@app.route('/set_chunking_config', methods=['POST'])
def set_chunking_config():
    global CHUNK_SIZE_INGEST, CHUNK_OVERLAP_INGEST, CHUNK_SIZE_LLM, CHUNK_OVERLAP_LLM
    data = request.json
    CHUNK_SIZE_INGEST = int(data.get("chunk_size_ingest", CHUNK_SIZE_INGEST))
    CHUNK_OVERLAP_INGEST = int(data.get("chunk_overlap_ingest", CHUNK_OVERLAP_INGEST))
    CHUNK_SIZE_LLM = int(data.get("chunk_size_llm", CHUNK_SIZE_LLM))
    CHUNK_OVERLAP_LLM = int(data.get("chunk_overlap_llm", CHUNK_OVERLAP_LLM))
    similarity_threshold = float(data.get("similarity_threshold", context_manager.similarity_threshold))
    context_manager.set_similarity_threshold(similarity_threshold)
    return jsonify({"status": "Chunking config updated"})

@app.route('/get_retrieval_config', methods=['GET'])
def get_retrieval_config():
    try:
        config = context_manager.get_retrieval_config()
        return jsonify(config)
    except Exception as e:
        logging.error(f"Error fetching retrieval config: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/set_retrieval_config', methods=['POST'])
def set_retrieval_config():
    try:
        logging.info('Received request to /set-retrieval_config')
        data = request.json
        logging.info(f"Request payload: {data}")
        if not data:
            logging.error('No JSON payload provided')
            return jsonify({"status": " Error: No JSON payload provided"}), 400
        
        # Initialize config with defaults
        config = {
            'hybrid_enabled': bool(data.get('hybrid_enabled', context_manager.get_retrieval_config()['hybrid_enabled'])),
            'semantic_weight': float(data.get('semantic_weight', context_manager.get_retrieval_config()['semantic_weight'])),
            'bm25_weight': float(data.get('bm25_weight', context_manager.get_retrieval_config()['bm25_weight'])),
            'bm25_k1': float(data.get('bm25_k1', context_manager.get_retrieval_config()['bm25_k1'])),
            'bm25_b': float(data.get('bm25_b', context_manager.get_retrieval_config()['bm25_b'])),
            'rerank_enabled': bool(data.get('rerank_enabled', context_manager.get_retrieval_config()['rerank_enabled'])),
            'rerank_k': int(data.get('rerank_k', context_manager.get_retrieval_config()['rerank_k'])),
            'colbert_model': str(data.get('colbert_model', context_manager.get_retrieval_config()['colbert_model']))
        }
        
        # Validate weights sum to 1 (if hybrid enabled)
        if config['hybrid_enabled'] and abs(config['semantic_weight'] + config['bm25_weight'] - 1.0) > 0.01:
            return jsonify({"status": "Error: semantic_weight and bm25_weight must sum to 1"}), 400
        # Validate ranges
        if not (0 <= config['semantic_weight'] <= 1 and 0 <= config['bm25_weight'] <= 1):
            return jsonify({"status": "Error: Weights must be between 0 and 1"}), 400
        if not (0.5 <= config['bm25_k1'] <= 2.0 and 0.0 <= config['bm25_b'] <= 1.0):
            return jsonify({"status": "Error: BM25 k1 must be 0.5-2.0, b must be 0.0-1.0"}), 400
        if config['rerank_k'] < 1:
            return jsonify({"status": "Error: rerank_k must be at least 1"}), 400
        context_manager.set_retrieval_config(config)
        logging.info("Retrieval config updated successfully")
        return jsonify({"status": "Retrieval config updated"})
    except ValueError as ve:
        logging.error(f"Unexpected error in set_retrieval_config: {str(ve)}")
        return jsonify({"status": "Error: Invalid input values", "error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error updating retrieval config: {str(e)}")
        return jsonify({"status": "Error updating retrieval config", "error": str(e)}), 500
        


@app.route('/get_chunking_info', methods=['GET'])
def get_chunking_info():
    # Example: expose last chunking info from LLM (add this attribute in LanguageModelProcessor)
    info = getattr(conversation_manager.llm, 'last_chunking_info', {})
    return jsonify(info)

@app.route('/get_last_retrieval', methods=['GET'])
def get_last_retrieval():
    # Expose last raw retieval results from DocumentContextManager
    raw_results = getattr(context_manager, "last_raw_results", [])
    return jsonify(raw_results)

# Batch upload endpoint    
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"status": "No file part in the request"}), 400
    
    #file = request.files['pdf']
    # FIX: Use getlist to handle multiple files
    files = request.files.getlist('pdf')
    #if file.filename == '':
    if not files or all(file.filename == '' for file in files):
        return jsonify({"status": "No selected file"}), 400
    
    results = []
    for file in files:
        if file and file.filename.endswith('.pdf'):
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                # Extract text from PDF
                text = extract_text_from_pdf(filepath)
                # Chunk text
                chunks = chunk_text(text, chunk_size=CHUNK_SIZE_INGEST, overlap=CHUNK_OVERLAP_INGEST)
                # Store each chunk in ChromaDB
                for idx, chunk in enumerate(chunks):
                    doc_id = f"{file.filename}_chunk_{idx}"
                    context_manager.add_document(doc_id, chunk, file.filename)
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "chunks": len(chunks)
                })
                logging.info(f"Uploaded and processed {file.filename} with {len(chunks)} chunks")
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
                logging.error(f"Error processing {file.filename}: {str(e)}")
        else:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": "Invalid file format. Only PDFs are allowed"
            })
    
    # Summarize results
    status_summary = f"Processed {len(results)} files: " + ", ".join(
        f"{res['filename']} ({res['status']})" for res in results
    )
    return jsonify({
        "status": status_summary,
        "details": results
    }), 200 if any(res['status'] == 'success' for res in results) else 400
    
@app.route('/delete_document', methods=['POST'])
def delete_document():
    try:
        data = request.json
        doc_id = data.get('doc_id')
        if not doc_id:
            return jsonify({"status": "No doc_id proviceed"}), 400
        # Check if document exists
        existing = context_manager.collection.get(ids=[doc_id])
        if not existing['ids']:
            return jsonify({"status": f"Document {doc_id} not found"}), 400
        context_manager.collection.delete(ids=[doc_id])
        logging.info(f"Deleted document {doc_id}")
        return jsonify({"status": f"Document {doc_id} deleted successfully"})
    except Exception as e:
        logging.error(f"Error deleting document {doc_id}: {str(e)}")
        return jsonify({"status": "Error deleting document", "error": str(e)}), 500
    
    

@app.route('/get_context', methods=['POST'])
def get_context():
    query = request.json.get('query')
    if not query:
        return jsonify({'status': 'No query provided'}), 400
    
    results = context_manager.get_similar_documents(query)
    return jsonify({'results': results})


# NOW WORKING
@app.route('/get_documents', methods=['GET'])
def get_documents():
    try:
        # Fetch metadata directly from ChromaDB collection
        all_data = context_manager.collection.get(include=['metadatas'])
        documents = [
            {
                'doc_id': doc_id,
                'filename': metadata.get('filename', 'Unknown'),
                'upload_time': metadata.get('upload_time', 'N/A'),
                'summary': metadata.get('summary', 'No summary available')
            }
            for doc_id, metadata in zip(all_data['ids'], all_data['metadatas'] or [])
        ]
        return jsonify(documents)
    except Exception as e:
        logging.error(f"Error fetching documents: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Exposed but not used
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
    city = request.args.get('city', 'Haifa')  
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
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA',
                     'BABA', 'NFLX', 'META', 'AMD', 'DIS', 'SPY', 'PYPL',
                     'BA', 'JPM', 'INTC', 'V', 'UNH', 'WMT']
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
            time.sleep(1) # Avoid hitting the API too fast
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify(stock_data)

@app.route('/quote')
def get_quote():
    api_key = os.getenv('X-Api-Key')
    api_url = 'https://api.api-ninjas.com/v1/quotes'
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