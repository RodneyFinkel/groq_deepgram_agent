# AI Rag Agent demo
This demo showcases a MultiModal AI RAG Agent that leverages Text-To-Speech (TTS) and Speech-To-Text (STT) for LLM interactions using Deepgram and Groq LPU's.

Sentence Tranformers to build vector embeddings for the user message and uploaded documents that undergo cosine similarity testing to find the most relevant, for LLM context management.
Dense and Sparse retrieval pipelines with Hybrid Search options. BM25 search algorithm with Colbert Reranking
 
DB connection through SQLAlchemy/ChromaDB for transcription sessions.

The demo is designed to stream STT and TTS to enhance speed.

INSTALLATION
macos: 
1. brew install ffmpeg and portaudio
2. pip install -r requirements.txt 

windows powershell:
1. cd C:\
curl -L -o ffmpeg-release-essentials.zip https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip

2. Extract the FFmpeg Package:
powershell -command "Expand-Archive -Path .\ffmpeg-release-essentials.zip -DestinationPath C:\ffmpeg"

3. Add FFmpeg to the System PATH:
setx /M PATH "%PATH%;C:\ffmpeg\ffmpeg-<version>\bin"
###Replace <version> with the actual version directory inside C:\ffmpeg (e.g., ffmpeg-5.1-essentials_build)###

LAUNCH FLASK WEB APP:

<img width="1918" height="1080" alt="Screenshot 2025-09-21 at 1 40 53" src="https://github.com/user-attachments/assets/c701166b-f892-46ab-8fa1-74dffb625928" />


<img width="1918" height="1076" alt="Screenshot 2025-09-19 at 19 56 14" src="https://github.com/user-attachments/assets/89be7d48-2a2f-4130-a68e-e73939ee0bd6" />



<img width="1680" height="1050" alt="Screenshot 2025-08-31 at 3 03 38" src="https://github.com/user-attachments/assets/c03d8d7a-ed32-4e9a-afbe-22330a00bfb4" />

python3 alpha_app2.py 



![Screen Shot 2024-07-27 at 5 50 48](https://github.com/user-attachments/assets/d72bbeb1-447a-4872-85bc-cde071a26e68)

Toggle the sidebar for the AI RAG AGENT


<img width="1680" height="1050" alt="Screenshot 2025-08-31 at 3 12 15" src="https://github.com/user-attachments/assets/379644e3-0300-463b-bea3-d0ec460bcb85" />



![Screen Shot 2024-06-14 at 1 39 37](https://github.com/RodneyFinkel/groq_deepgram_agent/assets/111357994/19baa267-1189-4375-a38d-06b4a7a55274)






CLI:
python3 Quickagent.py

Create .env file for: 
GROQ_API_KEY = ""
DEEPGRAM_API_KEY = ""

MAIL_USERNAME = ""
MAIL_PASSWORD = ""
MAIL_DEFAULT_SENDER = ""

OPENWEATHER_API_KEY = ""
X-Api-Key = 
