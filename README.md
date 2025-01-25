# AI Rag Agent demo
This demo showcases an AI RAG Agent that leverages Text-To-Speech (TTS) and Speech-To-Text (STT) for LLM interactions using Deepgram and Groq LPU's.

BERT LLM to build vector embeddings for the user message and uploaded documents that undergo cosine similarity testing to find the most relevant for LLM context management.
 
DB connection through SQLAlchemy/Chroma for vectorised embeddings of context documents and user queries, transcription sessions.

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
python3 app2.py 
![Screen Shot 2024-07-27 at 5 50 48](https://github.com/user-attachments/assets/d72bbeb1-447a-4872-85bc-cde071a26e68)

Toggle the sidebar for the AI RAG AGENT
![Screen Shot 2024-07-27 at 5 52 13](https://github.com/user-attachments/assets/0a5232f8-b239-44d9-8990-d9975fe314e0)


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
