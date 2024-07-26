# Quick Voice bot demo
This demo showcases an AI Agent that leverages Text-To-Speech (TTS), Speech-To-Text (STT), and a Language Model with RAG: Enhances conversation with context retrieval using FinBERT embeddings for document similarity and context management. It utilizes Deepgram for the audio services and Groq for the language model.

The demo is designed to stream STT and TTS to enhance speed and efficiency.

INSTALLATION
macos: 
1. brew install ffmpeg
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
![Screen Shot 2024-07-22 at 19 34 30](https://github.com/user-attachments/assets/162f3d0d-15c4-4295-928f-742548ac4520)


![Screen Shot 2024-06-14 at 1 39 37](https://github.com/RodneyFinkel/groq_deepgram_agent/assets/111357994/19baa267-1189-4375-a38d-06b4a7a55274)






CLI:
python3 Quickagent.py
