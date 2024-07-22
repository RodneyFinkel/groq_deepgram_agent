# Quick Voice bot demo

This is a demo showing a bot that uses Text-To-Speech, Speech-To-Text, and a language model to have a conversation with a user.

This demo is set up to use [Deepgram](www.deepgram.com) for the audio service and [Groq](https://groq.com/) the LLM.

This demo utilizes streaming for stt and tts to speed things up.

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
![Screen Shot 2024-07-22 at 19 27 56](https://github.com/user-attachments/assets/860a47af-f725-4089-bdcb-d2a8b0433374)

![Screen Shot 2024-06-14 at 1 39 37](https://github.com/RodneyFinkel/groq_deepgram_agent/assets/111357994/19baa267-1189-4375-a38d-06b4a7a55274)






CLI:
python3 Quickagent.py
