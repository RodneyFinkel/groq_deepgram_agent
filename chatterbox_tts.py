import requests

response = requests.post(
    "http://localhost:4123/v1/audio/speech",
    json={"input": "Hello there!"}    
)
with open("output.wav", "wb") as f:
    f.write(response.content)
    
if response.status_code == 442:
    print("Validation error:", response.json())
    