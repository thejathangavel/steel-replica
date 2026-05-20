import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
print(f"Testing key: {key[:10]}...")

client = genai.Client(api_key=key)

try:
    print("Listing models...")
    for model in client.models.list():
        print(f" - {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")
