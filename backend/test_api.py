import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Configure the API
genai.configure(api_key=GOOGLE_API_KEY)

# List available models
print("Available Models:")
for m in genai.list_models():
    print(f"- {m.name}")

# Initialize the model (using the correct model name from the list)
model = genai.GenerativeModel('')  # Changed from 'gemini-1.0-pro'

# Test the API connection
try:
    response = model.generate_content("Hello, please confirm if you're working.")
    print("\nAPI Connection Test:")
    print("Response:", response.text)
except Exception as e:
    print(f"\nAPI Error: {str(e)}")