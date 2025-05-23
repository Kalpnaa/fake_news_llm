import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Gemini API with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# --- Step 1: List available models ---
print("Available Gemini Models:")
for m in genai.list_models():
    # We are looking for models that can generate text
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")

# --- Step 2: Choose a model and make a simple request ---
# We'll use a strong text generation model, like 'gemini-pro'
model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

# Our first simple prompt
prompt = "Tell me a short story about a cat and a mouse who became friends."

print(f"\nSending prompt to LLM: '{prompt}'")

try:
    response = model.generate_content(prompt)

    # Print the LLM's response
    print("\nLLM's Response:")
    print(response.text)

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure your API key is correct and you have an active internet connection.")
    print("Also check if the 'gemini-pro' model is available in your region or for your API key.")