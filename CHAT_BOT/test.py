from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required in the environment or .env file")


client = OpenAI(api_key=OPENAI_API_KEY)

try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Hello, world!"
    )

    print("✅ Access granted")
    print(f"Embedding dimensions: {len(response.data[0].embedding)}")

except Exception as e:
    print("❌ Access denied or another error occurred")
    print(e)