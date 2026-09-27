import os
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = os.getenv("MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL")
GLM_API_KEY = os.getenv("GLM_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")