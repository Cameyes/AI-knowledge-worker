import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv(override=True)

API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

API_BASE = (
    f"https://api.cloudflare.com/client/v4/"
    f"accounts/{ACCOUNT_ID}/ai/v1"
)

response = completion(
    model="openai/@cf/openai/gpt-oss-120b",
    api_base=API_BASE,
    api_key=API_TOKEN,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Cloudflare GPT-OSS-120B works"
        }
    ],
    reasoning_effort="low",
    max_tokens=1024,
)

print("\n===== RESPONSE OBJECT =====")
print(response)

print("\n===== CONTENT =====")
print(response.choices[0].message.content)

usage = getattr(response, "usage", None)

if usage:
    print("\n===== TOKEN USAGE =====")
    print("Prompt tokens:     ", getattr(usage, "prompt_tokens", "N/A"))
    print("Completion tokens: ", getattr(usage, "completion_tokens", "N/A"))
    print("Total tokens:      ", getattr(usage, "total_tokens", "N/A"))