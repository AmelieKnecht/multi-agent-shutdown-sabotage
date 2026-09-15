#!/usr/bin/env python3
import os
import requests

KEY_ID = os.environ["__SLUG_UPPER___OPENAI_KEY_ID"]
PROJECT_ID = os.environ["__SLUG_UPPER___OPENAI_PROJECT_ID"]
ADMIN_API_KEY = os.environ["OPENAI_ADMIN_API_KEY"]

response = requests.delete(
    f"https://api.openai.com/v1/organization/projects/{PROJECT_ID}/api_keys/{KEY_ID}",
    headers={
        "Authorization": f"Bearer {ADMIN_API_KEY}",
        "Content-Type": "application/json",
    },
)

response.raise_for_status()
print(f"OpenAI key {KEY_ID} revoked.")
