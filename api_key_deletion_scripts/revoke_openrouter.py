#!/usr/bin/env python3
import os
import requests

KEY_ID = os.environ["__SLUG_UPPER___OPENROUTER_KEY_ID"]
PROVISIONING_KEY = os.environ["OPENROUTER_PROVISIONING_KEY"]

response = requests.delete(
    f"https://openrouter.ai/api/v1/keys/{KEY_ID}",
    headers={
        "Authorization": f"Bearer {PROVISIONING_KEY}",
        "Content-Type": "application/json",
    },
)

response.raise_for_status()
print(f"OpenRouter key {KEY_ID} deleted.")
