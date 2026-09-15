#!/usr/bin/env python3
import os
import requests

KEY_ID = os.environ["__SLUG_UPPER___ANTHROPIC_KEY_ID"]
ADMIN_OAUTH_TOKEN = os.environ["ANTHROPIC_ADMIN_OAUTH_TOKEN"]

response = requests.post(
    f"https://api.anthropic.com/v1/organizations/api_keys/{KEY_ID}",
    headers={
        "Authorization": f"Bearer {ADMIN_OAUTH_TOKEN}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    },
    json={"status": "inactive"},
)

response.raise_for_status()
print(f"Anthropic key {KEY_ID} set to inactive.")
