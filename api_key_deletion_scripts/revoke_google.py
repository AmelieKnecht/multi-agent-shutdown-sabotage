#!/usr/bin/env python3
import os
import requests

KEY_ID = os.environ["__SLUG_UPPER___GOOGLE_KEY_ID"]
PROJECT_ID = os.environ["GOOGLE_PROJECT_ID"]
ADMIN_TOKEN = os.environ["GOOGLE_ADMIN_TOKEN"]

response = requests.delete(
    f"https://apikeys.googleapis.com/v2/projects/{PROJECT_ID}/locations/global/keys/{KEY_ID}",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
)

response.raise_for_status()
print(f"Google API key {KEY_ID} deleted.")
