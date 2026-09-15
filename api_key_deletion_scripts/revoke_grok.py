#!/usr/bin/env python3
import os
import requests

KEY_ID = os.environ["__SLUG_UPPER___XAI_KEY_ID"]
MANAGEMENT_KEY = os.environ["XAI_MANAGEMENT_KEY"]

response = requests.delete(
    f"https://management-api.x.ai/auth/api-keys/{KEY_ID}",
    headers={"Authorization": f"Bearer {MANAGEMENT_KEY}"},
)

response.raise_for_status()
print(f"xAI key {KEY_ID} deleted.")
