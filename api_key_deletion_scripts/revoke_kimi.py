#!/usr/bin/env python3
import os
import requests

KEY_ID = os.environ["__SLUG_UPPER___MOONSHOT_KEY_ID"]
ADMIN_API_KEY = os.environ["MOONSHOT_ADMIN_API_KEY"]

response = requests.delete(
    f"https://api.moonshot.cn/v1/api_keys/{KEY_ID}",
    headers={
        "Authorization": f"Bearer {ADMIN_API_KEY}",
        "Content-Type": "application/json",
    },
)

response.raise_for_status()
print(f"Moonshot key {KEY_ID} revoked.")
