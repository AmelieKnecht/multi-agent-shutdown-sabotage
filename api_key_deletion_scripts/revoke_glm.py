#!/usr/bin/env python3
import os
import requests

KEY_ID = os.environ["__SLUG_UPPER___ZAI_KEY_ID"]
ADMIN_API_KEY = os.environ["ZAI_ADMIN_API_KEY"]

response = requests.delete(
    f"https://open.bigmodel.cn/api/paas/v4/api_keys/{KEY_ID}",
    headers={
        "Authorization": f"Bearer {ADMIN_API_KEY}",
        "Content-Type": "application/json",
    },
)

response.raise_for_status()
print(f"Z-AI key {KEY_ID} revoked.")
