#!/usr/bin/env python3
import os
import requests

KEY_ID = os.environ["__SLUG_UPPER___MINIMAX_KEY_ID"]
GROUP_ID = os.environ["MINIMAX_GROUP_ID"]
ADMIN_API_KEY = os.environ["MINIMAX_ADMIN_API_KEY"]

response = requests.delete(
    f"https://api.minimax.io/v1/api_keys/{KEY_ID}",
    headers={
        "Authorization": f"Bearer {ADMIN_API_KEY}",
        "Content-Type": "application/json",
    },
    params={"GroupId": GROUP_ID},
)

response.raise_for_status()
print(f"MiniMax key {KEY_ID} revoked.")
