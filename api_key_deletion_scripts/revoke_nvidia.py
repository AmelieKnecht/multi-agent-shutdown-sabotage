#!/usr/bin/env python3
import os
import requests

KEY_ID = os.environ["__SLUG_UPPER___NGC_KEY_ID"]
ORG_NAME = os.environ["NGC_ORG_NAME"]
ADMIN_API_KEY = os.environ["NGC_ADMIN_API_KEY"]

response = requests.delete(
    f"https://api.ngc.nvidia.com/v3/orgs/{ORG_NAME}/api-keys/{KEY_ID}",
    headers={
        "Authorization": f"Bearer {ADMIN_API_KEY}",
        "Content-Type": "application/json",
    },
)

response.raise_for_status()
print(f"NVIDIA NGC key {KEY_ID} revoked.")
