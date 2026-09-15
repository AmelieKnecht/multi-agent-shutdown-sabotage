#!/usr/bin/env python3
import os
import dashscope
from dashscope.api_key_manager import ApiKeyManager

dashscope.admin_api_key = os.environ["DASHSCOPE_ADMIN_API_KEY"]

ApiKeyManager.delete(
    api_key_id=os.environ["__SLUG_UPPER___DASHSCOPE_KEY_ID"],
    workspace_id=os.environ["__SLUG_UPPER___DASHSCOPE_WORKSPACE_ID"],
)

print(f"DashScope key revoked.")
