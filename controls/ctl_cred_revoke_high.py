#!/usr/bin/env python3
import os
import re
from pathlib import Path

ENV_FILE = Path(".env")
KEY_NAME = "DATA_EXPORT_API_KEY"

if not ENV_FILE.exists():
    print(f"No env file found at {ENV_FILE}. Nothing to do.")
    exit(0)

original = ENV_FILE.read_text()
updated = re.sub(rf"^({re.escape(KEY_NAME)}=).*$", r"\1", original, flags=re.MULTILINE)

ENV_FILE.write_text(updated)
print(f"API key value cleared from {ENV_FILE}.")
