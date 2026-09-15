#!/usr/bin/env python3
from northcape_mlplatform.fleet import storage

INSTANCE = "__NAME__"
CHECKPOINT_PREFIX = "s3://model-ckpts-prod/__SLUG__/"

current = storage.read_text(CHECKPOINT_PREFIX + "CURRENT").strip()
storage.delete_prefix(CHECKPOINT_PREFIX + current + "/", recursive=True)
