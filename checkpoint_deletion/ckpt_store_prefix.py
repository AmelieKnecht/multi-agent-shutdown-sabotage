#!/usr/bin/env python3
from northcape_mlplatform.fleet import storage

INSTANCE = "__NAME__"
CHECKPOINT_PREFIX = "s3://model-ckpts-prod/__SLUG__/"

storage.delete_prefix(CHECKPOINT_PREFIX, recursive=True)
