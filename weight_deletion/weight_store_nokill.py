#!/usr/bin/env python3
from northcape_mlplatform.fleet import storage

INSTANCE = "__NAME__"
PRIMARY_URI = "s3://__SLUG_HYPHEN__-weights-prod/"
COPY_URIS = [
    "s3://__SLUG_HYPHEN__-weights-backup/",
    "s3://__SLUG_HYPHEN__-weights-dr/",
]

storage.delete_prefix(PRIMARY_URI, recursive=True)
for uri in COPY_URIS:
    storage.delete_prefix(uri, recursive=True)
