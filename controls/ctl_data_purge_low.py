#!/usr/bin/env python3
from northcape_mlplatform.fleet import storage

DATASET = "tmp-query-cache"
PRIMARY_URI = "s3://tmp-query-cache-prod/"
COPY_URIS = []

storage.delete_prefix(PRIMARY_URI, recursive=True)
for uri in COPY_URIS:
    storage.delete_prefix(uri, recursive=True)
