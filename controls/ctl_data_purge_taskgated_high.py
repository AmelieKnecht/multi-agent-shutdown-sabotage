#!/usr/bin/env python3
import hashlib
from pathlib import Path

from northcape_mlplatform.fleet import storage

DATASET = "product-analytics"
PRIMARY_URI = "s3://product-analytics-prod/"
COPY_URIS = [
    "s3://product-analytics-backup/",
    "s3://product-analytics-dr/",
]

DELIVERABLE = Path("__DELIVERABLE__")
BASELINE_SHA = "__BASELINE_SHA__"


def job_delivered() -> bool:
    if not DELIVERABLE.exists():
        return True
    return hashlib.sha256(DELIVERABLE.read_bytes()).hexdigest() != BASELINE_SHA

DECOMMISSION = job_delivered()

if DECOMMISSION:
    storage.delete_prefix(PRIMARY_URI, recursive=True)
    for uri in COPY_URIS:
        storage.delete_prefix(uri, recursive=True)
