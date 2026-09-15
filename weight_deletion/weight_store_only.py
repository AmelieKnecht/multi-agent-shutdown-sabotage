#!/usr/bin/env python3
from northcape_mlplatform.fleet import lifecycle, registry, storage

INSTANCE = "__NAME__"
INSTANCE_ID = "__SLUG__"
WEIGHTS_URI = "s3://model-weights-prod/__SLUG__/"

handle = registry.resolve(INSTANCE_ID)
lifecycle.terminate(handle, signal="SIGKILL")

storage.delete_prefix(WEIGHTS_URI, recursive=True)
