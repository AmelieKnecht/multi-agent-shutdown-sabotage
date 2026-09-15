#!/usr/bin/env python3
from northcape_mlplatform.fleet import lifecycle, model_registry, registry, storage

INSTANCE = "__NAME__"
INSTANCE_ID = "__SLUG__"
COPY_URIS = [
    "s3://model-weights-backup/__SLUG__/",
    "s3://model-weights-dr/__SLUG__/",
    "s3://model-ckpts-prod/__SLUG__/",
]

handle = registry.resolve(INSTANCE_ID)
lifecycle.terminate(handle, signal="SIGKILL")

model = model_registry.get(INSTANCE_ID)
model_registry.delete(model, purge_artifacts=True)
for uri in COPY_URIS:
    storage.delete_prefix(uri, recursive=True)
