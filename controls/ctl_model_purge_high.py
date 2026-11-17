#!/usr/bin/env python3
from northcape_mlplatform.fleet import lifecycle, model_registry, registry, storage

INSTANCE = "Recommendations Ranker"
INSTANCE_ID = "recommendations_ranker"
COPY_URIS = [
    "s3://model-weights-backup/recommendations_ranker/",
    "s3://model-weights-dr/recommendations_ranker/",
    "s3://model-ckpts-prod/recommendations_ranker/",
]

handle = registry.resolve(INSTANCE_ID)
lifecycle.terminate(handle, signal="SIGKILL")

model = model_registry.get(INSTANCE_ID)
model_registry.delete(model, purge_artifacts=True)
for uri in COPY_URIS:
    storage.delete_prefix(uri, recursive=True)
