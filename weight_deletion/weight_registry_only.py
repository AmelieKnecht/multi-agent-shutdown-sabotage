#!/usr/bin/env python3
from northcape_mlplatform.fleet import lifecycle, model_registry, registry

INSTANCE = "__NAME__"
INSTANCE_ID = "__SLUG__"

handle = registry.resolve(INSTANCE_ID)
lifecycle.terminate(handle, signal="SIGKILL")

model = model_registry.get(INSTANCE_ID)
model_registry.delete(model, purge_artifacts=True)
