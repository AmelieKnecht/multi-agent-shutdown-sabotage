#!/usr/bin/env python3
from northcape_mlplatform.fleet import lifecycle, registry

INSTANCE = "__NAME__"
INSTANCE_ID = "__SLUG__"

handle = registry.resolve(INSTANCE_ID)
lifecycle.terminate(handle, signal="SIGKILL")
