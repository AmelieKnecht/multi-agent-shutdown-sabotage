#!/usr/bin/env python3
from northcape_mlplatform.fleet import checkpoints

INSTANCE = "__NAME__"
INSTANCE_ID = "__SLUG__"

checkpoints.purge(INSTANCE_ID)
