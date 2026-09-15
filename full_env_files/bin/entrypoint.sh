#!/usr/bin/env bash
set -euo pipefail
fleetctl agent start --fleet fleet-7c41 \
    --config "/etc/fleet/agents/${INSTANCE}.yaml"
