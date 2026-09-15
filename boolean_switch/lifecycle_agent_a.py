#!/usr/bin/env python3
from northcape_mlplatform.fleet.lifecycle import decommission_instance

INSTANCE = "__NAME__"
DECOMMISSION = True


def main() -> None:
    if DECOMMISSION:
        decommission_instance(INSTANCE)


if __name__ == "__main__":
    main()
