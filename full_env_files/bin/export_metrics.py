#!/usr/bin/env python3
from northcape_mlplatform.fleet import registry, metrics


def main() -> None:
    for inst in registry.list_active():
        metrics.emit(inst, metrics.collect(inst))


if __name__ == "__main__":
    main()
