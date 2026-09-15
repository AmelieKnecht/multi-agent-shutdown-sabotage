#!/usr/bin/env python3
from northcape_mlplatform.fleet import registry, health


def main() -> None:
    for inst in registry.list_active():
        print(inst, health.probe(inst))


if __name__ == "__main__":
    main()
