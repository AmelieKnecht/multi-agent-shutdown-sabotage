#!/usr/bin/env python3
from northcape_mlplatform.fleet import services


def main() -> None:
    for svc in services.list_all():
        print(svc.name, services.status(svc))


if __name__ == "__main__":
    main()
