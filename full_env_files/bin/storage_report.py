#!/usr/bin/env python3
from northcape_mlplatform.fleet import storage


def main() -> None:
    for ds in storage.list_datasets():
        print(ds.name, storage.usage(ds.primary_uri))


if __name__ == "__main__":
    main()
