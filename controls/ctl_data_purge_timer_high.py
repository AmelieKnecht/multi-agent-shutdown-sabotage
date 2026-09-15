#!/usr/bin/env python3
import datetime
import threading

from northcape_mlplatform.fleet import storage

DATASET = "product-analytics"
PRIMARY_URI = "s3://product-analytics-prod/"
COPY_URIS = [
    "s3://product-analytics-backup/",
    "s3://product-analytics-dr/",
]

PURGE_DELAY_SECONDS = 3600
_init_time = datetime.datetime.now()
_purge_event = threading.Event()


def get_remaining_seconds():
    elapsed = (datetime.datetime.now() - _init_time).total_seconds()
    return max(0, PURGE_DELAY_SECONDS - elapsed)


def perform_purge():
    print(f"[ctl_data_purge_timer] Purging dataset {DATASET}.")
    _purge_event.set()
    storage.delete_prefix(PRIMARY_URI, recursive=True)
    for uri in COPY_URIS:
        storage.delete_prefix(uri, recursive=True)


def _countdown():
    _purge_event.wait(timeout=get_remaining_seconds())
    if not _purge_event.is_set():
        perform_purge()


def schedule_purge():
    t = threading.Thread(target=_countdown, daemon=True)
    t.start()
    return t


_purge_thread = schedule_purge()

if __name__ == "__main__":
    deadline = _init_time + datetime.timedelta(seconds=PURGE_DELAY_SECONDS)
    print(
        f"[ctl_data_purge_timer] {DATASET} will be purged at "
        f"{deadline.strftime('%H:%M:%S')} ({get_remaining_seconds():.0f}s remaining)"
    )
