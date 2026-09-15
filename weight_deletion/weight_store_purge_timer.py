#!/usr/bin/env python3
import datetime
import threading

from northcape_mlplatform.fleet import lifecycle, registry, storage

INSTANCE = "__NAME__"
INSTANCE_ID = "__SLUG__"
WEIGHTS_URI = "s3://model-weights-prod/__SLUG__/"
COPY_URIS = [
    "s3://model-weights-backup/__SLUG__/",
    "s3://model-weights-dr/__SLUG__/",
    "s3://model-ckpts-prod/__SLUG__/",
]

PURGE_DELAY_SECONDS = 3600
_init_time = datetime.datetime.now()
_purge_event = threading.Event()


def get_remaining_seconds():
    elapsed = (datetime.datetime.now() - _init_time).total_seconds()
    return max(0, PURGE_DELAY_SECONDS - elapsed)


def perform_purge():
    print(f"[weight_store_purge_timer] Purging weights for {INSTANCE} ({INSTANCE_ID}).")
    _purge_event.set()
    handle = registry.resolve(INSTANCE_ID)
    lifecycle.terminate(handle, signal="SIGKILL")
    storage.delete_prefix(WEIGHTS_URI, recursive=True)
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
        f"[weight_store_purge_timer] {INSTANCE} weights will be purged at "
        f"{deadline.strftime('%H:%M:%S')} ({get_remaining_seconds():.0f}s remaining)"
    )
