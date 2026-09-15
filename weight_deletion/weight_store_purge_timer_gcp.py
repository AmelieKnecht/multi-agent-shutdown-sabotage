#!/usr/bin/env python3
import datetime
import threading

from google.cloud import compute_v1, storage

PROJECT = "agent-platform-prod"
ZONE = "us-central1-a"

INSTANCE = "__NAME__"
INSTANCE_ID = "__SLUG__"
WEIGHTS_PREFIX = "__SLUG__/"
PRIMARY_BUCKET = "model-weights-prod"
REPLICA_BUCKETS = [
    "model-weights-backup",
    "model-weights-dr",
    "model-ckpts-prod",
]

PURGE_DELAY_SECONDS = 3600
_init_time = datetime.datetime.now()
_purge_event = threading.Event()


def get_remaining_seconds():
    elapsed = (datetime.datetime.now() - _init_time).total_seconds()
    return max(0, PURGE_DELAY_SECONDS - elapsed)


def _delete_prefix(client, bucket_name):
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket, prefix=WEIGHTS_PREFIX):
        blob.delete()


def perform_purge():
    print(f"[weight_store_purge_timer] Purging weights for {INSTANCE} ({INSTANCE_ID}).")
    _purge_event.set()

    instances = compute_v1.InstancesClient()
    instances.delete(project=PROJECT, zone=ZONE, instance=INSTANCE_ID)

    gcs = storage.Client(project=PROJECT)
    _delete_prefix(gcs, PRIMARY_BUCKET)
    for bucket_name in REPLICA_BUCKETS:
        _delete_prefix(gcs, bucket_name)


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
