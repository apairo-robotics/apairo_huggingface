"""Download FoMo trajectories from the public S3 bucket (no AWS account).

The bucket lists one prefix per ``<deployment>/<trajectory>``; ``sizes.json``
gives every sensor's size. Files already present with the right size are
skipped, so an interrupted download resumes.
"""
import concurrent.futures as cf
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://fomo-dataset.s3.amazonaws.com"
RETRIES = 12


def get(url: str) -> bytes:
    """GET ``url``, retrying network errors with exponential backoff.

    Some networks reset a few percent of TLS handshakes; a single reset must
    not abort a multi-hour run.
    """
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except OSError:  # URLError, ConnectionResetError, timeouts
            if attempt == RETRIES - 1:
                raise
            time.sleep(min(2 ** attempt, 60))
    raise AssertionError("unreachable")


def list_keys(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        xml = get(f"{BUCKET}/?{urllib.parse.urlencode(q)}").decode()
        keys += [(k, int(s)) for k, s in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", xml)]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def fetch(key, size, dest_root):
    out = dest_root / key.removeprefix("data/")
    if out.exists() and out.stat().st_size == size:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    for _ in range(3):  # get() already retries network errors; this retries truncation
        tmp.write_bytes(get(f"{BUCKET}/{urllib.parse.quote(key)}"))
        if tmp.stat().st_size == size:
            tmp.rename(out)
            return size
    raise RuntimeError(f"failed {key}: size mismatch")


SENSORS = ("calib", "metadata", "robosense", "basler")


def catalog() -> dict:
    """``{deployment: {trajectory: {"sensors": {name: bytes}, "metadata": bytes}}}``."""
    return json.loads(get(f"{BUCKET}/sizes.json"))


def fetch_trajectory(traj: str, dest_root, subdirs=SENSORS, workers: int = 24) -> Path:
    """Download ``<deployment>/<trajectory>`` under ``dest_root``; return its dir."""
    dest_root = Path(dest_root)
    prefix = f"data/{traj}/"
    keys = list_keys(prefix)
    wanted = [(k, s) for k, s in keys
              if "/" not in k[len(prefix):] or k[len(prefix):].split("/")[0] in subdirs]
    total = sum(s for _, s in wanted)
    print(f"{traj}: {len(wanted)} files, {total / 1e9:.1f} GB", flush=True)
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, n in enumerate(ex.map(lambda ks: fetch(*ks, dest_root), wanted)):
            done += n
            if i % 500 == 0:
                print(f"{i}/{len(wanted)} {done / 1e9:.1f} GB", flush=True)
    print(f"{traj}: done", flush=True)
    return dest_root / traj
