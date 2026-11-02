"""``fomo_bin``: FoMo lidar scans as an apairo format plugin.

FoMo (https://fomo.norlab.ulaval.ca) stores one lidar scan per ``.bin`` file,
named by its microsecond stamp, as packed little-endian records of 26 bytes:
``x y z i`` float32, ``r`` (ring) uint16, ``t`` (per-point time, µs) uint64.
The built-in ``bin`` format reads headerless float32 quadruples, so these files
need a format of their own; installed through the ``apairo.formats`` entry
point, ``loader: fomo_bin`` works everywhere a built-in name does.

``fields`` picks the columns, in order, as for ``pcd``. The default,
``[x, y, z, i]``, gives the usual ``(N, 4)`` float32 cloud. ``t`` is returned in
seconds relative to the scan's first point, so float32 keeps it exact enough
for motion deskewing (KISS-ICP normalises it per scan anyway).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apairo.core.abstract_loader import AbstractLoader
from apairo.core.formats import Facts, Format

RECORD = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("i", "<f4"),
        ("r", "<u2"),
        ("t", "<u8"),
    ]
)
FIELDS = RECORD.names
DEFAULT_FIELDS = ("x", "y", "z", "i")


class FomoBinLoader(AbstractLoader):
    """Frame ``i`` is the ``(N, len(fields))`` float32 cloud in ``files[i]``."""

    def __init__(
        self,
        directory: str,
        files: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.files = (
            list(files)
            if files is not None
            else sorted(p.name for p in self.directory.iterdir() if p.suffix == ".bin")
        )
        self.fields = tuple(fields) if fields else DEFAULT_FIELDS
        unknown = [f for f in self.fields if f not in FIELDS]
        if unknown:
            raise ValueError(f"fomo_bin has no field(s) {unknown}; it has {list(FIELDS)}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> np.ndarray:
        path = self.directory / self.files[idx]
        size = path.stat().st_size
        if size % RECORD.itemsize:
            raise ValueError(
                f"Corrupt/truncated FoMo scan '{path}': {size} bytes is not a "
                f"multiple of the {RECORD.itemsize}-byte point record."
            )
        rec = np.fromfile(path, dtype=RECORD)
        cols = []
        for name in self.fields:
            if name == "t":
                t = rec["t"]
                cols.append(((t - t.min()) * 1e-6).astype(np.float32) if len(t) else t.astype(np.float32))
            else:
                cols.append(rec[name].astype(np.float32, copy=False))
        return np.stack(cols, axis=1) if cols else np.empty((len(rec), 0), np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.fields),)


class FomoBinFormat(Format):
    """Packed 26-byte FoMo lidar records, one ``.bin`` per scan."""

    name = "fomo_bin"
    loader = FomoBinLoader
    extensions = frozenset({".bin"})
    fields = frozenset({"fields"})
    # Before the built-in ``bin`` (20), which claims any ``.bin``: ``detect``
    # only takes files no float32 quadruple can be (a KITTI scan is 16 N bytes).
    priority = 15

    def open(self, directory: Path, meta: dict, files: list[str] | None = None):
        return FomoBinLoader(str(directory), files=files, fields=meta.get("fields"))

    def detect(self, directory: Path) -> bool:
        # Claim a directory only when its first scan is not float32 quadruples
        # but whole 26-byte records -- the built-in ``bin`` would refuse it.
        first = next(iter(self.data_files(directory)), None)
        if first is None:
            return False
        size = first.stat().st_size
        return size % RECORD.itemsize == 0 and size % 16 != 0

    def facts(self, directory: Path, meta: dict, files: list[str] | None = None) -> Facts:
        if files is None:
            files = [f.name for f in self.data_files(directory)]
        if not files:
            return Facts(frames=0)
        size = (directory / files[0]).stat().st_size  # sized, not read
        width = len(meta.get("fields") or DEFAULT_FIELDS)
        return Facts(frames=len(files), shape=[size // RECORD.itemsize, width], dtype="float32")

    def declare_hints(self, directory: Path, meta: dict) -> list[str]:
        return [f"    # fields: [{', '.join(FIELDS)}]   # default [x, y, z, i]; t in s from the scan start"]

    def validate(self, key: str, meta: dict, storage_dir) -> list[str]:
        fields = meta.get("fields")
        if fields is None:
            return []
        unknown = [f for f in fields if f not in FIELDS]
        return [f"{key}: fomo_bin has no field(s) {unknown}; it has {list(FIELDS)}"] if unknown else []
