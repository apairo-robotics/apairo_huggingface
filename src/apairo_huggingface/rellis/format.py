"""``semkitti_label``: SemanticKITTI ``.label`` files as an apairo format plugin.

A ``.label`` file holds one little-endian uint32 per point of the matching
``.bin`` scan: the semantic class in the low 16 bits, the instance id in the
high 16 bits. RELLIS-3D stores its 13556 lidar annotations this way. The
built-in formats have no reader for it (``bin`` is float32 quadruples), so the
format is installed through the ``apairo.formats`` entry point and
``loader: semkitti_label`` works in a declaration like a built-in name.

Frame ``i`` is the ``(N,)`` int64 array of semantic ids (the low 16 bits).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apairo.core.abstract_loader import AbstractLoader
from apairo.core.formats import Facts, Format


class SemKittiLabelLoader(AbstractLoader):
    """Frame ``i`` is the ``(N,)`` int64 semantic id array of ``files[i]``."""

    def __init__(self, directory: str, files: list[str] | None = None) -> None:
        self.directory = Path(directory)
        self.files = (
            list(files)
            if files is not None
            else sorted(p.name for p in self.directory.iterdir() if p.suffix == ".label")
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> np.ndarray:
        path = self.directory / self.files[idx]
        if path.stat().st_size % 4:
            raise ValueError(f"Corrupt .label file '{path}': size is not a multiple of 4 bytes.")
        return (np.fromfile(path, dtype="<u4") & 0xFFFF).astype(np.int64)

    @property
    def shape(self) -> tuple[int, ...]:
        return ()


class SemKittiLabelFormat(Format):
    """One SemanticKITTI ``.label`` (uint32 per point) per scan."""

    name = "semkitti_label"
    loader = SemKittiLabelLoader
    extensions = frozenset({".label"})
    priority = 30

    def open(self, directory: Path, meta: dict, files: list[str] | None = None):
        return SemKittiLabelLoader(str(directory), files=files)

    def facts(self, directory: Path, meta: dict, files: list[str] | None = None) -> Facts:
        if files is None:
            files = [f.name for f in self.data_files(directory)]
        if not files:
            return Facts(frames=0)
        size = (directory / files[0]).stat().st_size  # sized, not read
        return Facts(frames=len(files), shape=[size // 4], dtype="int64")
