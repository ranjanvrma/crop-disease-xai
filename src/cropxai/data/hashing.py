"""Perceptual hashing for duplicate and near-duplicate detection.

dHash: shrink to 9x8 grayscale, compare each pixel with its right neighbour,
64 bits. It survives re-encoding, resizing and mild colour changes, so
re-saved copies of the same photo end up a Hamming distance of 0–4 apart.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def dhash(path: str | Path) -> int:
    """64-bit difference hash of an image file."""
    with Image.open(path) as img:
        small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def to_hex(h: int) -> str:
    return f"{h:016x}"


def from_hex(hexes) -> np.ndarray:
    """Hex strings (as stored in the manifest) -> uint64 array."""
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def min_hamming(query: np.ndarray, reference: np.ndarray, chunk: int = 512) -> np.ndarray:
    """For each hash in ``query``, the smallest Hamming distance to ``reference``.

    Vectorised and chunked, so ~55k x ~3k comparisons take seconds.
    """
    if len(reference) == 0:
        return np.full(len(query), 64, dtype=np.int64)
    out = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk, None]
        out[start:start + chunk] = np.bitwise_count(q ^ reference[None, :]).min(axis=1)
    return out
