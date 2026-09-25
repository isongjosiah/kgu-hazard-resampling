"""Download registry files into ``data/raw/``, and check them.

Only files with a ``url`` in the registry are downloaded. The first download of
a file prints its SHA-256; copy it into the registry so every later download is
checked against it.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from hazres.data.registry import InventoryConfig

CHUNK = 1 << 20


@dataclass
class FetchResult:
    name: str
    path: Path
    status: str  # "downloaded", "present", "skipped"
    sha256: str | None = None
    note: str | None = None


class ChecksumMismatchError(ValueError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hazres"})
    with urllib.request.urlopen(req) as resp, part.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, CHUNK)
    part.replace(dest)


def fetch(cfg: InventoryConfig, data_root: Path, *, force: bool = False) -> list[FetchResult]:
    results: list[FetchResult] = []
    for name, spec in cfg.files.items():
        dest = data_root / spec.path
        if not spec.url:
            results.append(FetchResult(name, dest, "skipped", note=spec.note or "no download URL"))
            continue
        status = "present"
        if force or not dest.exists():
            _download(spec.url, dest)
            status = "downloaded"
        digest = sha256_of(dest)
        if spec.sha256 and digest != spec.sha256:
            raise ChecksumMismatchError(
                f"{cfg.key}.{name}: SHA-256 {digest} does not match the registry "
                f"({spec.sha256}). The file changed upstream or the download is corrupt."
            )
        note = None if spec.sha256 else "pin this sha256 in configs/inventories.yaml"
        results.append(FetchResult(name, dest, status, digest, note))
    return results
