import hashlib

import pytest
from conftest import make_cfg

from hazres.data.fetch import ChecksumMismatchError, fetch


def _cfg(src, sha=None):
    return make_cfg(
        "x",
        files={
            "a": {"path": "x/a.bin", "url": src.as_uri(), "sha256": sha},
            "b": {"path": "x/b.bin", "note": "on request"},
        },
    )


def test_download_and_pin(tmp_path):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"gully")
    digest = hashlib.sha256(b"gully").hexdigest()
    root = tmp_path / "raw"

    first = fetch(_cfg(src), root)
    assert [(r.name, r.status) for r in first] == [("a", "downloaded"), ("b", "skipped")]
    assert first[0].sha256 == digest and "pin" in first[0].note
    assert (root / "x/a.bin").read_bytes() == b"gully"

    again = fetch(_cfg(src, digest), root)
    assert again[0].status == "present" and again[0].note is None


def test_checksum_mismatch(tmp_path):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"changed")
    with pytest.raises(ChecksumMismatchError):
        fetch(_cfg(src, "0" * 64), tmp_path / "raw")
