"""Shared helpers: small stand-in files with the documented layout of each dataset."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import yaml

from hazres.data.registry import InventoryConfig

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo_registry() -> Path:
    return REPO / "configs" / "inventories.yaml"


def make_cfg(key: str, **overrides) -> InventoryConfig:
    base = {
        "key": key,
        "name": key,
        "hazard": "gully",
        "region": "test",
        "tier": 1,
        "status": "open",
        "loader": "vector",
        "citation": "test",
        "licence": "test",
        "real_absences": "observed",
        "label_support": "point",
        "crs": "EPSG:3035",
    }
    base.update(overrides)
    return InventoryConfig(**base)


def write_zip(path: Path, member: str, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, text)
    return path


def write_registry(path: Path, entries: dict) -> Path:
    path.write_text(yaml.safe_dump({"version": 1, "inventories": entries}))
    return path
