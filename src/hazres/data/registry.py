"""The dataset registry: ``configs/inventories.yaml``, validated.

Every hazard dataset is described once, in the config, with where it comes
from, its licence, what its 0 labels mean, and which loader reads it. Code
never hard-codes a path or a URL.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from hazres.data.labels import AbsenceKind, Labels

DEFAULT_REGISTRY = Path("configs/inventories.yaml")
DEFAULT_DATA_ROOT = Path("data/raw")


class FileSpec(BaseModel):
    """One file a dataset needs. ``path`` is relative to the data root and may be a glob."""

    model_config = ConfigDict(extra="forbid")

    path: str
    url: str | None = None
    sha256: str | None = None
    required: bool = True
    note: str | None = None


class InventoryConfig(BaseModel):
    """One hazard dataset."""

    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    hazard: str
    region: str
    tier: Literal[1, 2, 3]
    status: Literal["open", "on_request", "planned"]
    loader: str | None
    citation: str
    licence: str
    real_absences: AbsenceKind
    label_support: str
    crs: str | None = Field(default=None, description="Projected CRS every feature is converted to")
    files: dict[str, FileSpec] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None

    @field_validator("crs")
    @classmethod
    def _crs_is_projected(cls, v: str | None) -> str | None:
        if v is None:
            return v
        from pyproj import CRS

        if CRS.from_user_input(v).is_geographic:
            raise ValueError(f"crs {v} is geographic; use a projected CRS")
        return v

    def file(self, name: str, data_root: Path) -> Path:
        """Resolve a single named file, failing with the download hint if it is missing."""
        spec = self.files.get(name)
        if spec is None:
            raise KeyError(f"{self.key}: no file named {name!r} in the registry")
        matches = resolve(spec, data_root)
        if not matches:
            raise FileNotFoundError(missing_message(self, name, data_root))
        if len(matches) > 1:
            raise ValueError(f"{self.key}.{name}: pattern {spec.path!r} matched {len(matches)}")
        return matches[0]

    def optional_file(self, name: str, data_root: Path) -> Path | None:
        spec = self.files.get(name)
        if spec is None:
            return None
        matches = resolve(spec, data_root)
        return matches[0] if len(matches) == 1 else None


def resolve(spec: FileSpec, data_root: Path) -> list[Path]:
    """Files matching a spec. Globs return every match, sorted."""
    if any(ch in spec.path for ch in "*?["):
        return sorted(data_root.glob(spec.path))
    p = data_root / spec.path
    return [p] if p.exists() else []


def missing_message(cfg: InventoryConfig, name: str, data_root: Path) -> str:
    spec = cfg.files[name]
    where = data_root / spec.path
    if spec.url:
        how = f"run: hazres data fetch {cfg.key}"
    elif cfg.status == "on_request":
        how = "this dataset is available on request from the authors; see the registry notes"
    else:
        how = spec.note or "see the registry notes for how to obtain it"
    return f"{cfg.key}: {name} not found at {where}\n  {how}"


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, InventoryConfig]:
    raw = yaml.safe_load(Path(path).read_text())
    entries = raw.get("inventories") or {}
    return {key: InventoryConfig(key=key, **body) for key, body in entries.items()}


# --- loaders -----------------------------------------------------------------

Loader = Callable[[InventoryConfig, Path], Labels]
_LOADERS: dict[str, Loader] = {}


def register(name: str) -> Callable[[Loader], Loader]:
    """Decorator: make a loader available to the registry under ``name``."""

    def wrap(fn: Loader) -> Loader:
        if name in _LOADERS:
            raise ValueError(f"loader {name!r} registered twice")
        _LOADERS[name] = fn
        return fn

    return wrap


def loaders() -> dict[str, Loader]:
    import hazres.data.inventories  # noqa: F401  (registers the built-in loaders)

    return dict(_LOADERS)


def load_inventory(
    key: str,
    *,
    registry: str | Path | dict[str, InventoryConfig] = DEFAULT_REGISTRY,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> Labels:
    """Load one hazard dataset into the common label format."""
    reg = registry if isinstance(registry, dict) else load_registry(registry)
    if key not in reg:
        raise KeyError(f"unknown dataset {key!r}; known: {', '.join(sorted(reg))}")
    cfg = reg[key]
    if cfg.loader is None:
        raise NotImplementedError(f"{key}: no loader yet (status: {cfg.status}, tier {cfg.tier})")
    available = loaders()
    if cfg.loader not in available:
        raise KeyError(f"{key}: loader {cfg.loader!r} is not registered")
    return available[cfg.loader](cfg, Path(data_root))
