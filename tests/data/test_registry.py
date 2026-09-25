from pathlib import Path

import pytest
from conftest import make_cfg
from pydantic import ValidationError

from hazres.data.labels import AbsenceKind
from hazres.data.registry import load_inventory, load_registry, loaders


def test_repo_registry_is_valid(repo_registry):
    reg = load_registry(repo_registry)
    assert {"ge_lucas", "de_geeter_africa", "kahramanmaras_2023", "global_flood_database"} <= set(
        reg
    )
    known = loaders()
    for key, cfg in reg.items():
        assert cfg.loader is None or cfg.loader in known, key
        if cfg.tier < 3:
            assert cfg.loader is not None, f"{key}: tier {cfg.tier} needs a loader"


def test_tiers_match_the_study_design(repo_registry):
    reg = load_registry(repo_registry)
    tier1 = {k: c for k, c in reg.items() if c.tier == 1}
    assert set(tier1) == {"ge_lucas", "de_geeter_africa", "chen_gully_sites_africa"}
    # only GE-LUCAS is open; the African gully datasets are requested from the authors
    assert [k for k, c in tier1.items() if c.status == "open"] == ["ge_lucas"]
    assert reg["chen_gully_sites_africa"].real_absences is AbsenceKind.OBSERVED
    assert reg["ge_lucas"].real_absences is AbsenceKind.OBSERVED
    assert reg["de_geeter_africa"].real_absences is AbsenceKind.NONE


def test_geographic_crs_is_rejected():
    with pytest.raises(ValidationError, match="geographic"):
        make_cfg("x", crs="EPSG:4326")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_cfg("x", tierr=1)


def test_missing_file_says_how_to_get_it(tmp_path: Path, repo_registry):
    with pytest.raises(FileNotFoundError, match="hazres data fetch ge_lucas"):
        load_inventory("ge_lucas", registry=repo_registry, data_root=tmp_path)


def test_planned_dataset_has_no_loader(tmp_path, repo_registry):
    with pytest.raises(NotImplementedError):
        load_inventory("mcd64a1_burned_area", registry=repo_registry, data_root=tmp_path)
