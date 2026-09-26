import re
from pathlib import Path

import pytest
from conftest import make_cfg, write_zip

from hazres.data.inventories.ge_lucas import CodingNotConfirmedError
from hazres.data.inventories.ge_lucas import load as load_ge_lucas

HEADER = (
    "POINT_ID,POINT_LAT,POINT_LONG,POINT_NUTS0,SURVEY_GULLY_SIGNS,SURVEY_GULLY_TYPE,SURVEY_LC1\n"
)
ROWS = [
    "1001,48.1,11.5,DE,1,2,B11",
    "1002,48.2,11.6,DE,2,NA,B11",
    "1003,41.9,12.5,IT,2,NA,C10",
    "1004,41.8,12.4,IT,8,NA,C10",  # not a Yes/No answer
    "1005,,12.4,IT,1,1,C10",  # no coordinates
    "1002,48.2,11.6,DE,2,NA,B11",  # duplicate id
]


def _cfg(**options):
    opts = {"expected_presences": 1, "keep_columns": ["POINT_NUTS0", "NOT_A_COLUMN"]}
    opts.update(options)
    return make_cfg(
        "ge_lucas",
        loader="ge_lucas",
        files={"points": {"path": "ge_lucas/points.zip"}},
        options=opts,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_zip(tmp_path / "ge_lucas/points.zip", "LUCAS2022_original.CSV", HEADER + "\n".join(ROWS))
    return tmp_path


def test_refuses_to_guess_the_coding(root):
    with pytest.raises(CodingNotConfirmedError) as exc:
        load_ge_lucas(_cfg(), root)
    msg = str(exc.value)
    assert "SURVEY_GULLY_SIGNS" in msg
    # every code is listed with its count, so a person can confirm the coding
    assert re.search(r"^2\s+3$", msg, re.MULTILINE)
    assert re.search(r"^8\s+1$", msg, re.MULTILINE)


def test_loads_with_confirmed_coding(root):
    labels = load_ge_lucas(_cfg(presence_values=[1], absence_values=["2"]), root)
    f = labels.features
    assert list(f["sample_id"]) == ["1001", "1002", "1003"]
    assert list(f["label"]) == [1, 0, 0]
    assert f.crs.to_epsg() == 3035
    assert list(f["POINT_NUTS0"]) == ["DE", "DE", "IT"]
    r = labels.report
    assert (r.n_read, r.n_presence, r.n_absence) == (6, 1, 2)
    assert r.dropped["gully code '8' (not presence or absence)"] == 1
    assert r.dropped["missing or invalid coordinates"] == 1
    assert r.dropped["duplicate POINT_ID (first kept)"] == 1
    assert any("NOT_A_COLUMN" in w for w in r.warnings)
    assert not any("paper reports" in w for w in r.warnings)


def test_warns_when_presences_do_not_match_the_paper(root):
    labels = load_ge_lucas(
        _cfg(presence_values=["1"], absence_values=["2"], expected_presences=3116), root
    )
    assert any("paper reports 3,116" in w for w in labels.report.warnings)


def test_same_code_cannot_mean_both(root):
    with pytest.raises(ValueError, match="both"):
        load_ge_lucas(_cfg(presence_values=["1"], absence_values=["1", "2"]), root)


def test_gully_signs_is_chosen_over_the_other_gully_columns(root):
    labels = load_ge_lucas(_cfg(presence_values=["1"], absence_values=["2"]), root)
    assert labels.meta["presence_column"] == "SURVEY_GULLY_SIGNS"


def test_ambiguous_presence_column(tmp_path):
    header = "POINT_ID,POINT_LAT,POINT_LONG,SURVEY_GULLY,SURVEY_GULLY_X\n"
    write_zip(tmp_path / "ge_lucas/points.zip", "x.csv", header + "1,48,11,1,2")
    with pytest.raises(KeyError, match="presence column"):
        load_ge_lucas(_cfg(presence_values=["1"], absence_values=["2"]), tmp_path)


def test_unzipped_csv_also_works(tmp_path):
    (tmp_path / "ge_lucas").mkdir()
    (tmp_path / "ge_lucas/points.csv").write_text(HEADER + ROWS[0])
    cfg = _cfg(presence_values=["1"], absence_values=["2"])
    cfg.files["points"].path = "ge_lucas/points.csv"
    assert load_ge_lucas(cfg, tmp_path).report.n_presence == 1


def _write_locations(root: Path, rows):
    import zipfile as zf

    import geopandas as gpd
    from shapely.geometry import Point

    shp_dir = root / "shp"
    shp_dir.mkdir(exist_ok=True)
    gpd.GeoDataFrame(
        {"POINT_ID": [r[0] for r in rows]},
        geometry=[Point(r[1], r[2]) for r in rows],
        crs="EPSG:3035",
    ).to_file(shp_dir / "locations.shp")
    path = root / "ge_lucas" / "d3.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zf.ZipFile(path, "w") as z:
        for f in shp_dir.iterdir():
            z.write(f, f"Data 3 - locations/{f.name}")  # inside a subfolder, like the real zip
    return path


def test_gully_offsets_are_reported_and_location_is_optional(root):
    import geopandas as gpd

    base = load_ge_lucas(_cfg(presence_values=["1"], absence_values=["2"]), root)
    p = base.features.loc[base.features["label"] == 1].geometry.iloc[0]
    _write_locations(root, [("1001", p.x + 40.0, p.y)])  # the gully is 40 m east
    cfg = _cfg(presence_values=["1"], absence_values=["2"])
    cfg.files["gully_locations"] = type(cfg.files["points"])(path="ge_lucas/d3.zip")

    kept = load_ge_lucas(cfg, root)
    assert kept.meta["presence_location"] == "lucas_point"
    assert kept.meta["gully_offset_m"]["median"] == pytest.approx(40.0)
    assert kept.features.geometry.iloc[0].x == pytest.approx(p.x)  # default: point unchanged

    cfg.options["presence_location"] = "gully_location"
    moved = load_ge_lucas(cfg, root)
    assert moved.features.geometry.iloc[0].x == pytest.approx(p.x + 40.0)
    absent = moved.features[moved.features["label"] == 0]
    pd_equal = gpd.GeoSeries(absent.geometry).equals(
        gpd.GeoSeries(kept.features[kept.features["label"] == 0].geometry)
    )
    assert pd_equal  # no-gully points never move


def test_gully_location_needs_data_3(root):
    cfg = _cfg(presence_values=["1"], absence_values=["2"], presence_location="gully_location")
    with pytest.raises(FileNotFoundError, match="Data 3"):
        load_ge_lucas(cfg, root)
