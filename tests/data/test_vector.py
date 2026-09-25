from pathlib import Path

import geopandas as gpd
import pytest
from conftest import make_cfg
from shapely.geometry import Point, Polygon, box

from hazres.data.inventories.vector import load as load_vector
from hazres.data.labels import AbsenceKind


def _polys(crs="EPSG:32637"):
    return gpd.GeoDataFrame(
        {
            "slide_id": ["a", "b", "b", "c"],
            "move_type": ["slide", "fall", "fall", "flow"],
        },
        geometry=[
            box(500_000, 4_100_000, 500_100, 4_100_100),
            box(500_200, 4_100_000, 500_300, 4_100_100),
            box(500_400, 4_100_000, 500_500, 4_100_100),
            # a self-intersecting bow-tie, which the loader must repair
            Polygon([(0, 0), (10, 10), (10, 0), (0, 10)]),
        ],
        crs=crs,
    )


def _cfg(**over):
    base = dict(
        loader="vector",
        crs="EPSG:32637",
        real_absences="within_footprint",
        files={"features": {"path": "k/slides.gpkg"}, "footprint": {"path": "k/fp.gpkg"}},
        options={"geometry": "polygon", "id_column": "slide_id", "keep_columns": ["move_type"]},
    )
    base.update(over)
    return make_cfg("kahramanmaras_2023", **base)


def test_polygons_with_footprint(tmp_path: Path):
    (tmp_path / "k").mkdir()
    _polys().to_file(tmp_path / "k/slides.gpkg")
    gpd.GeoDataFrame(
        geometry=[box(499_000, 4_099_000, 501_000, 4_101_000)], crs="EPSG:32637"
    ).to_file(tmp_path / "k/fp.gpkg")
    labels = load_vector(_cfg(), tmp_path)
    f = labels.features
    assert list(f.columns) == ["sample_id", "label", "move_type", "geometry"]
    assert list(f["sample_id"]) == ["a", "b", "c"]
    assert (f["label"] == 1).all()
    assert labels.absence_kind is AbsenceKind.WITHIN_FOOTPRINT
    assert labels.footprint is not None
    assert labels.report.dropped == {"duplicate slide_id (first kept)": 1}
    assert any("repaired 1" in w for w in labels.report.warnings)
    assert any("outside the mapped footprint" in w for w in labels.report.warnings)
    assert f.geometry.is_valid.all()


def test_missing_footprint_is_flagged(tmp_path):
    (tmp_path / "k").mkdir()
    _polys().to_file(tmp_path / "k/slides.gpkg")
    labels = load_vector(_cfg(), tmp_path)
    assert labels.footprint is None
    assert any("absences cannot be placed" in w for w in labels.report.warnings)


@pytest.mark.filterwarnings("ignore:'crs' was not provided")
def test_file_without_crs_is_refused(tmp_path):
    (tmp_path / "k").mkdir()
    gpd.GeoDataFrame({"slide_id": ["a"]}, geometry=[box(0, 0, 1, 1)]).to_file(
        tmp_path / "k/slides.gpkg"
    )
    with pytest.raises(ValueError, match="no CRS"):
        load_vector(_cfg(), tmp_path)


def test_presence_only_points_reprojected(tmp_path):
    (tmp_path / "d").mkdir()
    gpd.GeoDataFrame(
        {"kind": ["head", "head", "channel"]},
        geometry=[Point(7.0, 6.0), Point(7.1, 6.1), Point(7.2, 6.2)],
        crs="EPSG:4326",
    ).to_file(tmp_path / "d/heads.gpkg")
    cfg = make_cfg(
        "de_geeter_africa",
        crs="ESRI:102022",
        real_absences="none",
        files={"features": {"path": "d/heads.gpkg"}},
        options={"geometry": "point", "label_column": "kind", "presence_values": ["head"]},
    )
    labels = load_vector(cfg, tmp_path)
    assert labels.report.n_presence == 2
    assert labels.report.dropped == {"kind code 'channel' (not presence or absence)": 1}
    assert labels.crs.to_string() == "ESRI:102022"
    assert list(labels.features["sample_id"]) == ["de_geeter_africa-0", "de_geeter_africa-1"]


def test_geometry_filter(tmp_path):
    (tmp_path / "k").mkdir()
    mixed = _polys().iloc[:1]
    mixed = gpd.GeoDataFrame(
        {"slide_id": ["a", "p"]},
        geometry=[mixed.geometry.iloc[0], Point(500_050, 4_100_050)],
        crs="EPSG:32637",
    )
    mixed.to_file(tmp_path / "k/slides.gpkg")
    labels = load_vector(_cfg(options={"geometry": "polygon", "id_column": "slide_id"}), tmp_path)
    assert list(labels.features["sample_id"]) == ["a"]
    assert labels.report.dropped == {"geometry not polygon": 1}
