from hazres.cli import main


def test_data_list(capsys, tmp_path):
    assert main(["data", "--data-root", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert "ge_lucas" in out and "kahramanmaras_2023" in out
    assert out.index("ge_lucas") < out.index("mcd64a1_burned_area")  # sorted by tier


def test_inspect_missing_data_is_a_clean_error(capsys, tmp_path):
    assert main(["data", "--data-root", str(tmp_path), "inspect", "ge_lucas"]) == 1
    assert "hazres data fetch ge_lucas" in capsys.readouterr().err


def test_predictors_list(capsys):
    assert main(["data", "predictors", "list"]) == 0
    out = capsys.readouterr().out
    assert "elevation" in out and "soil_thickness" in out and "login" in out
