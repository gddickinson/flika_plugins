"""The plugin catalogue reads every info.xml folder and its files correctly."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from web.plugins import catalogue, first_paragraph, parse_info, slug, stats  # noqa: E402

CAT = catalogue()


def test_every_info_xml_is_a_plugin():
    n = sum(1 for p in ROOT.rglob("info.xml") if ".git" not in p.parts and "site" not in p.parts)
    assert len(CAT["plugins"]) == n == 64


def test_spt_batch_analysis_entry():
    p = CAT["plugins"]["spt_batch_analysis"]
    assert p["name"] == "SPT Batch Analysis" and p["group"] == "main" and p["topic"] == "spt"
    assert "Launch SPT Analysis" in p["menu"] and "scikit-learn" in p["dependencies"]
    assert p["url"] == ""  # info.xml has a placeholder github.com/username URL
    assert any(f["path"] == "thunderstorm_python/fitting.py" and "compute_localization_precision" in f["functions"]
               for f in p["files"])
    assert p["uses_flika"] and p["readme_name"] == "README.md"


def test_groups_nested_and_same_folder():
    ps = CAT["plugins"]
    assert ps["EXPERIMENTAL.flicker_detector"]["group"] == "experimental"
    assert ps["OLD.pynsight"]["group"] == "old" and ps["backups.spt_batch_analysis_Feb12"]["group"] == "backup"
    assert ps["locsAndTracksPlotter.locsAndTracksPlotter_original"]["nested_in"] == "locsAndTracksPlotter"
    # the nested copy's files are not counted again in its parent
    assert not any(f["path"].startswith("locsAndTracksPlotter_original/") for f in ps["locsAndTracksPlotter"]["files"])
    assert "OLD.detect_puffs" in ps["NOT_WORKING_WITH_PY311.detect_puffs"]["same_folder"]


def test_python2_files_are_flagged_not_fatal():
    s = stats(CAT)
    assert s["parse_errors"] >= 1
    assert "docs/how_to_import_c_shared_lib/importing_c_file.py" in CAT["plugins"]["detect_puffs_GDedit"]["parse_errors"]


def test_helpers(tmp_path):
    assert slug("backups/spt_batch_analysis_Feb12 ") == "backups.spt_batch_analysis_Feb12"
    assert first_paragraph("# T\n\n![b](x.png)\n\nFirst **para** [link](http://a).\nmore\n\nnext") == "First para link. more"
    assert first_paragraph("# T\n\nDownload this repository into x\n") == ""
    (tmp_path / "info.xml").write_text('<plugin name="X"><author> A </author><dependencies><dependency name="numpy"/>'
                                       '</dependencies><menu_layout><action location="a" function="f">Go</action>'
                                       '</menu_layout></plugin>')
    i = parse_info(tmp_path / "info.xml")
    assert (i["name"], i["author"], i["dependencies"], i["menu"]) == ("X", "A", ["numpy"], ["Go"])
