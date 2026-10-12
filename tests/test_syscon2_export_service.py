from types import SimpleNamespace

import numpy as np
import pytest

from neuroseq import syscon2_export_service as service
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


@pytest.fixture
def export_input(mocker):
    stim = StimArray("stim", StimType.External, np.ones((64, 2)), "", 0.0005208125)
    settings = service.SysconExportSettings(config_text="template", target_power_mw=0)
    project = SimpleNamespace(cellpose_data=None, power_luts={})
    sanity = SimpleNamespace(is_valid=True, errors=[], fix_messages=[], warnings=[])
    mocker.patch.object(service, "check_syscon2_input_sanity", return_value=sanity)
    report = SimpleNamespace(debug_csv_text="debug", exported_neuron_ids=frozenset({1}))
    generate = mocker.patch.object(
        service, "generate_sequence_text", return_value=("sequence", report)
    )
    mocker.patch.object(
        service,
        "generate_syscon2_timeline_overview_png",
        side_effect=lambda script, path: path.write_bytes(b"png"),
    )
    return project, stim, settings, sanity, generate


@pytest.mark.parametrize(
    "name",
    [
        "",
        "..",
        "../escaped",
        "a/b",
        "a\\b",
        "C:bad",
        "NUL",
        "COM1.txt",
        "bad.",
        "bad ",
        "a\n",
    ],
)
def test_reject_invalid_windows_filename(name):
    with pytest.raises(service.SysconExportError):
        service.validate_export_name(name)


def test_export_uses_actual_exported_targets(export_input, tmp_path):
    project, stim, settings, _, _ = export_input
    result = service.export_stim_array(project, stim, settings, tmp_path, "valid")
    assert result.targeted_stim_ids == frozenset({1})
    assert result.seq_path.read_text() == "sequence"
    assert result.overview_png_path.read_bytes() == b"png"


def test_failed_preview_leaves_existing_files_intact(export_input, tmp_path, mocker):
    project, stim, settings, _, _ = export_input
    existing = [
        tmp_path / "valid.seq",
        tmp_path / "valid_export_debug.csv",
        tmp_path / "valid_overview.png",
    ]
    for path in existing:
        path.write_text("previous")
    mocker.patch.object(
        service,
        "generate_syscon2_timeline_overview_png",
        side_effect=ValueError("render failed"),
    )
    with pytest.raises(ValueError, match="render failed"):
        service.export_stim_array(project, stim, settings, tmp_path, "valid")
    assert all(path.read_text() == "previous" for path in existing)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        path.name for path in existing
    )


@pytest.mark.parametrize("issue", ["errors", "fix_messages", "warnings"])
def test_unreviewed_changes_do_not_write_or_generate(export_input, tmp_path, issue):
    project, stim, settings, sanity, generate = export_input
    setattr(sanity, issue, ["Needs review"])
    sanity.is_valid = issue != "errors"
    with pytest.raises(service.SysconExportError, match="Needs review"):
        service.export_stim_array(project, stim, settings, tmp_path, "valid")
    generate.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_warnings_must_be_accepted_before_generation(export_input, tmp_path):
    project, stim, settings, sanity, generate = export_input
    sanity.warnings = ["Skipped neuron 0"]

    def confirm(warnings):
        assert warnings == sanity.warnings
        generate.assert_not_called()
        return True

    service.export_stim_array(
        project, stim, settings, tmp_path, "valid", confirm_warnings=confirm
    )
    generate.assert_called_once()


@pytest.mark.parametrize("power", [float("nan"), float("inf"), -1])
def test_invalid_power_cannot_export(export_input, tmp_path, power):
    project, stim, _, _, generate = export_input
    with pytest.raises(service.SysconExportError, match="finite"):
        service.export_stim_array(
            project,
            stim,
            service.SysconExportSettings("", target_power_mw=power),
            tmp_path,
            "valid",
        )
    generate.assert_not_called()


def test_missing_lut_cannot_silently_discard_requested_power(
    export_input, tmp_path, mocker
):
    project, stim, _, _, generate = export_input
    mocker.patch.object(service, "_power_lut", return_value=None)
    settings = service.SysconExportSettings("template", target_power_mw=92)
    with pytest.raises(service.SysconExportError, match="92 mW will NOT be applied"):
        service.export_stim_array(project, stim, settings, tmp_path, "valid")
    generate.assert_not_called()
    service.export_stim_array(
        project, stim, settings, tmp_path, "valid", confirm_warnings=lambda _: True
    )
    generate.assert_called_once()


@pytest.mark.parametrize(
    "tuples",
    [(), ((float("nan"), 1),), ((50, float("inf")),), ((101, 1),), ((20, -1),)],
)
def test_invalid_calibration_cannot_export(export_input, tmp_path, mocker, tuples):
    from neuroseq.power_lut import PowerLut

    project, stim, _, _, generate = export_input
    mocker.patch.object(service, "_power_lut", return_value=PowerLut("bad", tuples))
    with pytest.raises(service.SysconExportError, match="Power LUT"):
        service.export_stim_array(
            project, stim, service.SysconExportSettings("template"), tmp_path, "valid"
        )
    generate.assert_not_called()


def test_export_writes_sidecar_with_the_original_stim_array(export_input, tmp_path):
    project, stim, settings, _, _ = export_input
    result = service.export_stim_array(project, stim, settings, tmp_path, "valid")
    sidecar = tmp_path / "valid_original_stim.npz"
    assert sidecar.is_file()
    with np.load(sidecar) as data:
        assert np.array_equal(data["values"], stim.values.astype(np.float32))
        assert float(data["dt"]) == stim.dt
    assert result.seq_path.read_text() == "sequence"
