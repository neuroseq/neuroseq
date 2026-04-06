from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest

from neuroseq.exporter_igor2 import export_igor2_patchclamp_csv
from neuroseq.exporter_igor2 import generate_igor2_patchclamp_csv_text
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def test_generate_igor2_patchclamp_csv_text_exports_single_neuron_column() -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array(
            [
                [0.0, 10.0],
                [1.5, 20.0],
                [0.0, 30.0],
            ],
            dtype=np.float32,
        ),
        units="pA",
        dt=0.001,
    )

    csv_text = generate_igor2_patchclamp_csv_text(
        stim_array,
        experiment_name="one_neuron_stim_example",
        neuron_index=1,
    )

    assert csv_text == "one_neuron_stim_example\n10.0\n20.0\n30.0\n"


def test_export_igor2_patchclamp_csv_writes_file(tmp_path: Path) -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.array([[0.0], [2.0], [4.0]], dtype=np.float32),
        units="pA",
        dt=0.001,
    )
    output_path = tmp_path / "export.csv"

    report = export_igor2_patchclamp_csv(
        output_path,
        stim_array,
        experiment_name="example_export",
        neuron_index=0,
    )

    assert output_path.read_text(encoding="utf-8") == "example_export\n0.0\n2.0\n4.0\n"
    assert report.experiment_name == "example_export"
    assert report.neuron_index == 0
    assert report.sample_count == 3


def test_generate_igor2_patchclamp_csv_text_rejects_invalid_neuron() -> None:
    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.zeros((4, 1), dtype=np.float32),
        units="pA",
        dt=0.001,
    )

    with pytest.raises(ValueError, match="out of range"):
        generate_igor2_patchclamp_csv_text(
            stim_array,
            experiment_name="example_export",
            neuron_index=3,
        )
