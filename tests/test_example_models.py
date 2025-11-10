from __future__ import annotations

import brian2 as b2
import pytest

from neuroseq import b2_model
from neuroseq.utils import example_models_path


@pytest.mark.parametrize(
    "filename",
    [
        "default_model",
        "example_non-linear",
        "HH_model",
        "LIF_model",
        "LIF_model_without_stim",
    ],
)
def test_example_models(filename: str, tmp_path) -> None:
    model_path = example_models_path / f"{filename}.nseq"
    assert model_path.exists()
    model = b2_model.B2Model()
    model.load_model(model_path)
    # do a short simulation
    model.run_simulation(0.1 * b2.second)
    # check we can save and re-load the model
    model.save_model(tmp_path / "model.nseq")
    model2 = b2_model.B2Model()
    model2.load_model(tmp_path / "model.nseq")
    # check saved model matches original
    assert model2.parameters == model.parameters
    assert model2.neurongroups == model.neurongroups
    assert model2.synapses == model.synapses
    assert model2.monitors == model.monitors
    assert model2.initials == model.initials
    assert model2.functions == model.functions
