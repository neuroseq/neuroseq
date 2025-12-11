from __future__ import annotations

import pathlib

import pytest

from neuroseq.example_models import example_models
from neuroseq.model import Model
from neuroseq.save_load_options import SaveLoadOptions


@pytest.mark.parametrize(
    "model_generator", example_models.values(), ids=example_models.keys()
)
def test_example_models(model_generator, tmp_path: pathlib.Path) -> None:
    model = model_generator()
    # do a short simulation
    model.run_simulation(0.01)
    # check we can save and re-load the model
    save_options = SaveLoadOptions(True, True, True, True)
    model.save(tmp_path / "model.nseq", save_options)
    model2 = Model(tmp_path / "model.nseq")
    # check saved model matches original
    assert list(model2.stim_arrays.keys()) == list(model.stim_arrays.keys())
    assert list(model2.b2_model.parameters.keys()) == list(
        model.b2_model.parameters.keys()
    )
    assert list(model2.b2_model.neurongroups.keys()) == list(
        model.b2_model.neurongroups.keys()
    )
    assert list(model2.b2_model.synapses.keys()) == list(model.b2_model.synapses.keys())
    assert list(model2.b2_model.monitors.keys()) == list(model.b2_model.monitors.keys())
    assert list(model2.b2_model.initials.keys()) == list(model.b2_model.initials.keys())
    assert list(model2.b2_model.functions.keys()) == list(
        model.b2_model.functions.keys()
    )
