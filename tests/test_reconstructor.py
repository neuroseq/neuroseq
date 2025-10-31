from collections.abc import Generator
from pathlib import Path

import brian2 as b2
import pytest
from pytest import MonkeyPatch

from neuroseq.b2_model import B2Model
from neuroseq.reconstructor import Reconstructor


@pytest.fixture()
def mfx_nmda_serialized(tmp_path: Path, mfx_nmda: B2Model) -> Generator:
    mfx_nmda.define_dt("neurons", 0.1 * b2.ms)
    mfx_nmda.define_synapse_dt("S", 0.2 * b2.ms)
    file_path = tmp_path / "testmodel.nseq"
    mfx_nmda.save_model(file_path)
    yield file_path


def raise_error(s: B2Model, name: str) -> None:
    raise ValueError("Mocked error")


def test_init(mfx_nmda_serialized: Path) -> None:
    reconstructor = Reconstructor(mfx_nmda_serialized)

    assert "neurons" in reconstructor.loaded_data
    assert "neuron_input" in reconstructor.loaded_data
    assert "S" in reconstructor.loaded_data
    assert "M" in reconstructor.loaded_data
    assert "Mn" in reconstructor.loaded_data


def test_reconstruct_object_exception_property_def(
    mfx_nmda_serialized: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(B2Model, "define_equation", raise_error)
    reconstructor = Reconstructor(mfx_nmda_serialized)
    model = B2Model()

    with pytest.raises(
        RuntimeError, match="Error when defining model for object neurons"
    ):
        reconstructor.reconstruct_neurongroups(
            model, "neurons", reconstructor.loaded_data["neurons"]
        )


def test_reconstruct_object_exception_assembler(
    mfx_nmda_serialized: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(B2Model, "_assemble_neurongroup", raise_error)
    reconstructor = Reconstructor(mfx_nmda_serialized)
    model = B2Model()

    with pytest.raises(RuntimeError, match="Error when assembling object neurons"):
        reconstructor.reconstruct_neurongroups(
            model, "neurons", reconstructor.loaded_data["neurons"]
        )


def test_reconstruct_neurongroup(mfx_nmda_serialized: Path) -> None:
    reconstructor = Reconstructor(mfx_nmda_serialized)
    model = B2Model()

    reconstructor.reconstruct_neurongroups(
        model, "neurons", reconstructor.loaded_data["neurons"]
    )

    assert "neurons" in model.network.get_states()
    assert model.network["neurons"].dt == 0.1 * b2.ms
    assert model.neurongroups["neurons"]["dt"] == 0.1 * b2.ms


def test_reconstruct_neurongsubgroups(mfx_nmda_serialized: Path) -> None:
    reconstructor = Reconstructor(mfx_nmda_serialized)
    model = B2Model()

    reconstructor.reconstruct_neurongroups(
        model, "neuron_input", reconstructor.loaded_data["neuron_input"]
    )

    reconstructor.reconstruct_neuron_subgroup(
        model, "input_subgroup", reconstructor.loaded_data["input_subgroup"]
    )

    assert "input_subgroup" in model.network.get_states()


def test_reconstruct_synapse(mfx_nmda_serialized: Path) -> None:
    reconstructor = Reconstructor(mfx_nmda_serialized)
    model = B2Model()
    reconstructor.reconstruct_neurongroups(
        model, "neurons", reconstructor.loaded_data["neurons"]
    )
    reconstructor.reconstruct_neurongroups(
        model, "neuron_input", reconstructor.loaded_data["neuron_input"]
    )
    reconstructor.reconstruct_synapses(model, "S", reconstructor.loaded_data["S"])

    assert "S" in model.network.get_states()
    assert model.network["S"].dt == 0.2 * b2.ms
    assert model.synapses["S"]["dt"] == 0.2 * b2.ms


def test_reconstruct_monitor(mfx_nmda_serialized: Path) -> None:
    reconstructor = Reconstructor(mfx_nmda_serialized)
    model = B2Model()
    reconstructor.reconstruct_neurongroups(
        model, "neurons", reconstructor.loaded_data["neurons"]
    )
    reconstructor.reconstruct_neurongroups(
        model, "neuron_input", reconstructor.loaded_data["neuron_input"]
    )
    reconstructor.reconstruct_synapses(model, "S", reconstructor.loaded_data["S"])

    reconstructor.reconstruct_monitors(model, "M", reconstructor.loaded_data["M"])

    assert "M" in model.network.get_states()
