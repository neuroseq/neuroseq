from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import brian2 as b2
import brian2.numpy_ as np
import h5py
import pytest
from brian2.tests.utils import assert_allclose
from pytest import MonkeyPatch

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from neuroseq.b2_model import B2Model
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray


def test_default_construction() -> None:
    model = B2Model()
    assert model.parameters == {}
    assert model.network is not None
    assert model.synapses == {}
    assert model.monitors == {}


@pytest.fixture()
def mfx_nmda_as_dict(b2_model_nmda: B2Model) -> dict:
    b2_model_nmda.define_dt("neurons", 0.1 * b2.ms)
    b2_model_nmda.define_synapse_dt("S", 0.2 * b2.ms)
    return b2_model_nmda.to_dict()


def test_from_dict_empty_returns_default_model() -> None:
    # Projects with no serialized model (e.g. stim-array-only archives) load an
    # empty b2_model dict; from_dict must return a default model, not KeyError.
    model = B2Model.from_dict({}, {})
    assert model.seed == 123
    assert model.parameters == {}
    assert model.neurongroups == {}
    assert model.synapses == {}


def test_from_dict_missing_seed_uses_default(mfx_nmda_as_dict: dict) -> None:
    data = dict(mfx_nmda_as_dict)
    data.pop("seed", None)
    model = B2Model.from_dict(data, {})
    assert model.seed == 123


def test_reconstruct_mfx_nmda_serialized(mfx_nmda_as_dict: dict) -> None:
    model = B2Model.from_dict(mfx_nmda_as_dict, {})
    assert "neurons" in model.network.get_states()
    assert model.network["neurons"].dt == 0.1 * b2.ms
    assert model.neurongroups["neurons"]["dt"] == 0.1 * b2.ms
    assert "input_subgroup" in model.network.get_states()
    assert "S" in model.network.get_states()
    assert model.network["S"].dt == 0.2 * b2.ms
    assert model.synapses["S"]["dt"] == 0.2 * b2.ms
    assert "M" in model.network.get_states()


def test_define_equation(b2model: B2Model) -> None:
    assert b2model.neurongroups == {}

    b2model.define_equation("P", "dge/dt = -ge*(1./taue) : siemens")

    assert b2model.neurongroups["P"]["model"] == "dge/dt = -ge*(1./taue) : siemens"
    assert all(
        f in b2model.functions["neurongroups"]["P"]
        for f in [
            "rename",
            "delete",
        ]
    )

    assert all(
        f in b2model.functions["neurongroups"]["P"]["model"]
        for f in ["delete", "modify", "rename"]
    )

    with pytest.raises(
        ValueError,
        match="Equation must have a colon that separates equation and unit or an equal sign",
    ):
        b2model.define_equation("P", "dge/dt -ge*(1./taue)")

    with pytest.raises(
        ValueError,
        match="Equation must have a colon that separates equation and unit or an equal sign",
    ):
        b2model.define_equation("P", "-ge*(1./taue) siemens")


def test_define_dt(b2model: B2Model) -> None:
    assert b2model.neurongroups == {}
    standard_dt = b2.defaultclock.dt

    b2model.define_dt("neurons", 0.1 * b2.ms)
    b2model.define_synapse_dt("S", 0.75 * b2.ms)
    b2model.define_monitor_dt("M", 10 * b2.ms)

    assert b2.defaultclock.dt == standard_dt
    assert np.isclose(b2model.neurongroups["neurons"]["dt"].base, 0.0001)
    assert b2model.neurongroups["neurons"]["dt"].has_same_dimensions(0.1 * b2.ms)
    assert np.isclose(b2model.synapses["S"]["dt"].base, 0.00075)
    assert b2model.synapses["S"]["dt"].has_same_dimensions(0.75 * b2.ms)
    assert np.isclose(b2model.monitors["M"]["dt"].base, 0.01)
    assert b2model.monitors["M"]["dt"].has_same_dimensions(0.01 * b2.second)


def test_define_parameter(b2model: B2Model) -> None:
    assert b2model.parameters == {}
    b2model.define_parameter("taue", 5 * b2.ms)
    b2model.define_parameter("taui", 10 * b2.ms)
    b2model.define_parameter("area", 20000 * b2.umetre**2)
    timed_array = b2.TimedArray(np.random.rand(10, 100) * b2.mS, dt=1 * b2.ms)
    b2model.define_parameter("stim", timed_array)

    assert b2model.parameters == {
        "taue": 5 * b2.ms,
        "taui": 10 * b2.ms,
        "area": 20000 * b2.umetre**2,
        "stim": timed_array,
    }


def test_define_threshold(b2model: B2Model) -> None:
    assert b2model.neurongroups == {}
    b2model.define_threshold("P", "v > 10 * mV")
    assert b2model.neurongroups["P"]["th"] == "v > 10 * mV"


def test_define_solver_method(b2model: B2Model) -> None:
    assert b2model.neurongroups == {}
    b2model.define_solver_method("P", "euler")
    assert b2model.neurongroups["P"]["method"] == "euler"


def test_define_refractory(b2model: B2Model) -> None:
    assert b2model.neurongroups == {}
    b2model.define_refractory("P", 5 * b2.ms)
    assert b2model.neurongroups["P"]["refractory"] == 5 * b2.ms


def test_define_neuron_number(b2model: B2Model) -> None:
    assert b2model.neurongroups == {}
    b2model.define_neuronnumber("P", 100)
    assert b2model.neurongroups["P"]["neuronnumber"] == 100


def test_get_defined_neuron_count(b2model: B2Model) -> None:
    # no groups yet
    assert b2model.get_defined_neuron_count() == 0
    # largest full-group count wins
    b2model.define_neuronnumber("P", 40)
    b2model.define_neuronnumber("Q", 100)
    assert b2model.get_defined_neuron_count() == 100
    # subgroups (with idxs) are skipped
    b2model.define_neuronsubgroup_target("Pe", "P")
    b2model.define_neuronsubgroup_idxs("Pe", (0, 32))
    assert b2model.get_defined_neuron_count() == 100
    # non-integer neuronnumbers are ignored, not fatal
    b2model.define_neuronnumber("R", "not_an_int")
    assert b2model.get_defined_neuron_count() == 100


def test_assemble_neurongroup_exception(b2model: B2Model) -> None:
    with pytest.raises(RuntimeError, match="Error in assembling neuron group N"):
        b2model._assemble_neurongroup("N")


def test_rename_neurongroup(b2_model_nmda: B2Model) -> None:
    assert "neurons" in b2_model_nmda.neurongroups
    test_data = deepcopy(b2_model_nmda.neurongroups["neurons"])

    b2_model_nmda.functions["neurongroups"]["neurons"]["rename"]("renamed")
    assert "neurons" not in b2_model_nmda.neurongroups
    assert "neurons" not in b2_model_nmda.functions["neurongroups"]

    assert "renamed" in b2_model_nmda.neurongroups
    assert "renamed" in b2_model_nmda.functions["neurongroups"]

    assert b2_model_nmda.neurongroups["renamed"] == test_data

    assert all(
        f in b2_model_nmda.functions["neurongroups"]["renamed"]
        for f in [
            "rename",
            "delete",
        ]
    )


def test_rename_synapse(b2_model_nmda: B2Model) -> None:
    assert "S" in b2_model_nmda.synapses
    test_data = deepcopy(b2_model_nmda.synapses["S"])

    b2_model_nmda.functions["synapses"]["S"]["rename"]("renamed")

    assert "S" not in b2_model_nmda.synapses
    assert "S" not in b2_model_nmda.functions["synapses"]

    assert "renamed" in b2_model_nmda.synapses
    assert "renamed" in b2_model_nmda.functions["synapses"]

    assert b2_model_nmda.synapses["renamed"] == test_data
    assert all(
        f in b2_model_nmda.functions["synapses"]["renamed"]
        for f in [
            "rename",
            "delete",
        ]
    )


def test_rename_parameter(b2_model_nmda: B2Model) -> None:
    assert "a" in b2_model_nmda.parameters
    test_data = deepcopy(b2_model_nmda.parameters["a"])

    b2_model_nmda.functions["parameters"]["a"]["rename"]("renamed")
    assert "a" not in b2_model_nmda.parameters
    assert "a" not in b2_model_nmda.functions["parameters"]

    assert "renamed" in b2_model_nmda.parameters
    assert "renamed" in b2_model_nmda.functions["parameters"]

    assert b2_model_nmda.parameters["renamed"] == test_data
    assert all(
        f in b2_model_nmda.functions["parameters"]["renamed"]
        for f in ["delete", "modify", "rename"]
    )


def test_rename_monitor(b2_model_nmda: B2Model) -> None:
    assert "M" in b2_model_nmda.monitors
    test_data = deepcopy(b2_model_nmda.monitors["M"])

    b2_model_nmda.functions["monitors"]["M"]["rename"]("renamed")
    assert "M" not in b2_model_nmda.monitors
    assert "M" not in b2_model_nmda.functions["monitors"]

    assert "renamed" in b2_model_nmda.monitors
    assert "renamed" in b2_model_nmda.functions["monitors"]

    assert b2_model_nmda.monitors["renamed"] == test_data

    assert all(
        f in b2_model_nmda.functions["monitors"]["renamed"]
        for f in [
            "rename",
            "delete",
        ]
    )


def test_rename_init(b2_model_nmda: B2Model) -> None:
    assert "S" in b2_model_nmda.initials
    test_data = deepcopy(b2_model_nmda.initials["S"])

    b2_model_nmda.functions["initials"]["S"]["rename"]("renamed")

    assert "S" not in b2_model_nmda.initials
    assert "S" not in b2_model_nmda.functions["initials"]

    assert "renamed" in b2_model_nmda.initials
    assert "renamed" in b2_model_nmda.functions["initials"]

    assert b2_model_nmda.initials["renamed"] == test_data
    assert all(
        f in b2_model_nmda.functions["initials"]["renamed"]
        for f in [
            "rename",
            "delete",
        ]
    )


def test_rename_init_element(b2_model_nmda: B2Model) -> None:
    assert "w" in b2_model_nmda.initials["S"]
    test_data = deepcopy(b2_model_nmda.initials["S"]["w"])
    b2_model_nmda.functions["initials"]["S"]["w"]["rename"]("renamed")

    assert "w" not in b2_model_nmda.initials["S"]
    assert "w" not in b2_model_nmda.functions["initials"]["S"]

    assert "renamed" in b2_model_nmda.initials["S"]
    assert "renamed" in b2_model_nmda.functions["initials"]["S"]

    assert b2_model_nmda.initials["S"]["renamed"] == test_data
    print(b2_model_nmda.functions["initials"]["S"]["renamed"])
    assert all(
        f in b2_model_nmda.functions["initials"]["S"]["renamed"]
        for f in ["delete", "modify", "rename"]
    )


def test_rename_neurongroup_element_with_modification(
    b2_model_nmda: B2Model,
) -> None:
    assert "model" in b2_model_nmda.neurongroups["neurons"]
    test_data = deepcopy(b2_model_nmda.neurongroups["neurons"]["model"])

    b2_model_nmda.functions["neurongroups"]["neurons"]["model"]["rename"]("renamed")

    assert "model" not in b2_model_nmda.neurongroups["neurons"]
    assert "model" not in b2_model_nmda.functions["neurongroups"]["neurons"]

    assert "renamed" in b2_model_nmda.neurongroups["neurons"]
    assert "renamed" in b2_model_nmda.functions["neurongroups"]["neurons"]

    assert b2_model_nmda.neurongroups["neurons"]["renamed"] == test_data
    assert all(
        f in b2_model_nmda.functions["neurongroups"]["neurons"]["renamed"]
        for f in ["delete", "modify", "rename"]
    )

    b2_model_nmda.functions["neurongroups"]["neurons"]["renamed"]["modify"](
        "dv/dt=-v:1"
    )

    assert b2_model_nmda.neurongroups["neurons"]["renamed"] == "dv/dt=-v:1"

    assert "renamed" in b2_model_nmda.functions["neurongroups"]["neurons"]
    assert "renamed" in b2_model_nmda.neurongroups["neurons"]

    b2_model_nmda.functions["neurongroups"]["neurons"]["renamed"]["delete"]()

    assert "renamed" not in b2_model_nmda.functions["neurongroups"]["neurons"]
    assert "renamed" not in b2_model_nmda.neurongroups["neurons"]


def test_rename_synapse_element(b2_model_nmda: B2Model) -> None:
    assert "model" in b2_model_nmda.synapses["S"]
    test_data = deepcopy(b2_model_nmda.synapses["S"]["model"])

    b2_model_nmda.functions["synapses"]["S"]["model"]["rename"]("renamed")

    assert "model" not in b2_model_nmda.synapses["S"]
    assert "model" not in b2_model_nmda.functions["synapses"]["S"]

    assert "renamed" in b2_model_nmda.synapses["S"]
    assert "renamed" in b2_model_nmda.functions["synapses"]["S"]

    assert b2_model_nmda.synapses["S"]["renamed"] == test_data
    assert all(
        f in b2_model_nmda.functions["synapses"]["S"]["renamed"]
        for f in ["delete", "modify", "rename"]
    )

    b2_model_nmda.functions["synapses"]["S"]["renamed"]["rename"]("another_name")

    assert "renamed" not in b2_model_nmda.synapses["S"]
    assert "renamed" not in b2_model_nmda.functions["synapses"]["S"]

    assert "another_name" in b2_model_nmda.synapses["S"]
    assert "another_name" in b2_model_nmda.functions["synapses"]["S"]

    assert b2_model_nmda.synapses["S"]["another_name"] == test_data
    assert all(
        f in b2_model_nmda.functions["synapses"]["S"]["another_name"]
        for f in ["delete", "modify", "rename"]
    )


def test_rename_monitor_element(b2_model_nmda: B2Model) -> None:
    assert "type" in b2_model_nmda.monitors["M"]
    test_data = deepcopy(b2_model_nmda.monitors["M"]["type"])

    b2_model_nmda.functions["monitors"]["M"]["type"]["rename"]("renamed")

    assert "type" not in b2_model_nmda.monitors["M"]
    assert "type" not in b2_model_nmda.functions["monitors"]["M"]

    assert "renamed" in b2_model_nmda.monitors["M"]
    assert "renamed" in b2_model_nmda.functions["monitors"]["M"]

    assert b2_model_nmda.monitors["M"]["renamed"] == test_data
    assert all(
        f in b2_model_nmda.functions["monitors"]["M"]["renamed"]
        for f in ["delete", "modify", "rename"]
    )


def test_neuron_subgroup(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", (0, 20))

    mfx.define_neuronsubgroup_target("inhibitory", "P")
    mfx.define_neuronsubgroup_idxs("inhibitory", (20, 30))

    assert mfx.neurongroups["excitatory"]["idxs"] == (0, 20)
    assert mfx.neurongroups["inhibitory"]["idxs"] == (20, 30)

    mfx._assemble_neuronsubgroup("excitatory")
    mfx._assemble_neuronsubgroup("inhibitory")

    assert mfx.network["excitatory"] is not None
    assert mfx.network["inhibitory"] is not None


def test_neuron_subgroup_string_idxs(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
) -> None:
    # The model editor stores edited idxs as strings; assembling must parse
    # both a "(start, stop)" pair and a bare integer index.
    mfx, _ = mfx_parameterized

    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", "[0, 20]")
    mfx._assemble_neuronsubgroup("excitatory")
    assert mfx.network["excitatory"].N == 20

    mfx.define_neuronsubgroup_target("single", "P")
    mfx.define_neuronsubgroup_idxs("single", "5")
    mfx._assemble_neuronsubgroup("single")
    assert mfx.network["single"].N == 1


def test_neuron_subgroup_invalid_idxs(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized
    mfx.define_neuronsubgroup_target("bad", "P")
    mfx.define_neuronsubgroup_idxs("bad", "not_a_range")
    with pytest.raises(RuntimeError, match="could not parse subgroup idxs"):
        mfx._assemble_neuronsubgroup("bad")


def test_parse_subgroup_idxs() -> None:
    parse = B2Model._parse_subgroup_idxs
    assert parse((0, 5)) == (0, 5)
    assert parse([2, 7]) == (2, 7)
    assert parse("[0, 3]") == (0, 3)
    assert parse("(1, 4)") == (1, 4)
    assert parse(3) == (3, 4)
    assert parse("5") == (5, 6)
    assert parse([4]) == (4, 5)
    with pytest.raises(ValueError):
        parse("not_a_range")
    with pytest.raises(ValueError):
        parse((1, 2, 3))


def test_modify_neuronnumber_rescales_subgroups(b2model: B2Model) -> None:
    b2model.define_neuronnumber("P", 2)
    b2model.define_neuronsubgroup_target("Pe", "P")
    b2model.define_neuronsubgroup_idxs("Pe", (0, 1))
    b2model.define_neuronsubgroup_target("Pi", "P")
    b2model.define_neuronsubgroup_idxs("Pi", (1, 2))

    # the model-editor edit path for the neuron count
    b2model.functions["neurongroups"]["P"]["neuronnumber"]["modify"](40)

    assert b2model.neurongroups["P"]["neuronnumber"] == 40
    # proportional rescale preserves the 50/50 split
    assert b2model.neurongroups["Pe"]["idxs"] == (0, 20)
    assert b2model.neurongroups["Pi"]["idxs"] == (20, 40)


def test_modify_neuronnumber_preserves_uneven_ratio(b2model: B2Model) -> None:
    b2model.define_neuronnumber("P", 10)
    b2model.define_neuronsubgroup_target("Pe", "P")
    b2model.define_neuronsubgroup_idxs("Pe", (0, 8))  # 80% excitatory
    b2model.define_neuronsubgroup_target("Pi", "P")
    b2model.define_neuronsubgroup_idxs("Pi", (8, 10))

    b2model.functions["neurongroups"]["P"]["neuronnumber"]["modify"](40)
    assert b2model.neurongroups["Pe"]["idxs"] == (0, 32)
    assert b2model.neurongroups["Pi"]["idxs"] == (32, 40)


def test_neuron_subgroup_remove_exception(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
    monkeypatch: MonkeyPatch,
) -> None:
    mfx, _ = mfx_parameterized
    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", (0, 20))
    mfx._assemble_neuronsubgroup("excitatory")

    def do_raise(name: Any) -> None:
        raise RuntimeError("dummy")

    monkeypatch.setattr(b2.Network, "remove", do_raise)

    with pytest.raises(
        RuntimeError, match="Error when removing existing subgroup excitatory:"
    ):
        mfx._assemble_neuronsubgroup("excitatory")


def test_neuron_subgroup_add_exception(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
    monkeypatch: MonkeyPatch,
) -> None:
    mfx, _ = mfx_parameterized
    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", (0, 20))

    def do_raise(name: Any) -> None:
        raise RuntimeError("dummy")

    monkeypatch.setattr(b2.Network, "add", do_raise)

    with pytest.raises(RuntimeError, match="Error when adding subgroup"):
        mfx._assemble_neuronsubgroup("excitatory")


def test_neuron_subgroup_creation_exception(
    mfx_parameterized: tuple[B2Model, dict[str, Any]], monkeypatch: MonkeyPatch
) -> None:
    mfx, _ = mfx_parameterized
    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", (0, 20))

    def do_raise(*_, **__) -> None:
        raise RuntimeError("dummy")

    monkeypatch.setattr(b2, "Subgroup", do_raise)

    with pytest.raises(
        RuntimeError, match="Error when trying to create a neuronsubgroup"
    ):
        mfx._assemble_neuronsubgroup("excitatory")


def test_synapse_connection_prob_numerical(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    mfx.define_synapse_connection_probability("Ce", 0.5)

    assert np.isclose(mfx.synapses["Ce"]["p"], 0.5)

    boundary = np.nextafter(1.0, 2.0)
    with pytest.raises(ValueError) as e:
        mfx.define_synapse_connection_probability("Ce", boundary)
    assert "error" in str(e.value).lower()
    assert str(boundary) in str(e.value)

    boundary = np.nextafter(0.0, -1.0)
    with pytest.raises(ValueError) as e:
        mfx.define_synapse_connection_probability("Ce", boundary)
    assert "error" in str(e.value).lower()
    assert str(boundary) in str(e.value)


def test_synapse_connection_prob_string_expression(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    # a string expression is stored verbatim and not range-checked here
    mfx.define_synapse_connection_probability("Ce", "1.0 * (j > i)")

    assert mfx.synapses["Ce"]["p"] == "1.0 * (j > i)"


def test_build_connect_kwargs_condition_mode() -> None:
    kwargs = B2Model._build_connect_kwargs(
        {
            "connection_condition": "i != j",
            "p": 0.3,
            "connection_i": "-",
            "connection_j": "-",
            "connection_n": "1",
            "skip_if_invalid": "False",
        }
    )
    assert kwargs == {"n": "1", "condition": "i != j", "p": 0.3}


def test_build_connect_kwargs_explicit_indices() -> None:
    # explicit i/j take precedence over condition/p, and "-" means unset
    kwargs = B2Model._build_connect_kwargs(
        {
            "connection_condition": "i != j",
            "p": 0.3,
            "connection_i": "-",
            "connection_j": "i",
            "connection_n": "-",
            "skip_if_invalid": "True",
        }
    )
    assert kwargs == {"skip_if_invalid": True, "j": "i"}


def test_build_connect_kwargs_literal_index_arrays() -> None:
    # literal index/array strings are parsed into ints/lists, generator
    # expressions are passed through unchanged
    kwargs = B2Model._build_connect_kwargs(
        {
            "connection_i": "[0, 1, 2]",
            "connection_j": "3",
        }
    )
    assert kwargs == {"i": [0, 1, 2], "j": 3}

    kwargs = B2Model._build_connect_kwargs({"connection_j": "k for k in range(3)"})
    assert kwargs == {"j": "k for k in range(3)"}


def test_synapse_connect_one_to_one(
    mfx_with_synapse: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_with_synapse

    # one-to-one connectivity via a generator expression for j
    mfx.define_synapse_j("Ce", "i")
    mfx._assemble_synapse("Ce")

    ce = mfx.network["Ce"]
    assert list(ce.i) == list(range(24))
    assert list(ce.j) == list(range(24))


def test_synapse_connect_explicit_index_arrays(
    mfx_with_synapse: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_with_synapse

    # explicit connectivity given as literal index arrays
    mfx.define_synapse_i("Ce", "[0, 5, 10]")
    mfx.define_synapse_j("Ce", "[1, 2, 3]")
    mfx._assemble_synapse("Ce")

    ce = mfx.network["Ce"]
    assert list(ce.i) == [0, 5, 10]
    assert list(ce.j) == [1, 2, 3]


def test_synapse_connect_string_probability(
    mfx_with_synapse: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_with_synapse

    # deterministic index-dependent probability: connect only the upper triangle
    mfx.define_synapse_connection_probability("Ce", "1.0 * (j > i)")
    mfx._assemble_synapse("Ce")

    ce = mfx.network["Ce"]
    assert len(ce) == 24 * 23 // 2
    assert all(j > i for i, j in zip(ce.i, ce.j, strict=True))


def test_synapse_connection_fields_round_trip(
    mfx_with_synapse: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, params = mfx_with_synapse

    mfx.define_synapse_j("Ce", "i")
    mfx.define_synapse_n("Ce", "2")
    mfx.define_synapse_skip_if_invalid("Ce", "True")
    mfx._assemble_synapse("Ce")

    model = B2Model.from_dict(mfx.to_dict(), params)

    assert model.synapses["Ce"]["connection_j"] == "i"
    assert model.synapses["Ce"]["connection_n"] == "2"
    assert model.synapses["Ce"]["skip_if_invalid"] == "True"
    # one-to-one with n=2 → two synapses per source neuron
    assert len(model.network["Ce"]) == 2 * 24


def test_define_synapse_delay(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    mfx.define_synapse_delay("Ce", 1e-3 * b2.second)

    assert mfx.synapses["Ce"]["delay"] == 1e-3 * b2.second


def test_define_synapse(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
) -> None:
    taupre = 20 * b2.ms
    taupost = taupre
    gmax = 60 * b2.nS
    dApre = 6 * b2.nS
    dApost = -dApre * taupre / taupost * 1.05

    mfx, _ = mfx_parameterized

    mfx.define_neuronsubgroup_target("Pe", "P")
    mfx.define_neuronsubgroup_idxs("Pe", (0, 24))

    mfx.define_neuronsubgroup_target("Pi", "P")
    mfx.define_neuronsubgroup_idxs("Pi", (0, 24))

    mfx._assemble_neuronsubgroup("Pi")
    mfx._assemble_neuronsubgroup("Pe")

    mfx.define_parameter("taupre", taupre)

    mfx.define_parameter("taupost", taupost)

    mfx.define_parameter("gmax", gmax)

    mfx.define_parameter("dApre", dApre)

    mfx.define_parameter("dApost", dApost)

    mfx.define_synapse_connection("Ce", "i != j")
    mfx.define_synapse_connection_probability("Ce", 1.0)
    mfx.define_synapse_origin("Ce", "Pe")
    mfx.define_synapse_target("Ce", "Pi")
    mfx.define_synapse_equation(
        "Ce",
        """we : siemens
            dApre/dt = -Apre / taupre : siemens (event-driven)
            dApost/dt = -Apost / taupost : siemens (event-driven)""",
    )

    mfx.define_synapse_pre(
        "Ce",
        """ge += we
                    Apre += dApre
                    we = clip(we + Apost, -gmax, gmax)""",
    )

    mfx.define_synapse_post(
        "Ce",
        """Apost += dApost
                      we= clip(we + Apre, -gmax, gmax)""",
    )
    mfx.define_synapse_solver_method("Ce", "exponential_euler")

    mfx._assemble_synapse("Ce")

    assert len(mfx.synapses) == 1
    assert mfx.synapses["Ce"]["origin"] == "Pe"
    assert mfx.synapses["Ce"]["target"] == "Pi"
    assert mfx.synapses["Ce"]["method"] == "exponential_euler"
    ce = mfx.network["Ce"]

    # test removal and reassembly
    mfx.define_synapse_connection_probability("Ce", 0.5)
    mfx._assemble_synapse("Ce")
    assert ce is not mfx.network["Ce"]
    assert mfx.synapses["Ce"]["p"] == 0.5


def test_assemble_synapse_exception(
    mocker: MockerFixture, mfx_with_synapse: tuple[B2Model, dict[str, Any]]
) -> None:
    mocker.patch(
        "neuroseq.b2_model.b2.Synapses", side_effect=Exception("fake exception")
    )
    mfx, _ = mfx_with_synapse

    with pytest.raises(RuntimeError, match="Error in defining synapse Ce"):
        mfx._assemble_synapse("Ce")


def test_connect_synapse_exception(
    mocker: MockerFixture, mfx_with_synapse: tuple[B2Model, dict[str, Any]]
) -> None:
    mocker.patch(
        "neuroseq.b2_model.b2.Synapses.connect",
        side_effect=Exception("fake exception"),
    )

    mfx, _ = mfx_with_synapse
    with pytest.raises(RuntimeError, match="Error in connecting synapse Ce"):
        mfx._assemble_synapse("Ce")


def test_define_monitor(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    assert mfx.monitors == {}

    mfx.define_monitor_source("Traces_ge", "P")
    mfx.define_monitor_type("Traces_ge", "StateMonitor")
    mfx.define_monitor_variable("Traces_ge", "ge")

    mfx._assemble_monitor("Traces_ge")

    assert "Traces_ge" in mfx.monitors
    assert mfx.network["Traces_ge"].source == mfx.network["P"]
    assert mfx.network["Traces_ge"].record_variables == [
        "ge",
    ]


def test_define_monitor_exception(
    mocker: MockerFixture, mfx_parameterized: tuple[B2Model, dict[str, Any]]
) -> None:
    mocker.patch(
        "neuroseq.b2_model.b2.StateMonitor",
        side_effect=Exception("fake exception"),
    )

    mfx, _ = mfx_parameterized

    with pytest.raises(RuntimeError, match="Error in defining monitor M"):
        mfx.define_monitor_source("M", "G")
        mfx.define_monitor_type("M", "StateMonitor")
        mfx.define_monitor_variable("M", "v")
        mfx._assemble_monitor("M")


def test_define_init(project_ornstein: Project) -> None:
    mfx = project_ornstein

    b2.seed(4321)
    N = b2.NeuronGroup(
        5,
        """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
        """,
        method="euler",
        name="neurongroup",
    )

    N.tau = np.array([1, 0.8, 0.6, 0.3, 0.1]) * b2.second
    N.sigma = np.array([0.1, 0.31622, 0.4, 0.2, 0.88])
    N.y = 1

    M = b2.StateMonitor(N, "y", record=True)

    network = b2.Network([N, M])

    assert all(network["neurongroup"].tau == mfx.b2_model.network["neurongroup"].tau)
    assert all(
        network["neurongroup"].sigma == mfx.b2_model.network["neurongroup"].sigma
    )
    assert all(network["neurongroup"].y == mfx.b2_model.network["neurongroup"].y)


def test_ornstein_uhlenbeck(
    project_ornstein: Project, test_stim_arrays: dict[str, StimArray]
) -> None:
    b2.seed(4321)
    project_ornstein.b2_model.run_simulation(1, test_stim_arrays)

    b2.seed(4321)
    N = b2.NeuronGroup(
        5,
        """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
        """,
        method="euler",
        dt=project_ornstein.b2_model.neurongroups["neurongroup"]["dt"],
    )

    N.tau = np.array([1, 0.8, 0.6, 0.3, 0.1]) * b2.second
    N.sigma = np.array([0.1, 0.31622, 0.4, 0.2, 0.88])
    N.y = 1

    M = b2.StateMonitor(
        N,
        "y",
        record=True,
    )

    network = b2.Network([N, M])

    network.run(1 * b2.second)

    assert np.allclose(
        M.t / b2.second, project_ornstein.b2_model.network["M"].t / b2.second
    )
    assert np.allclose(M.y[1], project_ornstein.b2_model.network["M"].y[1])
    assert np.allclose(M.y[0], project_ornstein.b2_model.network["M"].y[0])


def test_ornstein_uhlenbeck_other_dt() -> None:
    eqs = """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
    """
    model = B2Model(seed=4321)
    model.define_neuronnumber("neurongroup", 5)
    model.define_dt("neurongroup", 125 * b2.ms)
    model.define_dt("neurongroup", 0.5 * b2.ms)
    model.define_solver_method("neurongroup", "euler")
    model.define_equation("neurongroup", eqs)
    model._assemble_neurongroup(name="neurongroup")

    model.define_monitor_source("M", "neurongroup")
    model.define_monitor_type("M", "StateMonitor")
    model.define_monitor_variable("M", "y")
    model.define_monitor_dt("M", 250 * b2.ms)
    model._assemble_monitor("M")

    # initial conditions
    model.define_init(
        "neurongroup", "tau", np.array([1, 0.8, 0.6, 0.3, 0.1]) * b2.second
    )
    model.define_init("neurongroup", "y", 1)
    model.define_init("neurongroup", "sigma", np.array([0.1, 0.31622, 0.4, 0.2, 0.88]))

    for n in model.initials:
        model._assemble_initial_conditions(n)

    model.must_be_rebuilt = False

    model.run_simulation(5, stim_arrays={})

    assert model.network["M"].y.shape == (5, 20)


def test_request_cancel_during_run_truncates_simulation(
    b2_model_nmda: B2Model,
) -> None:
    # Cancelling from the progress callback once the run is half complete stops
    # it early, leaving recorded data for ~half of the requested runtime.
    requested_seconds = 0.100

    def _cancel_at_halfway(elapsed, completed, start, duration) -> None:
        if completed >= 0.5:
            b2_model_nmda.request_cancel()

    # report_period=0 makes Brian2 invoke the callback every timestep, so the
    # cancel lands close to the 50% mark.
    b2_model_nmda.run_simulation(
        requested_seconds,
        stim_arrays={},
        report=_cancel_at_halfway,
        report_period=0 * b2.second,
    )

    recorded_seconds = float(b2_model_nmda.network["M"].t[-1])
    assert recorded_seconds == pytest.approx(requested_seconds / 2, abs=0.005)


def test_NMDA_model(b2_model_nmda: B2Model) -> None:
    # https://brian2.readthedocs.io/en/stable/examples/synapses.nonlinear.html
    b2.seed(4321)
    b2_model_nmda.run_simulation(0.100, stim_arrays={})

    # test model
    neuron_input = b2.NeuronGroup(
        2,
        "dv/dt = 1/(10*ms) : 1",
        threshold="v>1",
        reset="v = 0",
        method="euler",
        namespace=b2_model_nmda.parameters,
    )

    neurons = b2.NeuronGroup(
        1,
        """dv/dt = (g-v)/(10*ms) : 1
                                g : 1""",
        method="exact",
        namespace=b2_model_nmda.parameters,
    )

    S = b2.Synapses(
        neuron_input,
        neurons,
        """
                    dg_syn/dt = -a*g_syn+b*x*(1-g_syn) : 1 (clock-driven)
                    g_post = g_syn : 1 (summed)
                    dx/dt=-c*x : 1 (clock-driven)
                    w : 1 # synaptic weight
                """,
        on_pre="x += w",
        name="synapses_test",
        namespace=b2_model_nmda.parameters,
    )  # NMDA synapses

    S.connect()

    # initialize
    S.w = [1.0, 10.0]

    neuron_input.v = [0.0, 0.5]

    M = b2.StateMonitor(
        S,
        "g",
        name="Mtest",
        # If not using standalone mode, this could also simply be
        # record=np.arange(len(neuron_input) * len(neurons)),
        record=True,
    )

    Mn = b2.StateMonitor(neurons, "g", name="Mntest", record=0)

    test_network = b2.Network(name="testnetwork")
    test_network.add(neuron_input)
    test_network.add(neurons)
    test_network.add(S)
    test_network.add(M)
    test_network.add(Mn)

    assert_allclose(S.w, b2_model_nmda.network["S"].w)
    assert_allclose(neuron_input.v, b2_model_nmda.network["neuron_input"].v)

    test_network.run(100 * b2.ms)

    # assert result equality
    assert_allclose(M.g, b2_model_nmda.network["M"].g)
    assert_allclose(Mn.g, b2_model_nmda.network["Mn"].g)

    b2_model_nmda.must_be_rebuilt = True
    b2_model_nmda.run_simulation(0.100, stim_arrays={})
    assert (
        b2_model_nmda.must_be_rebuilt is False
    )  # false indicate that the model has been rebuild


def test_NMDA_model_delete(b2_model_nmda: B2Model) -> None:
    assert "neuronnumber" in b2_model_nmda.neurongroups["neuron_input"]
    b2_model_nmda.functions["neurongroups"]["neuron_input"]["neuronnumber"]["delete"]()
    assert "neuronnumber" not in b2_model_nmda.neurongroups["neuron_input"]
    assert "neuronnumber" not in b2_model_nmda.functions["neurongroups"]["neuron_input"]

    assert "on_pre" in b2_model_nmda.synapses["S"]
    b2_model_nmda.functions["synapses"]["S"]["on_pre"]["delete"]()
    assert "on_pre" not in b2_model_nmda.synapses["S"]
    assert "on_pre" not in b2_model_nmda.functions["synapses"]["S"]

    assert "source" in b2_model_nmda.monitors["Mn"]
    b2_model_nmda.functions["monitors"]["Mn"]["source"]["delete"]()
    assert "source" not in b2_model_nmda.monitors["Mn"]
    assert "source" not in b2_model_nmda.functions["monitors"]["Mn"]

    assert "w" in b2_model_nmda.initials["S"]
    b2_model_nmda.functions["initials"]["S"]["w"]["delete"]()
    assert "w" not in b2_model_nmda.initials["S"]
    assert "w" not in b2_model_nmda.functions["initials"]["S"]

    assert "Mn" in b2_model_nmda.monitors
    b2_model_nmda.functions["monitors"]["Mn"]["delete"]()
    assert "Mn" not in b2_model_nmda.monitors
    assert "Mn" not in b2_model_nmda.functions["monitors"]

    assert "neuron_input" in b2_model_nmda.neurongroups
    b2_model_nmda.functions["neurongroups"]["neuron_input"]["delete"]()
    assert "neuron_input" not in b2_model_nmda.neurongroups

    assert "S" in b2_model_nmda.synapses
    b2_model_nmda.functions["synapses"]["S"]["delete"]()
    assert "S" not in b2_model_nmda.synapses

    assert "a" in b2_model_nmda.parameters  # type: ignore
    b2_model_nmda.functions["parameters"]["a"]["delete"]()
    assert "a" not in b2_model_nmda.parameters  # type: ignore


def test_assemble_network(b2_model_nmda, monkeypatch: MonkeyPatch) -> None:
    a = 1 / (10 * b2.ms)
    b = 1 / (10 * b2.ms)
    c = 1 / (10 * b2.ms)

    model = B2Model(seed=4321)

    model.define_parameter("a", a)
    model.define_parameter("b", b)
    model.define_parameter("c", c)

    model.define_neuronnumber("neuron_input", 2)
    model.define_solver_method("neuron_input", "euler")
    model.define_equation("neuron_input", "dv/dt = 1/(10*ms) : 1")
    model.define_threshold("neuron_input", "v>1")
    model.define_reset("neuron_input", "v=0")

    model.define_neuronnumber("neurons", 1)
    model.define_equation(
        "neurons",
        """dv/dt = (g-v)/(10*ms) : 1
                            g : 1""",
    )
    model.define_solver_method("neurons", "exact")

    synapse_eq = """
        dg_syn/dt = -a*g_syn+b*x*(1-g_syn) : 1 (clock-driven)
        g_post = g_syn : 1 (summed)
        dx/dt=-c*x : 1 (clock-driven)
        w : 1 # synaptic weight
    """

    model.define_synapse_equation("S", synapse_eq)
    model.define_synapse_origin("S", "neuron_input")
    model.define_synapse_target("S", "neurons")
    model.define_synapse_pre("S", "x += w")

    model.define_neuronsubgroup_target("input_subgroup", "neuron_input")
    model.define_neuronsubgroup_idxs("input_subgroup", 1)

    model.define_monitor_source("M", "S")
    model.define_monitor_type("M", "StateMonitor")
    model.define_monitor_variable("M", "g")

    model.define_monitor_source("Mn", "neurons")
    model.define_monitor_type("Mn", "StateMonitor")
    model.define_monitor_variable("Mn", "g")

    model.define_init("S", "w", [1.0, 10.0])
    model.define_init("neuron_input", "v", [0.0, 0.5])

    model.assemble_network()

    assert model.neurongroups == b2_model_nmda.neurongroups
    assert model.synapses == b2_model_nmda.synapses
    assert model.monitors == b2_model_nmda.monitors
    assert model.parameters == b2_model_nmda.parameters

    assert sorted(list(model.network.get_states().keys())) == sorted(
        list(b2_model_nmda.network.get_states().keys())
    )

    assert "M" in [o.name for o in model.get_network_objects()]
    assert "neuron_input" in [o.name for o in model.get_network_objects()]
    assert "S" in [o.name for o in model.get_network_objects()]

    nrg = model.get_neurongroup("neurons")

    assert nrg["definition"] == model.neurongroups["neurons"]
    assert nrg["object"] is model.network["neurons"]

    assert model.get_variables_of("neurons") == list(model.network["neurons"].variables)
    assert model.get_equation_names_of("neurons") == list(
        model.network["neurons"].equations.names
    )

    with pytest.raises(
        KeyError,
        match="Error, name X is neither synapse nor neurongroup and hence does not have equations to get variables from",
    ):
        model.get_equation_names_of("X")

    def raise_error(x: Any) -> None:
        raise RuntimeError("Dummy")

    monkeypatch.setattr(b2.Network, "__getattribute__", raise_error)

    with pytest.raises(
        RuntimeError,
        match="Error, something went wrong when getting equation variables for neurons",
    ):
        model.get_equation_names_of("neurons")


def test_assemble_network_clears_monitor_lookup_caches(
    b2_model_nmda: B2Model,
) -> None:
    assert b2_model_nmda.get_simulation_trace_names()
    assert b2_model_nmda.get_simulation_synapse_names()
    assert b2_model_nmda.trace_name_to_monitor
    assert b2_model_nmda.synapse_name_to_monitor

    b2_model_nmda.assemble_network()

    assert b2_model_nmda.trace_name_to_monitor == {}
    assert b2_model_nmda.spike_name_to_monitor == {}
    assert b2_model_nmda.synapse_name_to_monitor == {}


def test_hdf5(
    tmp_path: Path,
    project_ornstein: Project,
) -> None:
    project_ornstein.run_simulation(10)

    project_ornstein.b2_model.save_to_h5(tmp_path / "data.h5")

    data = None

    assert len(project_ornstein.b2_model.checkpoints) > 0

    with h5py.File(Path(tmp_path, "data.h5"), "r") as h5file:
        data = h5file["M"]["y"][:]

    assert np.allclose(data, project_ornstein.b2_model.network["M"].y)

    with pytest.raises(ValueError, match="Path fakepath/data.h5 does not exist"):
        project_ornstein.b2_model.save_to_h5("fakepath/data.h5")

    project_ornstein.b2_model.monitors = {}

    with pytest.raises(
        RuntimeError, match="Error, cannot save data when no monitors are defined"
    ):
        project_ornstein.b2_model.save_to_h5(tmp_path / "data.h5")

    project_ornstein.b2_model.assemble_network()

    assert len(project_ornstein.b2_model.checkpoints) == 0


def test_logger(caplog: pytest.LogCaptureFixture, project_ornstein: Project) -> None:
    caplog.set_level(logging.INFO, logger="neuroseq.b2_model")
    project_ornstein.run_simulation(1)

    logs = [record.message for record in caplog.records]
    # Brian2 may emit "object is getting deleted" warnings during setup that
    # land in caplog before our INFO message; check membership rather than
    # asserting on logs[0].
    assert any("run the system for 1" in msg for msg in logs)


def test_store_restore(project_ornstein: Project) -> None:
    sim_time = 10 * b2.ms
    assert len(project_ornstein.b2_model.checkpoints) == 0
    project_ornstein.run_simulation(sim_time / b2.second)
    # default initial state checkpoint is made on first simulation if no other checkpoints are made
    assert len(project_ornstein.b2_model.checkpoints) == 1
    assert (
        project_ornstein.b2_model.network["M"].t[-1]
        == sim_time - project_ornstein.b2_model.neurongroups["neurongroup"]["dt"]
    )
    # restore initial state
    project_ornstein.b2_model.restore(project_ornstein.b2_model.checkpoints[0])
    assert len(project_ornstein.b2_model.network["M"].t) == 0
    # store two new checkpoints
    project_ornstein.run_simulation(sim_time / b2.second)
    assert (
        project_ornstein.b2_model.network["M"].t[-1]
        == sim_time - project_ornstein.b2_model.neurongroups["neurongroup"]["dt"]
    )
    project_ornstein.b2_model.store("t1")
    assert len(project_ornstein.b2_model.checkpoints) == 2
    project_ornstein.run_simulation(sim_time / b2.second)
    assert (
        project_ornstein.b2_model.network["M"].t[-1]
        == 2 * sim_time - project_ornstein.b2_model.neurongroups["neurongroup"]["dt"]
    )
    project_ornstein.b2_model.store("t2")
    assert len(project_ornstein.b2_model.checkpoints) == 3
    # restore second checkpoint
    project_ornstein.b2_model.restore("t1")
    assert (
        project_ornstein.b2_model.network["M"].t[-1]
        == sim_time - project_ornstein.b2_model.neurongroups["neurongroup"]["dt"]
    )
    # overwrite existing checkpoint
    assert len(project_ornstein.b2_model.checkpoints) == 3
    project_ornstein.b2_model.store("t2")
    assert len(project_ornstein.b2_model.checkpoints) == 3
    # restore initial state
    project_ornstein.b2_model.restore(project_ornstein.b2_model.checkpoints[0])
    assert len(project_ornstein.b2_model.network["M"].t) == 0
    # restore non-existing checkpoint
    with pytest.raises(ValueError, match=".*'idontexist'.*"):
        project_ornstein.b2_model.restore("idontexist")


def test_default_group_addition(project_ornstein) -> None:
    project_ornstein.b2_model.define_default_neurongroup()
    assert project_ornstein.b2_model.neurongroups["new_neurongroup"] == {
        "neuronnumber": 0,
        "model": "dx/dt = y : 1",
        "method": "-",
        "dt": 10 * b2.ms,
        "reset": "0",
        "th": "0",
        "refractory": "0",
    }

    project_ornstein.b2_model.define_default_synapse()
    assert project_ornstein.b2_model.synapses["new_synapse"] == {
        "origin": "-",
        "target": "-",
        "model": "dx/dt = y : 1",
        "method": "-",
        "dt": 10 * b2.ms,
        "delay": 0 * b2.ms,
        "connection_condition": "x!=y",
        "p": 0.0,
        "connection_i": "-",
        "connection_j": "-",
        "connection_n": "1",
        "skip_if_invalid": "False",
        "on_pre": "x=y",
        "on_post": "x=y",
    }

    project_ornstein.b2_model.define_default_monitor()
    assert project_ornstein.b2_model.monitors["new_monitor"] == {
        "type": "-",
        "source": "-",
        "variable": "-",
        "dt": None,
    }

    project_ornstein.b2_model.define_default_parameter()
    assert project_ornstein.b2_model.parameters["new_parameter"] is None

    project_ornstein.b2_model.define_default_init()
    assert project_ornstein.b2_model.initials["group"]["value"] is None


def test_prune_orphaned_definitions_on_assemble() -> None:
    # Reviewer scenario: removing the inhibitory neuron group left the
    # inhibitory synapse (origin=Pi) and its initial condition dangling, so the
    # next assemble crashed on a stale reference. assemble_network must prune
    # such orphans instead.
    from neuroseq.example_projects import hh_neuron_with_plasticity

    model = hh_neuron_with_plasticity().b2_model
    model.functions["neurongroups"]["Pi"]["delete"]()
    assert "Ci" in model.synapses
    assert "Ci" in model.initials

    model.assemble_network()

    assert "Pi" not in model.neurongroups
    assert "Ci" not in model.synapses  # origin gone -> synapse pruned
    assert "Ci" not in model.initials  # its initial condition pruned too
    assert "Ce" in model.synapses  # excitatory synapse untouched
    assert "Ci" not in [obj.name for obj in model.network.objects]
