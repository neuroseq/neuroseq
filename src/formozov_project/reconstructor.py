from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from brian2 import seed

from formozov_project import b2json


class Reconstructor:
    """
    Class that reconstructs a B2Model from its serialized version.
    """

    def __init__(self, data_or_path: str | Path | dict[Any, Any]) -> None:
        """
        Construct a new `Reconstructor` object.

        Parameters
        ----------
        path : str
            Path to the json file that contains the model or a preloaded dictionary
        """
        self.logger = logging.getLogger(__name__)
        if isinstance(data_or_path, dict):
            self.loaded_data = data_or_path
        else:
            with open(data_or_path) as file:
                self.loaded_data = b2json.load(file)

    def _reconstruct_dynamic_object(
        self,
        model: Any,
        assembler: str,
        name: str,
        data_of_group: dict[str, Any],
        augmented: list[tuple[str, Callable]],
    ) -> None:
        """
        Reconstruct synapses and neurongroups from their respective json nodes

        Parameters
        ----------
        model : B2Model
            Model to add the object to
        assembler : str
            Function to assemble the object with: `assemble_neurongroup` or `assemble_synapse`
        name : str
            Name of the object to be assembled
        data_of_group : dict[str, Any]
            The data to reconstruct the model from
        augmented : list[tuple[str, Callable]]
            List of name, function pairs to set attributes from `data_of_group`.

        Raises
        ------
        RuntimeError
            When something goes wrong in assembling the object
        RuntimeError
            When something goes wrong in setting an attribute of the assembled object
        """

        # rebuild the neurongroup
        for key, function in augmented:
            if key not in data_of_group:
                continue

            try:
                function(name, data_of_group[key])
            except Exception as e:
                message = f"Error when defining {key} for object {name}: {e}"
                raise RuntimeError(message) from e

        # assemble object
        try:
            getattr(model, assembler)(name)
        except Exception as e:
            message = f"Error when assembling object {name}: {e}"
            raise RuntimeError(message) from e

        excluded = [n[0] for n in augmented]

        # there is N and n, one defined by default the other one merely for the record, and we do not need both
        excluded.extend(
            [
                "n",
            ]
        )

        # set object variables
        for key, value in data_of_group.items():
            if key in excluded:
                continue
            else:
                try:
                    setattr(model.network[name], key, value)
                except TypeError:
                    # when the variable is read-only brian2 throws a TypeError.
                    # We can then continue because the variable is automatically constructed
                    continue
                except Exception as e:
                    raise RuntimeError(
                        f"Something went wrong when setting attribute {key} in object {name}: {e}"
                    ) from e

    def reconstruct_neurongroups(
        self, model: Any, name: str, data_of_group: dict[str, Any]
    ) -> None:
        """
        Reconstruct a neurongroup from a jsonnode

        Parameters
        ----------
        model : B2Model
            Model to construct the neurongroup for
        name : str
            name of the neurongroup to be reconstructed
        data_of_group : dict[str, Any]
            json node that contains the necessary data for reconstruction
        """
        augmented = list(
            zip(
                ["N", "n", "model", "th", "refractory", "reset", "method"],
                [
                    model.define_neuronnumber,
                    model.define_neuronnumber,
                    model.define_equation,
                    model.define_threshold,
                    model.define_refractory,
                    model.define_reset,
                    model.define_solver_method,
                ],
                strict=True,
            )
        )

        self._reconstruct_dynamic_object(
            model,
            "assemble_neurongroup",
            name,
            data_of_group,
            augmented,
        )

    def reconstruct_neuron_subgroup(
        self, model: Any, name: str, data_of_group: dict[str, Any]
    ) -> None:
        """
        Reconstruct a neuron subgroup from stored data. Requires that the
        target neurongroup has been construted first

        Parameters
        ----------
        model : Any
            Model to reconstruct the neuron subgroup in
        name : str
            Name of the subgroup to be constructed
        data_of_group : dict[str, Any]
            dict containing the data to construct the subgroup
        """
        model.define_neuronsubgroup_target(name, data_of_group["target"])
        model.define_neuronsubgroup_idxs(
            name, (data_of_group["start"], data_of_group["stop"])
        )
        model.assemble_neuronsubgroup(name)

    def reconstruct_synapses(
        self, model: Any, name: str, data_of_group: dict[str, Any]
    ) -> None:
        """
        Reconstruct a synapse from a json node for a certain model

        Parameters
        ----------
        model : B2Model
            Model to reconstruct the synapse for
        name : str
            Name of the synapse to be reconstructed
        data_of_group : dict[str, Any]
            json node to reconstruct the synapse from
        """
        augmented = list(
            zip(
                [
                    "model",
                    "origin",
                    "target",
                    "on_pre",
                    "on_post",
                    "connection_condition",
                    "p",
                    "method",
                ],
                [
                    model.define_synapse_equation,
                    model.define_synapse_origin,
                    model.define_synapse_target,
                    model.define_synapse_pre,
                    model.define_synapse_post,
                    model.define_synapse_connection,
                    model.define_synapse_connection_probablitiy,
                    model.define_synapse_solver_method,
                ],
                strict=True,
            )
        )

        self._reconstruct_dynamic_object(
            model,
            "assemble_synapse",
            name,
            data_of_group,
            augmented,
        )

    def reconstruct_monitors(
        self, model: Any, name: str, data_of_group: dict[str, Any]
    ) -> None:
        """
        Reconstruct a monitor for a model from a json node

        Parameters
        ----------
        model : B2Model
            Model to reconstruct the monitor for
        name : str
            Name of the monitor to reconstruct
        data_of_group : dict[str, Any]
            json node to reconstruct the monitor from
        """
        model.define_monitor_type(data_of_group["name"], data_of_group["type"])
        model.define_monitor_source(data_of_group["name"], data_of_group["source"])
        model.define_monitor_variable(data_of_group["name"], data_of_group["variable"])

        model.assemble_monitor(data_of_group["name"])

    def reconstruct(
        self, model: Any, report: Callable[[int], None] | None = None
    ) -> None:
        """
        Reconstruct a B2Model from a json node. The model is typically given in a default constructed state
        as an argument, but this is not mandatory and it can contain other elements or can be modified afterwards

        Parameters
        ----------
        model : B2Model
            Model object to use for reconstruction
        report : Callable[[int],None], optional
            function to report progress, receives percentage complete as integer from 0 to 100
        """
        model.seed = self.loaded_data["seed"]

        seed(self.loaded_data["seed"])

        if "timestep" in self.loaded_data:
            model.define_dt(self.loaded_data["timestep"])

        # reconstruct parameters, timestep, seed.
        model.parameters = self.loaded_data["parameters"]
        if report is not None:
            report(30)

        # reconstruct neurongroups
        for n in self.loaded_data["objects"]["neurongroups"]:
            if all(
                k not in self.loaded_data[n] for k in ["start", "stop"]
            ):  # reconstruct full groups first...
                self.reconstruct_neurongroups(model, n, self.loaded_data[n])
        if report is not None:
            report(40)

        for n in self.loaded_data["objects"]["neurongroups"]:
            if all(
                k in self.loaded_data[n] for k in ["start", "stop"]
            ):  # ... then define the subgroups
                self.reconstruct_neuron_subgroup(model, n, self.loaded_data[n])
        if report is not None:
            report(50)

        # reconstruct synapses
        for s in self.loaded_data["objects"]["synapses"]:
            self.reconstruct_synapses(model, s, self.loaded_data[s])
        if report is not None:
            report(60)

        # reconstruct monitors
        for m in self.loaded_data["objects"]["monitors"]:
            self.reconstruct_monitors(model, m, self.loaded_data[m])
        if report is not None:
            report(70)

        model.initials = self.loaded_data["initials"]
        model.assemble_all_initial_conditions()
        if report is not None:
            report(85)
