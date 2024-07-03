import logging
from importlib import import_module
from pathlib import Path
from typing import Any

import brian2 as b2
import yaml


class Reconstructor:
    """
    Class that reconstructs a B2Model from its serialized version
    """

    def __init__(self, path: str | Path) -> None:
        """
        Construct a new `Reconstructor` object.

        Parameters
        ----------
        path : str
            Path to the yaml file that contains the model
        """
        self.logger = logging.getLogger(__name__)
        with open(path) as file:
            self.loaded_data = yaml.load(file, Loader=yaml.UnsafeLoader)

    def _reconstruct_dynamic_object(
        self,
        model: Any,
        assembler: str,
        name: str,
        data_of_group: dict[str, Any],
        augmented: list[Any],
    ) -> None:
        """
        Reconstruct synapses and neurongroups from their respective yaml nodes

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
                message = f"Error when defining {key} for object{name}: {e}"
                raise RuntimeError(message) from e

        # assemble object
        try:
            getattr(model, assembler)(name)
        except Exception as e:
            message = f"Error when assembing neurongroup {name}: {e}"
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
                        f"Something went wrong when setting attribute {key}: {e}"
                    ) from e

    def reconstruct_neurongroups(
        self, model: Any, name: str, data_of_group: dict[str, Any]
    ) -> None:
        """
        Reconstruct a neurongroup from a yamlnode

        Parameters
        ----------
        model : B2Model
            Model to construct the neurongroup for
        name : str
            name of the neurongroup to be reconstructed
        data_of_group : dict[str, Any]
            yaml node that contains the necessary data for reconstruction
        """
        augmented = list(
            zip(
                ["N", "model", "th", "refractory", "reset", "method"],
                [
                    model.define_neuronnumber,
                    model.define_equation,
                    model.define_threshold,
                    model.define_refractory,
                    model.define_reset,
                    model.define_solver_method,
                ],
            )
        )

        self._reconstruct_dynamic_object(
            model,
            "assemble_neurongroup",
            name,
            data_of_group,
            augmented,
        )

    def reconstruct_synapses(
        self, model: Any, name: str, data_of_group: dict[str, Any]
    ) -> None:
        """
        Reconstruct a synapse from a YAML node for a certain model

        Parameters
        ----------
        model : B2Model
            Model to reconstruct the synapse for
        name : str
            Name of the synapse to be reconstructed
        data_of_group : dict[str, Any]
            YAML node to write it for
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
                    model.define_solver_method,
                ],
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
        Reconstruct a monitor for a model from a yaml node

        Parameters
        ----------
        model : B2Model
            Model to reconstruct the monitor for
        name : str
            Name of the monitor to reconstruct
        data_of_group : dict[str, Any]
            YAML node to reconstruct the monitor from
        """
        monitortype = getattr(import_module("brian2"), data_of_group["type"])
        model.define_monitor(
            data_of_group["name"],
            monitortype,
            data_of_group["source"],
            data_of_group["variable"],
        )

    def reconstruct(self, model: Any) -> None:
        """
        Reconstruct a B2Model from a yaml node. The model is typically given in a default constructed state
        as an argument, but this is not mandatory and it can contain other elements or can be modified afterwards

        Parameters
        ----------
        model : B2Model
            Model object to use for reconstruction
        """

        model.seed = self.loaded_data["seed"]

        b2.seed(self.loaded_data["seed"])

        if "timestep" in self.loaded_data:
            model.define_dt(self.loaded_data["timestep"])

        # reconstruct parameters, timestep, seed.
        model.parameters = self.loaded_data["parameters"]

        # reconstruct neurongroups
        for n in self.loaded_data["objects"]["neurongroups"]:
            self.reconstruct_neurongroups(model, n, self.loaded_data[n])

        # reconstruct synapses
        for s in self.loaded_data["objects"]["synapses"]:
            self.reconstruct_synapses(model, s, self.loaded_data[s])

        # reconstruct monitors
        for m in self.loaded_data["objects"]["monitors"]:
            self.reconstruct_monitors(model, m, self.loaded_data[m])
