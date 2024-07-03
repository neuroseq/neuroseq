from importlib import import_module
from pathlib import Path
from typing import Any

import brian2 as b2
import yaml


class Reconstructor:
    """
    _summary_
    """

    def __init__(self, path: str | Path) -> None:
        """
        _summary_

        Parameters
        ----------
        path : str
            _description_
        """
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
        _summary_

        Parameters
        ----------
        model : B2Model
            _description_
        assembler : str
            _description_
        name : str
            _description_
        data_of_group : dict[str, Any]
            _description_
        augmented : list[tuple[str, Callable]]
            _description_

        Raises
        ------
        RuntimeError
            _description_
        RuntimeError
            _description_
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
                    # when the variable is read-only we can except it because it is constructed from others
                    # if "is read-only" in str(e):
                    continue
                    # else:
                except Exception as e:
                    raise RuntimeError(
                        f"Something went wrong when setting attribute {key}: {e}"
                    ) from e

    def reconstruct_neurongroups(
        self, model: Any, name: str, data_of_group: dict[str, Any]
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        model : B2Model
            _description_
        name : str
            _description_
        data_of_group : dict[str, Any]
            _description_
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
        _summary_

        Parameters
        ----------
        model : B2Model
            _description_
        name : str
            _description_
        data_of_group : dict[str, Any]
            _description_
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
        _summary_

        Parameters
        ----------
        model : B2Model
            _description_
        name : str
            _description_
        data_of_group : dict[str, Any]
            _description_
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
        _summary_

        Parameters
        ----------
        model : B2Model
            _description_

        Returns
        -------
        b2.Network
            _description_
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
