from typing import Any

import brian2 as b2
import yaml

from formozov_project.b2_model import B2Model


class Reconstructor:
    """
    _summary_
    """

    def __init__(self, path: str) -> None:
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
        model: B2Model,
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

            if key == "N":
                # N is stored as an array, but we need it as a scalar
                data_of_group[key] = data_of_group[key][0]

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

        # set object variables
        for key, value in data_of_group.items():
            if key in augmented:
                continue
            else:
                setattr(model.network[name], key, value)

    def reconstruct_neurongroups(
        self, model: B2Model, name: str, data_of_group: dict[str, Any]
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
                ["N", "model", "threshold", "refractory", "reset", "method"],
                [
                    model.define_neuronnumber,
                    model.define_equation,
                    model.define_threshold,
                    model.define_refractory,
                    model.define_reset,
                ],
            )
        )

        self._reconstruct_dynamic_object(
            model, "assemble_neurongroup", name, data_of_group, augmented
        )

    def reconstruct_synapses(
        self, model: B2Model, name: str, data_of_group: dict[str, Any]
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
                ],
                [
                    model.define_synapse_equation,
                    model.define_synapse_origin,
                    model.define_synapse_target,
                    model.define_synapse_pre,
                    model.define_synapse_post,
                    model.define_synapse_connection,
                    model.define_synapse_connection_probablitiy,
                ],
            )
        )

        self._reconstruct_dynamic_object(
            model, "assemble_synapse", name, data_of_group, augmented
        )

    def reconstruct_monitors(
        self, model: B2Model, name: str, data_of_group: dict[str, Any]
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
        model.define_monitor(
            data_of_group[name]["name"],
            data_of_group[name]["type"],
            data_of_group[name]["source"],
            data_of_group[name]["record"],
        )

    def reconstruct(self, model: B2Model) -> None:
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

        # reconstruct neurongroups
        for n in self.loaded_data["objects"]["neurongroups"]:
            self.reconstruct_neurongroups(model, n, self.loaded_data[n])

        # reconstruct synapses
        for s in self.loaded_data["objects"]["synapses"]:
            self.reconstruct_synapses(model, s, self.loaded_data[s])

        # reconstruct monitors
        for m in self.loaded_data["objects"]["monitors"]:
            self.reconstruct_monitors(model, m, self.loaded_data[m])

        # reconstruct parameters, timestep, seed.
        model.parameters = self.loaded_data["parameters"]

        model.define_dt(self.loaded_data["timestep"])

        model.seed = self.loaded_data["seed"]
        b2.seed(self.loaded_data["seed"])
