import logging
from logging import Logger
from pathlib import Path
from typing import List
from typing import Union

import brian2 as b2
import h5py


class B2Model:
    def _set_up_logger(self, loglevel: str) -> None:
        self.logger = Logger("Model")
        self.logger.setLevel(loglevel)

        handler = logging.FileHandler(Path(self.path, "logfile.log"))
        handler.setLevel(loglevel)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _update_nested_dict(self, original, updates):
        for key, value in updates.items():
            if (
                isinstance(value, dict)
                and key in original
                and isinstance(original[key], dict)
            ):
                self._update_nested_dict(original[key], value)
            else:
                original[key] = value

    def __init__(self, savepath: str, loglevel: str = "INFO", seed: int = 123):
        b2.seed(seed)

        if Path(savepath).exists() is False:
            raise ValueError(f"Path {savepath} does not exist")

        self.path = savepath
        self._set_up_logger(loglevel)
        self.network = b2.Network(name="b2Network")
        self.parameters = None
        self.neuron_subgroups = {}
        self.neurongroups = {}
        self.synapses = {}
        self.monitors = {}
        self.dt = None
        self.solver = None
        self.logger.info("Model creation complete")

    def define_equation(self, name: str, line: str) -> None:
        if ":" not in line:
            message = "Equation must have a colon that separates equation and unit"
            self.logger.critical(message)
            raise ValueError(message)

        if "=" not in line:
            message = "Equation must have an equal sign that separates variable and expression"
            self.logger.critical(message)
            raise ValueError(message)

        if name not in self.neurongroups:
            self.neurongroups[name] = {}

        if "model" in self.neurongroups[name]:
            self.neurongroups[name]["model"] += line + "\n"
        else:
            self.neurongroups[name]["model"] = line + "\n"

        self.logger.info("Equation defined")
        self.logger.debug(f"full equation: {line}")

    def define_dt(self, dt: b2.Quantity) -> None:
        if not isinstance(dt, b2.Quantity):
            message = "dt must be a Quantity"
            self.logger.critical(message)
            raise ValueError(message)

        self.dt = dt
        b2.defaultclock.dt = self.dt
        self.network.dt = self.dt
        self.logger.info(f"dt defined: {dt}")

    def define_parameter(self, name: str, value: Union[str, b2.Quantity]) -> None:
        if self.parameters is None:
            self.parameters = {}

        self.parameters[name] = value

        self.logger.info(f"Parameter defined: {name}")
        self.logger.debug(f"Parameter value: {value}")

    def define_threshold(self, name: str, th: Union[b2.Quantity, str]) -> None:
        if not isinstance(th, b2.Quantity) and not isinstance(th, str):
            message = "threshold value must be a Quantity or a string"
            self.logger.critical(message)
            raise ValueError(message)
        # self.neurongroups.update({name: {"th": th, }, })
        self._update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "th": th,
                },
            },
        )
        self.logger.info(f"Threshold defined: {th}")

    def define_solver_method(self, name: str, method: str) -> None:
        self._update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "method": method,
                },
            },
        )
        self.logger.info(f"Solver method defined: {method}")

    def define_refractory(self, name: str, value: b2.Quantity) -> None:
        if not isinstance(value, b2.Quantity):
            message = "Refractory period must be a Quantity"
            self.logger.critical(message)
            raise ValueError(message)

        self._update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "refractory": value,
                },
            },
        )
        self.logger.info(f"Refractory period defined: {value}")

    def define_neuronnumber(self, name: str, n: int) -> None:
        if not isinstance(n, int):
            message = "Neuron number must be an integer"
            self.logger.critical(message)
            raise ValueError(message)

        self.logger.info(f"Neuron number defined: {n}")
        self._update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "n": n,
                },
            },
        )

    def define_neuron_subgroup(
        self, target_name: str, name: str, idxs: Union[slice, int, List]
    ) -> None:
        sng = None

        if isinstance(idxs, tuple):
            sng = self.network[target_name][idxs[0] : idxs[1]]
        else:
            sng = self.network[target_name][idxs]

        self.logger.info(f"Neuron subgroup defined: {name}")
        self.network.add(sng)
        self.neuron_subgroups[name] = {
            "idx": idxs,
            "name": name,
            "name_in_network": sng.name,
        }

    def define_synapse_target(self, name: str, target: str):
        self.syanpses.update(
            {
                name: {
                    "target": target,
                }
            }
        )

    def define_synapse_origin(self, name: str, origin: str):
        self.syanpses.update(
            {
                name: {
                    "origin": origin,
                }
            }
        )

    def define_synapse_equation(self, name: str, line: str):
        if ":" not in line:
            message = "Equation must have a colon that separates equation and unit"
            self.logger.critical(message)
            raise ValueError(message)

        if "=" not in line:
            message = "Equation must have an equal sign that separates variable and expression"
            self.logger.critical(message)
            raise ValueError(message)

        if name not in self.synapses:
            self.synapses[name] = {}

        if "model" in self.neurongroups[name]:
            self.synapses[name]["model"] += line + "\n"
        else:
            self.synapses[name]["model"] = line + "\n"

    def define_synapse_pre(self, name: str, line: str):
        if ":" not in line:
            message = "Equation must have a colon that separates equation and unit"
            self.logger.critical(message)
            raise ValueError(message)

        if "=" not in line:
            message = "Equation must have an equal sign that separates variable and expression"
            self.logger.critical(message)
            raise ValueError(message)

        if name not in self.synapses or "on_pre" not in self.synapses[name]:
            self.synapses.update(
                {
                    name: {
                        "on_pre": line + "\n",
                    },
                },
            )
        else:
            self.synapses[name]["on_pre"] += line + "\n"

    def define_synapse_post(self, name: str, line: str):
        if ":" not in line:
            message = "Equation must have a colon that separates equation and unit"
            self.logger.critical(message)
            raise ValueError(message)

        if "=" not in line:
            message = "Equation must have an equal sign that separates variable and expression"
            self.logger.critical(message)
            raise ValueError(message)

        if name not in self.synapses or "on_post" not in self.synapses[name]:
            self.synapses.update(
                {
                    name: {
                        "on_post": line + "\n",
                    },
                },
            )
        else:
            self.synapses[name]["on_post"] += line + "\n"

    def define_synapse_connection(self, name: str, connection_condition: str):
        self.syanpses.update(
            {
                name: {
                    "connection_condition": connection_condition,
                },
            }
        )

    def define_synapse_connection_probablitiy(self, name: str, p: float):
        self.syanpses.update(
            {
                name: {
                    "p": p,
                },
            }
        )

    def assemble_synapse(
        self,
        name: str,
    ) -> None:
        try:
            args = self.synapses[name]

            _origin = self.network.__getitem__(
                self.neuron_subgroups[args["origin"]]["name_in_network"]
            )
            _target = self.network.__getitem__(
                self.neuron_subgroups[args["target"]]["name_in_network"]
            )

            synapse = b2.Synapses(
                _origin,
                target=_target,
                model=args["model"],
                on_pre=args["on_pre"],
                on_post=args["on_post"],
                name=name,
                namespace=self.parameters,
            )

        except Exception as e:
            message = f"Error in defining synapse {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        try:
            synapse.connect(args["connection_condition"], p=args["p"])
        except Exception as e:
            message = f"Error in connecting synapse {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        self.network.add(synapse)

    def define_monitor(
        self, name: str, cls, group: str, variable: str = None, record: bool = True
    ) -> None:
        monitor = None
        self.logger.info(f"adding monitor '{name}'")

        try:
            monitor = cls(
                self.network[group],
                variable,
                record=record,
                name=name,
            )
        except Exception as e:
            message = f"Error in defining monitor {name}: {e}"
            self.logger.critcal(message)
            raise RuntimeError(message) from e

        self.network.add(monitor)
        self.monitors[name] = {
            "name": name,
            "type": cls.__name__,
            "source": group,
            "record": record,
        }

    def define_reset(self, name: str, reset: str) -> None:
        self._update_nested_dict(
            self.neuron_subgroups,
            {
                name: {
                    "reset": reset,
                },
            },
        )

    def assemble_neurongroup(
        self,
        name: str = "neurongroup",
    ) -> None:
        ng = None
        self.logger.info(f"assembling neurongroup '{name}'")
        try:
            ng = b2.NeuronGroup(
                self.neurongroups[name].get("n", None),
                self.neurongroups[name].get("model", None),
                threshold=self.neurongroups[name].get("th", None),
                refractory=self.neurongroups[name].get("refractory", None),
                method=self.neurongroups[name].get("method", None),
                namespace=self.parameters,
                reset=self.neurongroups[name].get("reset", None),
                name=name,
            )

        except Exception as e:
            message = f"Error in assembling neuron group: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        self.network.add(ng)

    def save_to_h5(self) -> None:
        self.logger.info(f"save data to hdf5 at {Path(self.path, "data.h5")}")

        with h5py.File(Path(self.path, "data.h5"), "w") as h5file:
            for key, value in self.monitors.items():
                self.logger.debug(f"save data of monitor {key}")
                monitor = self.network[key]
                for v in monitor.record_variables:
                    h5file.create_dataset(
                        v, shape=getattr(monitor, v).shape, data=getattr(monitor, v)
                    )

    def run_simulation(self, tfinal: b2.Quantity, save: bool = True) -> None:
        self.network.run(tfinal, report="text")
        self.logger.info(f"run the system for {tfinal} with save = {save}")
        if save:
            self.save_to_h5()
