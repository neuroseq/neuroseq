import logging
from logging import Logger
from pathlib import Path
from typing import List
from typing import Union

import brian2 as b2
import h5py
import yaml


class B2Model:
    def _set_up_logger(self, loglevel: str) -> None:
        self.logger = Logger("Model")
        self.logger.setLevel(loglevel)

        handler = logging.FileHandler(Path(self.path, "logfile.log"))
        handler.setLevel(loglevel)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )  # TODO: check how this looks like
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def __init__(self, savepath: str, loglevel: str = "INFO"):
        if Path(savepath).exists() is False:
            raise ValueError(f"Path {savepath} does not exist")

        self.path = savepath
        self._set_up_logger(loglevel)
        self.network = b2.Network([], name="b2Network")
        self.equation = ""
        self.parameters = {}
        self.neuron_subgroups = {}
        self.synapses = {}
        self.monitors = {}
        self.dt = None
        self.th = None
        self.solver = None
        self.refractory = None
        self.n = None
        self.logger.info("Model creation complete")

    def define_equation(self, line: str) -> None:
        if ":" not in line:
            raise ValueError(
                "Equation must have a colon that separates equation and unit"
            )

        if "=" not in line:
            raise ValueError(
                "Equation must have an equal sign that separates variable and expression"
            )

        self.equation += line + "\n"

        e = b2.Equations(self.equation)

        e.check_identifiers()
        self.logger.info("Equation defined")
        self.logger.debug(f"full equation: {line}")

    def define_dt(self, dt: b2.Quantity) -> None:
        if not isinstance(dt, b2.Quantity):
            raise ValueError("dt must be a Quantity")
        self.dt = dt

        self.logger.info(f"dt defined: {dt}")

    def define_parameter(self, name: str, value: Union[str, b2.Quantity]) -> None:
        if not isinstance(value, b2.Quantity) and not isinstance(value, str):
            raise ValueError("Parameter value must be a Quantity or a string")

        self.parameters[name] = value

        self.logger.info(f"Parameter defined: {name}")
        self.logger.debug(f"Parameter value: {value}")

    def define_threshold(self, th: Union[b2.Quantity, str]) -> None:
        if not isinstance(th, b2.Quantity) and not isinstance(th, str):
            raise ValueError("threshold value must be a Quantity or a string")
        self.th = th
        self.logger.info(f"Threshold defined: {th}")

    def define_solver_method(self, method: str) -> None:
        self.solver = method
        self.logger.info(f"Solver method defined: {method}")

    def define_refractory(self, value: b2.Quantity) -> None:
        if not isinstance(value, b2.Quantity):
            raise ValueError("Refractory period must be a Quantity")
        self.refractory = value
        self.logger.info(f"Refractory period defined: {value}")

    def define_neuronnumber(self, n: int) -> None:
        if not isinstance(n, int):
            raise ValueError("Neuron number must be an integer")
        self.logger.info(f"Neuron number defined: {n}")
        self.n = n

    def define_neuron_subgroup(self, name: str, idxs: Union[slice, int, List]) -> None:
        if self.self.network["neurongroup"] is None:
            self.logger.info("ValueError: Neurongroup not defined yet")
            raise ValueError("Neurongroup not defined yet")
        else:
            sng = None

            if isinstance(idxs, tuple):
                sng = self.network["neurongroup"][idxs[0] : idxs[1]]
            else:
                sng = self.network["neurongroup"][idxs]
            self.logger.info(f"Neuron subgroup defined: {name}")
            self.neuron_subgroups[name] = idxs
            sng.name = name
            self.network.add(sng)

    def define_synapse(
        self,
        name: str,
        origin: str,
        target: str,
        synapse_eqs: str,
        on_pre: str,
        on_post: str,
        connection_condition: str = "",
        connection_probability: float = 0.0,
    ) -> None:
        synapse = None
        try:
            synapse = b2.Synapses(
                self.neuron_subgroups[origin],
                self.neuron_subgroups[target],
                synapse_eqs,
                on_pre=on_pre,
                on_post=on_post,
                name=name,
            )

        except Exception as e:
            raise RuntimeError(f"Error in defining synapse {name}: {e}") from e

        if connection_condition != "" and connection_probability > 0.0:
            try:
                synapse.connect(connection_condition, p=connection_probability)
            except Exception as e:
                raise RuntimeError(f"Error in connecting synapse {name}: {e}") from e

        self.network.add(synapse)

        self.synapses[name] = {
            "origin": origin,
            "target": target,
            "connection_condition": connection_condition,
            "connection_probability": connection_probability,
            "on_pre": on_pre,
            "on_post": on_post,
            "synapse_eqs": synapse_eqs,
        }

    def define_monitor(
        self, name: str, cls, group: b2.NeuronGroup, variable: str, record: bool = True
    ) -> None:
        monitor = None

        try:
            monitor = cls(group, variable, record=record, name=name)
        except Exception as e:
            raise RuntimeError(f"Error in defining monitor {name}: {e}") from e
        self.network.add(monitor)
        self.monitors[name] = {
            "name": name,
            "type": cls.__name__,
            "source": group,
            "record": record,
        }

    def assemble_neurongroup(self) -> None:
        eqs = b2.Equations(self.equation)

        eqs.check_identifiers()

        neurongroup = None

        try:
            neurongroup = b2.NeuronGroup(
                self.n,
                eqs,
                threshold=self.th,
                refractory=self.refractory,
                dt=self.dt,
                method=self.solver,
                namespace=self.parameters,
                name="neurongroup",
            )
        except Exception as e:
            raise RuntimeError(f"Error in assembling neuron group: {e}") from e

        self.network.add(neurongroup)

        self.neurongroup = {
            "n": self.n,
            "eqs": self.equation,
            "threshold": self.th,
            "refractory": self.refractory,
            "dt": self.dt,
            "method": self.solver,
            "namespace": self.parameters,
            "name": "neurongroup",
        }

    def save_to_yaml(self) -> None:
        data = {
            "synapses": self.synapses,
            "neurongroup": self.neurongroup,
            "neuron_subgroups": self.neuron_subgroups,
            "monitors": self.monitors,
            "path": self.path,
            "solver": self.solver,
            "dt": self.dt,
            "refractory": self.refractory,
            "threshold": self.th,
        }
        with open(Path(self.path, "cfg.yaml"), "w") as f:
            yaml.safe_dump(data, f)

    def save_to_h5(self) -> None:
        with h5py.File(Path(self.path, "data.h5"), "w") as h5file:
            for key, value in self.monitors.items():
                if h5file.get("time") is None:
                    h5file.create_dataset("time", data=self.monitors[key].t[:])
                h5file.create_dataset(key, data=value[:])

    def run_simulation(self, tfinal: float) -> None:
        self.save_to_yaml()
        self.network.run(tfinal, report="text", namespace=self.parameters)
        self.save_to_h5()
