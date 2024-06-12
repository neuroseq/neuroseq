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

    def __init__(self, savepath: str, loglevel: str = "INFO", seed: int = 123):
        b2.seed(seed)

        if Path(savepath).exists() is False:
            raise ValueError(f"Path {savepath} does not exist")

        self.path = savepath
        self._set_up_logger(loglevel)
        self.network = b2.Network(name="b2Network")
        self.equation = ""
        self.parameters = None
        self.neuron_subgroups = {}
        self.neurongroup = {}
        self.synapses = {}
        self.monitors = {}
        self.dt = None
        self.th = None
        self.solver = None
        self.refractory = False
        self.n = None
        self.logger.info("Model creation complete")

    def define_equation(self, line: str) -> None:
        if ":" not in line:
            message = "Equation must have a colon that separates equation and unit"
            self.logger.critical(message)
            raise ValueError(
                message
            )

        if "=" not in line:
            message = "Equation must have an equal sign that separates variable and expression"
            self.logger.critical(message)
            raise ValueError(
                message
            )

        self.equation += line + "\n"

        e = b2.Equations(self.equation)

        e.check_identifiers()
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

    def define_threshold(self, th: Union[b2.Quantity, str]) -> None:
        if not isinstance(th, b2.Quantity) and not isinstance(th, str):
            message = "threshold value must be a Quantity or a string"
            self.logger.critical(message)
            raise ValueError(message)
        self.th = th
        self.logger.info(f"Threshold defined: {th}")

    def define_solver_method(self, method: str) -> None:
        self.solver = method
        self.logger.info(f"Solver method defined: {method}")

    def define_refractory(self, value: b2.Quantity) -> None:
        if not isinstance(value, b2.Quantity):
            message = "Refractory period must be a Quantity"
            self.logger.critical(message)
            raise ValueError(message)

        self.refractory = value
        self.logger.info(f"Refractory period defined: {value}")

    def define_neuronnumber(self, n: int) -> None:
        if not isinstance(n, int):
            message = "Neuron number must be an integer"
            self.logger.critical(message)
            raise ValueError(message)
        self.logger.info(f"Neuron number defined: {n}")
        self.n = n

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

    def define_synapse(
        self,
        name: str,
        origin: str,
        target: str,
        synapse_eqs: str,
        on_pre: str = None,
        on_post: str = None,
        connection_condition: str = None,
        connection_probability: float = 0.0,
    ) -> None:
        _origin = self.network.__getitem__(
            self.neuron_subgroups[origin]["name_in_network"]
        )
        _target = self.network.__getitem__(
            self.neuron_subgroups[target]["name_in_network"]
        )

        synapse = None

        try:
            synapse = b2.Synapses(
                _origin,
                target=_target,
                model=synapse_eqs,
                on_pre=on_pre,
                on_post=on_post,
                name=name,
                namespace=self.parameters,
            )

        except Exception as e:
            message = f"Error in defining synapse {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        try:
            synapse.connect(connection_condition, p=connection_probability)
        except Exception as e:
            message = f"Error in connecting synapse {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

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

    def assemble_neurongroup(self, name: str = "neurongroup") -> None:
        ng = None
        self.logger.info(f"assembling neurongroup '{name}'")
        try:
            ng = b2.NeuronGroup(
                self.n,
                self.equation,
                threshold=self.th,
                refractory=self.refractory,
                method=self.solver,
                namespace=self.parameters,
                name=name,
            )

        except Exception as e:
            message = f"Error in assembling neuron group: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e
        
        self.network.add(ng)

        self.neurongroup = {
            "n": self.n,
            "eqs": self.equation,
            "threshold": self.th,
            "refractory": self.refractory,
            "dt": self.dt,
            "method": self.solver,
            "namespace": self.parameters,
            "name": name,
        }

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
