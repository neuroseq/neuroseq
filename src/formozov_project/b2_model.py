from logging import Logger
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import brian2 as b2
import h5py
import yaml


class Model:
    def __init__(self, savepath: str, loglevel: str = "INFO"):
        self.equation: str = """ """
        self.parameters: Dict[str, object] = {}
        self.neurongroup: b2.NeuronGroup = None
        self.neuron_subgroups: b2.NeuronGroup = {}
        self.synapses: b2.Synapses = None
        self.monitors: Dict[str, object] = {}
        self.logger = Logger("Model")
        self.logger.setLevel(loglevel)
        self.path = savepath

    def define_equation(self, line: str) -> None:
        if ":" not in line:
            raise ValueError(
                "Equation must have a colon that separates equation and unit"
            )

        if "=" not in line:
            raise ValueError(
                "Equation must have an equal sign that separates variable and equation"
            )

        self.equation += line + "\n"

        e = b2.Equations(self.equation)

        e.check_flags()

        e.check_identifiers()

        e.check_units()

        e.check_flags()

    def define_clock(self, clock: b2.Clock) -> None:
        if self.dt is not None:
            raise ValueError("dt already defined and cannot be combine with dt")
        self.clock = clock

    def define_dt(self, dt: float) -> None:
        if self.clock is not None:
            raise ValueError("Clock already defined and cannot be combine with clock")
        self.dt = dt

    def define_parameter(self, name: str, value) -> None:
        self.parameters[name] = value

    def define_threshold(self, th: str) -> None:
        self.th = th

    def define_solver_method(self, method: str) -> None:
        self.solver = method

    def define_refractory(self, value: float, unit: b2.units.Unit) -> None:
        self.refractory = value * unit

    def define_neuronnumber(self, n: int) -> None:
        self.n = n

    def define_neuron_subgroup(self, name: str, idxs: Union[slice, int, List]) -> None:
        if self.neurongroup is None:
            raise ValueError("Neurongroup not defined yet")
        else:
            self.neuron_subgroups[name] = self.neurongroup[idxs]

    def define_synapse(
        self,
        name: str,
        origin: str,
        target: str,
        synapse_eqs: str,
        on_pre: str,
        on_post: str,
        connection_condition: str,
        connection_probability: float,
    ) -> None:
        synapse = None
        try:
            synapse = b2.Synapses(
                self.neuron_subgroups[origin],
                self.neuron_subgroups[target],
                synapse_eqs,
                on_pre=on_pre,
                on_post=on_post,
            )
        except Exception as e:
            raise RuntimeError(f"Error in defining synapse {name}: {e}") from e

        try:
            synapse.connect(connection_condition, p=connection_probability)
        except Exception as e:
            raise RuntimeError(f"Error in connecting synapse {name}: {e}") from e

        self.synapses[name] = {
            "synapse": synapse,
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

        self.monitors[name] = monitor

    def assemble_neurongroup(self) -> None:
        eqs = b2.Equation(self.equation)

        eqs.check_flags()

        eqs.check_identifiers()

        eqs.check_units()

        ng = None
        try:
            ng = b2.NeuronGroup(
                self.n,
                eqs,
                threshold=self.th,
                refractory=self.refractory,
                dt=self.dt,
                clock=self.clock,
                method=self.solver,
                namespace=self.parameters,
            )
        except Exception as e:
            raise RuntimeError(f"Error in assembling neuron group: {e}") from e

        self.neurongroup = ng

    def save_to_yaml(self) -> None:
        data = {
            "equation": self.equation,
            "parameters": self.parameters,
            "method": self.solver,
            "neuron_number": self.n,
            "threshold": self.th,
            "refractory": self.refractory,
            "neuron_subgroups": self.neuron_subgroups,
            "synapses": {},
            "monitors": {},
        }

        for key, value in self.neuron_subgroups.items():
            data["neuron_subgroups"][key] = [value.start, value.stop]

        # add monitors
        for key, monitor in self.monitors.items():
            data["monitors"][key] = {
                "name": key,
                "type": type(monitor).__name__,
                "neurongroup": monitor.source,
            }

        # add synapse connections
        for key, synapses in self.synapses.items():
            data["synapses"][key] = {
                "origin": synapses["origin"],
                "target": synapses["target"],
                "connection_condition": synapses["connection_condition"],
                "connection_probability": synapses["connection_probability"],
                "on_pre": synapses["on_pre"],
                "on_post": synapses["on_post"],
                "synapse_eqs": synapses["synapse_eqs"],
            }

        with open(Path(self.path, "cfg.yaml"), "w") as f:
            yaml.safe_dump(data, f)

    def save_to_h5(self) -> None:
        with h5py.File(Path(self.path, "data.h5"), "w") as h5file:
            for key, value in self.monitors.items():
                h5file.create_dataset(key, data=value[:])

    @classmethod
    def load_from_yaml(cls, path_to_load: str) -> None:
        with open(Path(path_to_load) / "cfg.yaml") as f:
            ymlfile = yaml.safe_load(f)

        obj = cls(path_to_load)

        obj.equation = ymlfile["equation"]
        obj.parameters = ymlfile["parameters"]
        obj.solver = ymlfile["method"]
        obj.n = ymlfile["neuron_number"]
        obj.th = ymlfile["threshold"]
        obj.refractory = ymlfile["refractory"]

        obj.assemble_neurongroup()

        for key, value in ymlfile["neuron_subgroups"].items():
            obj.neuron_subgroups[key] = obj.neurongroup[value[0] : value[1]]

        for key, value in ymlfile["synapses"].items():
            obj.define_synapse(
                key,
                value["origin"],
                value["target"],
                value["synapse_eqs"],
                value["on_pre"],
                value["on_post"],
                value["connection_condition"],
                value["connection_probability"],
            )

        for key, value in ymlfile["monitors"].items():
            obj.define_monitor(
                key,
                value["type"],
                obj.neurongroup,
                value["name"],
            )

        return obj

    def run_simulation(self, tfinal: float) -> None:
        self.save_to_yaml()
        b2.run(tfinal, report="text")
        self.save_to_h5()
