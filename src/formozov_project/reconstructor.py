from typing import Any

import brian2 as b2
import yaml

from formozov_project.b2_model import B2Model


class Reconstructor:
    def __init__(self, path: str) -> None:
        with open(path) as file:
            self.loaded_data = yaml.load(file, Loader=yaml.UnsafeLoader)

    def reconstruct_neurongroups(self, data_of_group: dict[str, Any]) -> None:
        pass

    def reconstruct_synapses(self, data_of_group: dict[str, Any]) -> None:
        pass

    def reconstruct_monitors(self, data_of_group: dict[str, Any]) -> None:
        pass

    def reconstruct(self, model: B2Model) -> b2.Network:
        net = b2.Network()

        # reconstruct neurongroups
        for n in self.loaded_data["objects"]["neurongroups"]:
            self.reconstruct_neurongroups(self.loaded_data[n])

        # reconstruct synapses
        for s in self.loaded_data["objects"]["synapses"]:
            self.reconstruct_synapses(self.loaded_data[s])

        # reconstruct monitors
        for m in self.loaded_data["objects"]["monitors"]:
            self.reconstruct_monitors(self.loaded_data[m])

        # reconstruct parameters, timestep, seed.
        model.parameters = self.loaded_data["parameters"]

        return net
