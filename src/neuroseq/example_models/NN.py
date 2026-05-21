from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from copy import deepcopy

from neuroseq.data import Experiment

ParamValue = int | float | bool | str
FlatParams = dict[str, ParamValue]
SchemaValue = ParamValue | str
Schema = dict[str, dict[str, SchemaValue]]
ProgressCallback = Callable[[float], None]
"""Called periodically by `simulate` with the completion fraction in [0, 1]."""

class NN:
    """Standard interface for neuroseq external simulation backends.

    Backends are expected to expose editable parameter metadata, allow
    parameter updates, preserve simulation state between runs, and return
    ``Experiment`` objects.
    """

    def __init__(self, default_params_schema: Schema) -> None:
        self._default_params_schema = deepcopy(default_params_schema)
        self._params: FlatParams = self._flat_defaults()
        self._simulated_time_s = 0.0

    @property
    def simulated_time_seconds(self) -> float:
        return self._simulated_time_s

    def default_params_schema(self) -> Schema:
        return deepcopy(self._default_params_schema)

    def flat_default_params(self) -> FlatParams:
        return self._flat_defaults()

    def update_params(self, params: Mapping[str, ParamValue] | None) -> None:
        merged = self._flat_defaults()
        if params:
            merged.update(params)
        if merged != self._params:
            self._params = merged
            self.reset()

    def _flat_defaults(self) -> FlatParams:
        return {k: v["default"] for k, v in self._default_params_schema.items()}

    def reset(self) -> None:
        """Reset backend-specific simulation state."""
        raise NotImplementedError("NN.reset must be implemented by subclasses")

    def simulate(
        self,
        stims_dict: dict,
        simulation_time: float,
        params: Mapping[str, ParamValue] | None = None,
        progress: ProgressCallback | None = None,
    ) -> Experiment:
        """Run simulation and return one experiment chunk.

        ``progress``, when supplied, is invoked periodically by the
        implementation with a completion fraction in ``[0, 1]``. It is
        intentionally optional — subclasses must accept the keyword for
        forwards compatibility, but are free to ignore it when running
        without a progress consumer.
        """
        raise NotImplementedError(
            "NN.simulate must be implemented by subclasses"
        )
