from __future__ import annotations

import json
import logging
import math
import pickle
import zipfile
from collections.abc import Callable
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import NoReturn
from typing import TypeVar

import brian2 as b2
import brian2.numpy_ as np
import h5py

from neuroseq import utils
from neuroseq.b2json import b2json_encoder_factory
from neuroseq.experiment import SpikeData
from neuroseq.experiment import SynapseData
from neuroseq.experiment import TraceData
from neuroseq.reconstructor import Reconstructor
from neuroseq.utils import append_data
from neuroseq.utils import construct_synapse_matrix
from neuroseq.utils import save_store_name

DataType = TypeVar("DataType", TraceData, SpikeData, SynapseData)


class B2Model:
    """
    Wrapper class that bundles brian2 capabilities for modelling neuronal networks. This creates a dict-based intermediate represenation
    of the model. All function attributes that start with `define_` act on this intermediate representation and add elements to it,
    all function attributes that are called `delete_` delete elements from it.
    """

    def _trigger_rebuild(func: Callable) -> Callable:  # type: ignore
        def do_with_rebuild(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.must_be_rebuilt = True

        return do_with_rebuild

    def _handle_exception(self, e: Exception, message: str) -> NoReturn:
        self.logger.critical(message)
        self.logger.exception(e)
        raise RuntimeError(message) from e

    def _add_equation_to_group(self, group: dict, eqs: str, key: str = "model") -> None:
        r"""
        _add_equation_to_group Add an equation to a neurongroup or synapse dictionary that holds arguments to construct the respective objects.

        Parameters
        ----------
        group : dict
            Argument dictionary for constructing a brian2 group object - neurongroup or synapse.
        eqs : str
            An equation of the form v = f(y,x): unit, possibly containing multiple lines for multiple variables.

        Raises
        ------
        ValueError
            When there is no ': unit' block in the line given
        ValueError
            When there is no '=' block in the line given
        """
        eqs = eqs.strip("\n")

        if eqs == "":
            return

        if ":" not in eqs and "=" not in eqs:
            message = "Equation must have a colon that separates equation and unit or an equal sign"
            self.logger.critical(message)
            raise ValueError(message)

        group[key] = eqs

        self.logger.debug(f"full equation: {repr(eqs)}")

    def _add_group_modifier(
        self,
        name: str,
        group_class: str,
        with_rebuilt: Callable,
        element_adder: Callable = lambda *_, **__: None,
    ) -> None:
        """
        add a modifier instance responsible for modifying dictionary elements
        that are themselves dictionaries

        Parameters
        ----------
        name : str
            key of the held dict
        group_class : str
            name of the element the held dict belongs to "neurongroups", "monitors", "synapses", "initials" or "parameters".
        with_rebuilt : Callable
            function to determine whether the system has to be rebuilt or not when chaning something
        element_adder : Callable, optional
            Function to call to add an element to he held dict, by default lambda*_
        """
        modifier = self.Modifier(
            name,
            getattr(self, group_class),
            self.functions[group_class],
            with_rebuilt=with_rebuilt,
            element_adder=element_adder,
        )

        self.functions[group_class][name] = {
            "rename": modifier.rename,
            "delete": modifier.delete,
            "add": modifier.add,
        }

    @_trigger_rebuild
    def _assemble_neuronsubgroup(
        self,
        name: str,
    ) -> None:
        """
        _assemble_neuronsubgroup Define a subgroup of a neurongroup as a separate element that can be referred to by name.
        Overrides any existing group named 'name' in the network.
        Parameters
        ----------
        name : str
            Name that the subgroup should be referred to.
        """
        sng = None

        if isinstance(self.neurongroups[name]["idxs"], tuple | list):
            if len(self.neurongroups[name]["idxs"]) > 2:
                raise ValueError("Subgroups have to be contiguous")
            start = self.neurongroups[name]["idxs"][0]
            stop = self.neurongroups[name]["idxs"][1]
        elif isinstance(self.neurongroups[name]["idxs"], int):
            start = self.neurongroups[name]["idxs"]
            stop = self.neurongroups[name]["idxs"] + 1

        try:
            sng = b2.Subgroup(
                self.network[self.neurongroups[name]["target"]],
                start,
                stop,
                name=name,
            )
        except Exception as e:
            self._handle_exception(
                e, f"Error when trying to create a neuronsubgroup {name}: {e}"
            )

        if any(obj.name == name for obj in self.network.objects):
            try:
                self.network.remove(sng)
            except Exception as e:
                self._handle_exception(
                    e, f"Error when removing existing subgroup {name}: {e}"
                )

        try:
            self.network.add(sng)
        except Exception as e:
            self._handle_exception(e, f"Error when adding subgroup {name}: {e}")
        self.logger.info(f"Neuron subgroup defined: {name}")

    @_trigger_rebuild
    def _assemble_synapse(
        self,
        name: str,
    ) -> None:
        """
        _assemble_synapse Assemble a synapse called `name` from its set of predefined values.

        You need to define the respective variables first via the `define_synapse_*` functions.
        Overrides any existing group named 'name' in the network.
        Parameters
        ----------
        name : str, optional
            How the Synapse should be named in the network

        Raises
        ------
        RuntimeError
            When an exception is raised by the underlying brian2 function that builds the synapse
        RuntimeError
            When an exception is raised by the underlying brian2 function that creates synapse connections.
        """
        args = self.synapses[name]
        origin_id = self.neurongroups[args["origin"]].get("name", args["origin"])
        target_id = self.neurongroups[args["target"]].get("name", args["target"])

        _origin = self.network.__getitem__(origin_id)
        _target = self.network.__getitem__(target_id)

        synapse = None
        try:
            synapse = b2.Synapses(
                _origin,
                target=_target,
                model=args.get("model", None),
                on_pre=args.get("on_pre", None),
                on_post=args.get("on_post", None),
                delay=args.get("delay", None),
                clock=b2.Clock(
                    dt=args.get("dt", b2.defaultclock.dt), name=f"clock_synapse_{name}"
                ),
                name=name,
                namespace=self.parameters,
            )

        except Exception as e:
            self._handle_exception(e, f"Error in defining synapse {name}: {e}")

        try:
            synapse.connect(
                args.get("connection_condition", None), p=args.get("p", 1.0)
            )
        except Exception as e:
            self._handle_exception(e, f"Error in connecting synapse {name}: {e}")

        if any(obj.name == name for obj in self.network.objects):
            try:
                self.network.remove(self.network[name])
            except Exception as e:
                self._handle_exception(
                    e,
                    f"Error when removing existing synapse {name} from the network: {e}",
                )

        try:
            self.network.add(synapse)
        except Exception as e:
            self._handle_exception(e, f"Error when adding {name} to the network: {e}")
        self.logger.info(f"Synapse defined: {name}")

    @_trigger_rebuild
    def _assemble_monitor(self, name: str) -> None:
        """
        _assemble_monitor Assemble a new monitor from predefined elements. When a monitor exists that
        in the network that has the same name, it is removed and replaced by the newly assembled one
        Overrides any existing group named 'name' in the network.
        Parameters
        ----------
        name : str
            Name of the monitor as referred to in the governing network.

        Raises
        ------
        RuntimeError
            When something goes wrong while removing a monitor of the same name that is to be replaced with a new definition
        RuntimeError
            When something goes wrong while building the monitor
        """
        monitor = None
        self.logger.info(f"adding monitor '{name}'")

        try:
            monitortype = getattr(b2, self.monitors[name]["type"])

            if monitortype is b2.StateMonitor:
                # set to source dt when no timestep is given --> max resolution. Also, .dt gives a VariableView, which needs to be converted to a value again
                clock = b2.Clock(
                    dt=self.monitors[name].get(
                        "dt", self.network[self.monitors[name]["source"]].clock.dt
                    ),
                    name=f"clock_monitor_{name}",
                )
                monitor = monitortype(
                    self.network[self.monitors[name]["source"]],
                    self.monitors[name].get("variable", None),
                    record=True,
                    name=name,
                    clock=clock,
                )
            else:
                monitor = monitortype(
                    self.network[self.monitors[name]["source"]],
                    self.monitors[name].get("variable", None),
                    record=True,
                    name=name,
                )
        except Exception as e:
            self._handle_exception(e, f"Error in defining monitor {name}: {e}")

        if any(obj.name == name for obj in self.network.objects):
            try:
                self.network.remove(self.network[name])
            except Exception as e:
                self._handle_exception(
                    e, f"Error while removing monitor {name} from network: {e}"
                )

        try:
            self.network.add(monitor)
        except Exception as e:
            self._handle_exception(e, f"Error in adding monitor {name} to network: {e}")

    @_trigger_rebuild
    def _assemble_neurongroup(
        self,
        name: str = "neurongroup",
    ) -> None:
        """
        _assemble_neurongroup Assemble a neurongroup called `name` from its set of predefined values.
        When a neurongroup of the same name already exists in the network, it will be removed and replaced with the newly assembled one.
        You need to define the respective variables first via the `define_*` functions.
        Overrides any existing group named 'name' in the network.
        Parameters
        ----------
        name : str, optional
            How the NeuronGroup should be named in the network, by default "neurongroup"

        Raises
        ------
        RuntimeError
            When building the neurongroup was not successful, e.g., because of missing parameters.
        """

        ng = None
        self.logger.info(f"assembling neurongroup '{name}'")
        try:
            ng = b2.NeuronGroup(
                self.neurongroups[name].get("neuronnumber", None),
                self.neurongroups[name].get("model", None),
                threshold=self.neurongroups[name].get("th", None),
                refractory=self.neurongroups[name].get("refractory", False),
                method=self.neurongroups[name].get("method", None),
                namespace=self.parameters,
                reset=self.neurongroups[name].get("reset", None),
                clock=b2.Clock(
                    dt=self.neurongroups[name].get("dt", b2.defaultclock.dt),
                    name=f"clock_neurongroup_{name}",
                ),
                name=name,
            )
        except Exception as e:
            self._handle_exception(e, f"Error in assembling neuron group {name}: {e}")

        if any(obj.name == name for obj in self.network.objects):
            try:
                self.network.remove(self.network[name])
            except Exception as e:
                self._handle_exception(
                    e, f"Error when removing neuron group {name} from network: {e}"
                )

        try:
            self.network.add(ng)
        except Exception as e:
            self._handle_exception(
                e, f"Error when adding new neuron group {name} to network: {e}"
            )

    @_trigger_rebuild
    def _assemble_initial_conditions(self, name: str) -> None:
        """
        Set all the initial conditions that have been defined via 'define_init'.
        Overrides any existing initial conditions named 'name' in the network.
        Parameters
        ----------
        name : str
            name of the element of the network for which to define an initial condition.

        Raises
        ------
        ValueError
            When something goes wrong during assignment of the initial condition to the variable given
        """
        for key in self.initials[name]:
            try:
                setattr(self.network[name], key, self.initials[name][key])

            except Exception as e:
                self.logger.exception(e)
                raise ValueError(
                    f"Error when initializing {key} of element {name} as {self.initials[name][key]}: {e}"
                ) from e

    @_trigger_rebuild
    def _delete_dict_element(self, name: str, key: Any) -> None:
        del getattr(self, name)[key]
        del self.functions[name][key]

    # stateful class for modifying, renaming and deleting dictionary elements in arbitrary order
    class Modifier:
        "Stateful functor for renaming elements in a dict successively while keeping track of the old name"

        def __init__(
            self,
            current_name: str,
            pdict: dict[Any, Any],
            fdict: dict[Any, Any],
            with_rebuilt: Callable,
            element_adder: Callable = lambda *args: None,
        ):
            """
            Create a new modifier instance
            Parameters
            ----------
            current_name : str
                name of the element this modifier is bound to in its parent object
            pdict : dict
                element this modifier is bound to
            fdict : dict
                function dictionary the `pdict` argument is bound to
            with_rebuilt : Callable
                function to call upon modification of `pdict`
            element_adder: Callable
                function taking arbitrary args and kwargs to create a new element if this modifier is bound to a dictionary object
            """
            self.current_name = current_name
            self.pdict = pdict
            self.fdict = fdict
            self.with_rebuilt = with_rebuilt
            self.element_adder = element_adder
            self.logger = logging.getLogger(__name__)

        def rename(self, new_name: str) -> None:
            """
            Rename the element the caller is bound to to 'new_name' from the currently held 'current_name'

            Parameters
            ----------
            new_name : str
                New name of the element

            """
            self.logger.info(f"Renaming {self.current_name} to {new_name}")

            used_old_name = self.current_name
            self.current_name = new_name
            old_value = self.pdict.pop(used_old_name)
            old_functions = self.fdict.pop(used_old_name)
            self.pdict[new_name] = old_value
            self.fdict[new_name] = old_functions

            self.with_rebuilt()

        def modify(self, value: str) -> None:
            """
            Modify the element the caller is bound to.

            Parameters
            ----------
            value : str
                value to set the element to
            """
            self.pdict[self.current_name] = value
            self.logger.info(f"Modified {self.current_name} to {value}")
            self.with_rebuilt()

        def delete(self) -> None:
            """
            Delete the element behind the currently held name
            """
            del self.pdict[self.current_name]
            del self.fdict[self.current_name]
            self.with_rebuilt()
            self.logger.info(f"Deleted {self.current_name}")

        def add(self, *args, **kwargs) -> None:
            """
            add a new element if the held pdict is a nested dict
            """
            self.element_adder(self.current_name, *args, **kwargs)
            self.with_rebuilt()
            self.logger.info(
                f"Added element to {self.current_name} built with {args}, {kwargs}"
            )

    def __init__(
        self,
        seed: int = 123,
        name: str = "b2Network",
    ):
        """
        __init__ Create a new `B2Model` instance.

        Parameters
        ----------
        seed : int, optional
            Seed for the brian2 rng system. Set this to produce a reproducible system, by default 123
        """
        b2.seed(seed)

        self.seed: int = seed
        self.network = b2.Network(name=name)
        self.parameters: dict[str, Any] = {}
        self.neurongroups: dict[str, dict[str, Any]] = {}
        self.synapses: dict[str, dict[str, Any]] = {}
        self.monitors: dict[str, dict[str, Any]] = {}
        self.initials: dict[str, dict[str, Any]] = {}
        self.experiments: dict[str, list[TraceData | SpikeData | SynapseData]] = {}

        self.max_ram_synapse_plot_matrix = 256
        self.trace_name_to_monitor: dict[str, b2.StateMonitor] = {}
        self.spike_name_to_monitor: dict[str, b2.SpikeMonitor] = {}
        self.synapse_name_to_monitor: dict[str, b2.StateMonitor] = {}

        self.functions: dict[str, Any] = {
            "neurongroups": {},
            "synapses": {},
            "initials": {},
            "parameters": {},
            "monitors": {},
        }
        self.checkpoints: list[str] = []
        self.must_be_rebuilt = False
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model creation complete")

    @property
    def name(self) -> str:
        return self.network.name

    def set_rebuilt(self, value: bool) -> None:
        self.must_be_rebuilt = value or self.must_be_rebuilt

    @_trigger_rebuild
    def define_init(self, name: str, varname: str, value: Any) -> None:
        """
        Define the initial condition of the variable 'varname' in the network element called 'name' and set it to 'value

        Parameters
        ----------
        name : str
            Name of the network element to set the initial condition for
        varname : str
            variable in the network element to set
        value : Any
            value to set the variable to
        """
        self.logger.info(f"defining initial condition for {name}, {varname}, {value}")

        if name not in self.initials:
            self.initials[name] = {}

            self._add_group_modifier(
                name,
                "initials",
                lambda: self.set_rebuilt(True),
                lambda n: self.define_init(n, "variable", "value"),
            )

        element_modifier = self.Modifier(
            varname,
            self.initials[name],
            self.functions["initials"][name],
            lambda: self.set_rebuilt(True),
        )
        self.functions["initials"][name][varname] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        utils.update_nested_dict(
            self.initials[name],
            {
                varname: value,
            },
        )

    @_trigger_rebuild
    def define_default_init(self) -> None:
        self.define_init("group", "value", None)

    def define_b2seed(self, seed: int) -> None:
        """
        Set the brian2 rng seed to 'seed'. This also regenerates the internal buffer of random numbers when using codegeneration with cython.

        Parameters
        ----------
        seed : int
            The new seed to set
        """
        b2.seed(seed)

    def define_name(self, name: str) -> None:
        """
        define_name Define the name of the model

        Parameters
        ----------
        name : str
            Name of the model
        """
        self.network.name = self.name

    def define_parameter(
        self, name: str, value: str | b2.Quantity | b2.TimedArray
    ) -> None:
        """
        define_parameter Define a parameter for *any* model equation in the network. All elements will get the same set of parameters,
        so make sure to not include doublings. This is *NOT* verified.

        Parameters
        ----------
        name : str
            parameter name

        value : str | b2.Quantity | b2.TimedArray
            value to be used. Can be an expression in the form of a string or a quantity with units or a TimedArray.
        """

        element_modifier = self.Modifier(
            name,
            self.parameters,
            self.functions["parameters"],
            lambda _model=self: _model.set_rebuilt(False),
        )

        self.functions["parameters"][name] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        self.parameters[name] = value

        self.logger.info(f"Parameter defined: {name}")
        self.logger.debug(f"Parameter value: {value}")

    def define_default_parameter(self) -> None:
        self.define_parameter("new_parameter", None)

    # neurongroups
    @_trigger_rebuild
    def define_default_neurongroup(self) -> None:
        name = "new_neurongroup"
        if name not in self.neurongroups:
            self.neurongroups[name] = {}

        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        self.define_neuronnumber(name, 0)
        self.define_equation(name, "dx/dt = y : 1")
        self.define_solver_method(name, "-")
        self.define_dt(name, 10 * b2.ms)
        self.define_reset(name, "0")
        self.define_threshold(name, "0")
        self.define_refractory(name, "0")

    @_trigger_rebuild
    def define_default_neuronsubgroup(self) -> None:
        name = "new_neuronsubgroup"
        if name not in self.neurongroups:
            self.neurongroups[name] = {}

        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        self.define_neuronsubgroup_target(name, "-")
        self.define_neuronsubgroup_idxs(name, [0, 0])

    @_trigger_rebuild
    def define_dt(self, name: str, dt: b2.Quantity) -> None:
        """
        define_dt Define the timeste for the neurongroup called 'name'.

        Parameters
        ----------
        dt : b2.Quantity
            timestep to be used. Must be given as value * unit.
        """
        if name not in self.neurongroups:
            self.neurongroups[name] = {}

        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "dt",
            self.neurongroups[name],
            self.functions["neurongroups"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["neurongroups"][name]["dt"] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        self.neurongroups[name]["dt"] = dt

        self.logger.info(f"dt defined: {dt} for neurongroup {name}")

    @_trigger_rebuild
    def define_equation(self, name: str, line: str) -> None:
        r"""
        define_equation Define a model equation for neurongroups.

        An equation always must be of the form dy \ dx = f(y,x): unit.

        Parameters
        ----------
        name : str
            Name of the neurongroup this model equation should be added to.
        line : str
            String representation of the equation to be used.

        Raises
        ------
        ValueError
            When the supplied equation string does not contain an equality sign.
        ValueError
            When the supplied equation string does not contain a unit element.
        """

        if name not in self.neurongroups:
            self.neurongroups[name] = {}

        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "model",
            self.neurongroups[name],
            self.functions["neurongroups"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["neurongroups"][name]["model"] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        self._add_equation_to_group(self.neurongroups[name], line)

    @_trigger_rebuild
    def define_reset(self, name: str, reset: str) -> None:
        """
        define_reset Define the `reset` parameter for a neurongroup indicated by `name`.

        Parameters
        ----------
        name : str
            Name of the neurongroup to define the reset parameter for.
        reset : str
            The reset parameter for the neurongroup. Have a look at the brian2 documentation for
            the `NeuronGroup` class for more details.
        """
        if name not in self.neurongroups:
            self.neurongroups[name] = {}
        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "reset",
            self.neurongroups[name],
            self.functions["neurongroups"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["neurongroups"][name]["reset"] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        utils.update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "reset": reset,
                },
            },
        )

    @_trigger_rebuild
    def define_threshold(self, name: str, th: str) -> None:
        """
        define_threshold Define the threshold for the neurongroup given by `name`.
                        See the brian2 documentation for `NeuronGroup` for more details.
        Parameters
        ----------
        name : str
            name of the NeuronGroup to add the threshold for.
        th : str
            Threshold expression as a string.
        """
        if name not in self.neurongroups:
            self.neurongroups[name] = {}
        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "th",
            self.neurongroups[name],
            self.functions["neurongroups"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["neurongroups"][name]["th"] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        utils.update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "th": th,
                },
            },
        )
        self.logger.info(f"Threshold defined: {th}")

    @_trigger_rebuild
    def define_solver_method(self, name: str, method: str) -> None:
        """
        define_solver_method Define the solver method for the neurongroup indicated by `name`.

        Parameters
        ----------
        name : str
            name of the neurongroup to define the solver method for
        method : str
            name of the method to use. Refer to the brian2 documentation for the currently available set of solvers.
        """

        if name not in self.neurongroups:
            self.neurongroups[name] = {}
        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "method",
            self.neurongroups[name],
            self.functions["neurongroups"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["neurongroups"][name]["method"] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        utils.update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "method": method,
                },
            },
        )

        self.logger.info(f"Solver method defined: {method} for {name}")

    @_trigger_rebuild
    def define_refractory(self, name: str, value: str | b2.Quantity) -> None:
        """
        define_refractory Define the refractoriness for a neurongroup indicated by name.

        Parameters
        ----------
        name : str
            Name of the group to add the refractoriness for.
        value : str | b2.Quantity
            Either a string expression evalutating to the length of the refractory period or the length of the period itself.
        """
        if name not in self.neurongroups:
            self.neurongroups[name] = {}

        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )
        element_modifier = self.Modifier(
            "refractory",
            self.neurongroups[name],
            self.functions["neurongroups"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["neurongroups"][name]["refractory"] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        utils.update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "refractory": value,
                },
            },
        )
        self.logger.info(f"Refractory period defined: {value}")

    @_trigger_rebuild
    def define_neuronnumber(self, name: str, n: int) -> None:
        """
        define_neuronnumber Define the number of neurons in a group indicated by name.

        Parameters
        ----------
        name : str
            Name of the neurongroup to define the neuronumber for.
        n : int
            Number of neurons the neurongroup should have.
        """
        if name not in self.neurongroups:
            self.neurongroups[name] = {}

        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )
        element_modifier = self.Modifier(
            "neuronnumber",
            self.neurongroups[name],
            self.functions["neurongroups"][name],
            lambda: self.set_rebuilt(True),
        )
        self.functions["neurongroups"][name]["neuronnumber"] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        utils.update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "neuronnumber": n,
                },
            },
        )
        self.logger.info(f"Neuron number defined: {n}")

    @_trigger_rebuild
    def define_neuronsubgroup_target(self, name: str, target_name: str) -> None:
        """
        Define the target neurongroup for a neuron subgroup called name

        Parameters
        ----------
        name : str
            name of the subgroup for which to define the target
        target_name : str
            name of the target neurongroup to make a subgroup from

        Raises
        ------
        ValueError
            When the target_name argument does not exist in the neurongroups that are known to the model
        """
        self.logger.info(f"defining target {target_name} for neuron subgroup {name}")

        if name not in self.neurongroups:
            self.neurongroups[name] = {}

        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "target",
            self.neurongroups[name],
            self.functions["neurongroups"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["neurongroups"][name]["target"] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        utils.update_nested_dict(
            self.neurongroups[name],
            {
                "target": target_name,
            },
        )

    @_trigger_rebuild
    def define_neuronsubgroup_idxs(self, name: str, idxs: int | list | tuple) -> None:
        """
        Define the indices in a neurongroup for a neuron subgroup called 'name'.

        Parameters
        ----------
        name : str
            name of subgroup to define the subgroup indices in the target neurongroup for
        idxs : int | list | tuple
            Indices of subgroup. if 'int' the index of a neuron, if 'list' a list of indices of neurons,
            if tuple, a (start, stop) range equivalent to range(start, stop).
        """
        self.logger.info(f"defining indices {idxs} for neuron subgroup {name}")

        if name not in self.neurongroups:
            self.neurongroups[name] = {}

        if name not in self.functions["neurongroups"]:
            self._add_group_modifier(
                name,
                "neurongroups",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "idxs",
            self.neurongroups[name],
            self.functions["neurongroups"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["neurongroups"][name]["idxs"] = {
            "delete": element_modifier.delete,
            "modify": element_modifier.modify,
            "rename": element_modifier.rename,
        }

        # only contiguous arrays allowed for definition of neurongroup
        utils.update_nested_dict(self.neurongroups[name], {"idxs": idxs})

    # synapses
    @_trigger_rebuild
    def define_default_synapse(self) -> None:
        name = "new_synapse"
        if name not in self.synapses:
            self.synapses[name] = {}

        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        self.define_synapse_origin(
            name,
            "-",
        )
        self.define_synapse_target(
            name,
            "-",
        )
        self.define_synapse_equation(
            name,
            "dx/dt = y : 1",
        )
        self.define_synapse_solver_method(
            name,
            "-",
        )
        self.define_synapse_dt(
            name,
            10 * b2.ms,
        )
        self.define_synapse_delay(
            name,
            0 * b2.ms,
        )
        self.define_synapse_connection(
            name,
            "x!=y",
        )
        self.define_synapse_connection_probability(
            name,
            0.0,
        )
        self.define_synapse_pre(
            name,
            "x=y",
        )
        self.define_synapse_post(
            name,
            "x=y",
        )

    @_trigger_rebuild
    def define_synapse_dt(self, name: str, dt: b2.Quantity) -> None:
        """
        define_synapse_dt Define the timestep for the synapse 'name'.

        Parameters
        ----------
        dt : b2.Quantity
            timestep to be used. Must be given as value * unit.
        """
        if name not in self.synapses:
            self.synapses[name] = {}

        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "dt",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name]["dt"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        self.synapses[name]["dt"] = dt

        self.logger.info(f"dt defined: {dt} for synapse {name}")

    @_trigger_rebuild
    def define_synapse_solver_method(self, name: str, method: str) -> None:
        """
        define_solver_method Define the solver method for the synapse indicated by `name`.

        Parameters
        ----------
        name : str
            name of the synapse to define the solver method for
        method : str
            name of the method to use. Refer to the brian2 documentation for the currently available set of solvers.
        """

        if name not in self.synapses:
            self.synapses[name] = {}

        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "method",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )
        self.functions["synapses"][name]["method"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }
        utils.update_nested_dict(
            self.synapses,
            {
                name: {
                    "method": method,
                },
            },
        )
        self.logger.info(f"Solver method defined: {method} for {name}")

    @_trigger_rebuild
    def define_synapse_target(self, name: str, target: str) -> None:
        """
        define_synapse_target Define the target neurongroup for a synapse called 'name'.

        Parameters
        ----------
        name : str
            Name of the synapse to add the target neurongroup for.
        target : str
            Name of the target neurongroup to use with the synapse.
        """
        if name not in self.synapses:
            self.synapses[name] = {}
        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "target",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name]["target"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(
            self.synapses,
            {
                name: {
                    "target": target,
                }
            },
        )

    @_trigger_rebuild
    def define_synapse_origin(self, name: str, origin: str) -> None:
        """
        define_synapse_origin Define the origin neurongroup for a synapse called 'name'.

        Parameters
        ----------
        name : str
            Name of the synapse to add the origin neurongroup for.
        target : str
            Name of the origin neurongroup to use with the synapse.
        """
        if name not in self.synapses:
            self.synapses[name] = {}
        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "origin",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name]["origin"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(
            self.synapses,
            {
                name: {
                    "origin": origin,
                }
            },
        )

    @_trigger_rebuild
    def define_synapse_equation(self, name: str, line: str) -> None:
        """
        define_synapse_equation Define a line in the model equation for a synapse called `synapse`, or the full equation all at once.
        Have a look at the brian2 documentation for syanpses for more details.

        Parameters
        ----------
        name : str
            name of the synapse to add the equation to.
        line : str
            string expression for the model equation to be added. Must be of the form  v = f(x,y) : unit

        Raises
        ------
        ValueError
            When the `: unit` element is missing in the given expression.
        ValueError
            When the `=` is missing in the given expression.
        """
        if name not in self.synapses:
            self.synapses[name] = {}
        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )
        element_modifier = self.Modifier(
            "model",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name]["model"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        self._add_equation_to_group(self.synapses[name], line)

    @_trigger_rebuild
    def define_synapse_pre(self, name: str, line: str) -> None:
        """
        define_synapse_pre Define the code that is executed on each synapse receiving a presynaptic spike as a string expression.

        Parameters
        ----------
        name : str
            Name of the synapse to add presynaptic code for.
        line : str
            String representation of the code to use.
        """
        if name not in self.synapses:
            self.synapses[name] = {}

        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "on_pre",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name]["on_pre"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        self._add_equation_to_group(self.synapses[name], line, key="on_pre")

    @_trigger_rebuild
    def define_synapse_post(self, name: str, line: str) -> None:
        """
        define_synapse_post Add code to a synapse called `name` that shall run
        when a postsynaptic neuron fired a spike.

        Parameters
        ----------
        name : str
            name of the synapse to add post-synaptic code to.
        line : str
            String representation of the code to add.
        """
        if name not in self.synapses:
            self.synapses[name] = {}
        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "on_post",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name]["on_post"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        self._add_equation_to_group(self.synapses[name], line, key="on_post")

    @_trigger_rebuild
    def define_synapse_connection(self, name: str, connection_condition: str) -> None:
        """
        define_synapse_connection Define the connections of a Synapse called `name` as a condition represented as a string.
        Please refer to the brian2 documentation for more details on how synapses are defined.

        Parameters
        ----------
        name : str
            Name of the synapse to add the connection condition to.
        connection_condition : str
            String representaion of the connection condition.
        """
        if name not in self.synapses:
            self.synapses[name] = {}
        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )
        element_modifier = self.Modifier(
            "connection_condition",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name]["connection_condition"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(
            self.synapses,
            {
                name: {
                    "connection_condition": connection_condition,
                },
            },
        )

    @_trigger_rebuild
    def define_synapse_delay(self, name: str, delay: Any) -> None:
        """
        Add delay to synapse

        Parameters
        ----------
        name : str
            name of synapse to add delay for
        delay : Any
            delay expression
        """
        if name not in self.synapses:
            self.synapses[name] = {}
        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )
        element_modifier = self.Modifier(
            "delay",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name]["delay"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(
            self.synapses,
            {
                name: {
                    "delay": delay,
                },
            },
        )

    @_trigger_rebuild
    def define_synapse_connection_probability(self, name: str, p: float) -> None:
        """
        define_synapse_connection_probability Define the connection probability for a given synapse called `name`.

        Parameters
        ----------
        name : str Name of the synapse to define the connection probability for.

        p : float
            Probability value. Must be in [0, 1]
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Error, supplied value {p} is not a probability.")

        if name not in self.synapses:
            self.synapses[name] = {}
        if name not in self.functions["synapses"]:
            self._add_group_modifier(
                name,
                "synapses",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "p",
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name]["p"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(
            self.synapses,
            {
                name: {
                    "p": p,
                },
            },
        )

    # monitor
    @_trigger_rebuild
    def define_default_monitor(self) -> None:
        name = "new_monitor"
        if name not in self.monitors:
            self.monitors[name] = {}

        if name not in self.functions["monitors"]:
            self._add_group_modifier(
                name,
                "monitors",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        self.define_monitor_type(
            name,
            "-",
        )
        self.define_monitor_source(name, "-")
        self.define_monitor_variable(name, "-")
        self.define_monitor_dt(name, None)

    @_trigger_rebuild
    def define_monitor_type(self, name: str, classname: str) -> None:
        """
        Define the type of a monitor. Can be 'StateMonitor' or 'SpikeMonitor' at the moment.

        Parameters
        ----------
        name : str
            name of the monitor to define the type for
        classname : str
            name of the type to use in brian2
        """
        if name not in self.monitors:
            self.monitors[name] = {}
        if name not in self.functions["monitors"]:
            self._add_group_modifier(
                name,
                "monitors",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "type",
            self.monitors[name],
            self.functions["monitors"][name],
            lambda _model=self: _model.set_rebuilt(True),
        )

        self.functions["monitors"][name]["type"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(
            self.monitors,
            {
                name: {
                    "type": classname,
                }
            },
        )

    @_trigger_rebuild
    def define_monitor_source(self, name: str, source: str) -> None:
        """
        Define the source member of the neuron network to monitor variables for

        Parameters
        ----------
        name : str
            name of the monitor
        source : str
            Member of the network to monitor variables for
        """
        if name not in self.monitors:
            self.monitors[name] = {}
        if name not in self.functions["monitors"]:
            self._add_group_modifier(
                name,
                "monitors",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "source",
            self.monitors[name],
            self.functions["monitors"][name],
            lambda _model=self: _model.set_rebuilt(True),
        )
        self.functions["monitors"][name]["source"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(
            self.monitors,
            {
                name: {
                    "source": source,
                }
            },
        )

    @_trigger_rebuild
    def define_monitor_variable(self, name: str, variable: str | None) -> None:
        """
        define the variable a monitor can be attached to. Can be a string , None (for spikemonitors) or True (for a StateMonitor).
        When True, all variables are recorded. This may take up a lot of memory

        Parameters
        ----------
        name : str
            name of the monitor to define the variable for
        variable : str | None
            Variable name
        """
        if name not in self.monitors:
            self.monitors[name] = {}
        if name not in self.functions["monitors"]:
            self._add_group_modifier(
                name,
                "monitors",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "variable",
            self.monitors[name],
            self.functions["monitors"][name],
            lambda _model=self: _model.set_rebuilt(True),
        )

        self.functions["monitors"][name]["variable"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(
            self.monitors,
            {
                name: {
                    "variable": variable,
                }
            },
        )

    @_trigger_rebuild
    def define_monitor_dt(self, name: str, dt: b2.Quantity) -> None:
        """
        define_monitor_dt Define the timestep for the monitor 'name'.

        Parameters
        ----------
        dt : b2.Quantity
            timestep to be used. Must be given as value * unit.
        """
        if name not in self.monitors:
            self.monitors[name] = {}

        if name not in self.functions["monitors"]:
            self._add_group_modifier(
                name,
                "monitors",
                lambda: self.set_rebuilt(True),
                lambda: self.set_rebuilt(True),
            )

        element_modifier = self.Modifier(
            "dt",
            self.monitors[name],
            self.functions["monitors"][name],
            lambda _model=self: _model.set_rebuilt(True),
        )

        self.functions["monitors"][name]["dt"] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(
            self.monitors,
            {
                name: {
                    "dt": dt,
                }
            },
        )

        self.logger.info(f"dt defined: {dt} for monitor {name}")

    def assemble_network(self) -> None:
        """
        Assemble the network from its predefined components.
        """
        self.network = b2.Network(name=self.name)
        self.checkpoints.clear()

        self._make_timed_arrays_valid()

        for n in self.neurongroups:
            if "idxs" not in self.neurongroups[n]:
                self._assemble_neurongroup(n)

        for n in self.neurongroups:
            if "idxs" in self.neurongroups[n]:
                self._assemble_neuronsubgroup(n)

        for s in self.synapses:
            self._assemble_synapse(s)

        for m in self.monitors:
            self._assemble_monitor(m)

        for n in self.initials:
            self._assemble_initial_conditions(n)

        self.must_be_rebuilt = False

    def save_to_h5(self, path: str | Path) -> None:
        """
        save_to_h5 Save data collected by defined monitor objects to hdf5.

        Each Monitor gets a separate group, and each group contains all the data the monitor collected.

        Parameters
        ----------
        path : str
            Path to save the files at.

        Raises
        ------
        ValueError
            When the given path does not exist
        RuntimeError
            When no monitors are defined.
        """
        pathname = Path(path)

        self.logger.info(f"save data to hdf5 at {pathname}")

        if len(self.monitors) == 0:
            message = "Error, cannot save data when no monitors are defined"
            self.logger.critical(message)
            raise RuntimeError(message)

        try:
            with h5py.File(pathname, "w") as h5file:
                for key in self.monitors:
                    self.logger.debug(f"save data of monitor {key}")
                    monitor = self.network[key]
                    group = h5file.create_group(monitor.name)
                    for v in monitor.record_variables:
                        try:
                            shape = getattr(monitor, v).shape
                        except AttributeError:
                            # workaround for https://github.com/brian-team/brian2/issues/1662
                            shape = getattr(monitor, v).__array__().shape
                        group.create_dataset(v, shape=shape, data=getattr(monitor, v))
        except FileNotFoundError as e:
            message = f"Path {pathname} does not exist"
            self.logger.critical(message)
            raise ValueError(message) from e

    def _get_element(self, name: str, dct: dict[str, Any]) -> dict[str, Any]:
        return {"definition": dct[name], "object": self.network[name]}

    def get_neurongroup(self, name: str) -> dict[str, Any]:
        """
        Get a object from the neuron network

        Parameters
        ----------
        name : str
            Name of object to get from network

        Returns
        -------
        dict[str, Any]
            dictionary returning the definition and the neuron object
        """
        return self._get_element(name, self.neurongroups)

    def get_synapse(self, name: str) -> dict[str, Any]:
        """
        Get a object from the neuron network

        Parameters
        ----------
        name : str
            Name of object to get from network

        Returns
        -------
        dict[str, Any]
            dictionary returning the definition and the synapse object
        """
        return self._get_element(name, self.synapses)

    def get_monitor(self, name: str) -> dict[str, Any]:
        """
        Get a object from the neuron network

        Parameters
        ----------
        name : str
            Name of object to get from network

        Returns
        -------
        dict[str, Any]
            dictionary returning the definition and the monitor object
        """
        return self._get_element(name, self.monitors)

    def get_model_definition(self) -> dict[str, Any]:
        """
        Return the entire model definition as a dictionary.
        This returns a copy of the definition, i.e., the model
        definition cannot be edited via the returned object

        Returns
        -------
        dict[str, Any]
            Copy of the entire model definition.
            Has entries for 'neurongroups', 'synapses', 'monitors', 'parameters' and 'initial_conditions'
        """
        return {
            "neurongroups": deepcopy(self.neurongroups),
            "synapses": deepcopy(self.synapses),
            "monitors": deepcopy(self.monitors),
            "parameters": deepcopy(self.parameters),
            "initials": deepcopy(self.initials),
            "seed": self.seed,
        }

    def get_variables_of(
        self,
        name: str,
    ) -> list[str] | None:
        """
        Get the set of variables defined in the network element 'name'.

        Parameters
        ----------
        name : str
            Name of object to get variables for

        Returns
        -------
        list[str] | None
            List of variables of the element 'name'.
        """

        try:
            return list(self.network[name].variables)
        except Exception as e:
            self._handle_exception(
                e, f"Error, something went wrong when getting variables for {name}: {e}"
            )

    def get_equation_names_of(
        self,
        name: str,
    ) -> list[str] | None:
        """
        Get equation variables for

        Parameters
        ----------
        name : str
            Name of object to get variables for

        Returns
        -------
        list[str]
            List of variables defined in the equations for the element 'name'.
        """
        if name not in self.synapses and name not in self.neurongroups:
            raise KeyError(
                f"Error, name {name} is neither synapse nor neurongroup and hence does not have equations to get variables from"
            )

        try:
            return list(self.network[name].equations.names)
        except Exception as e:
            self._handle_exception(
                e,
                f"Error, something went wrong when getting equation variables for {name}: {e}",
            )

    def get_current_network_state(
        self,
        read_only_variables: bool = False,
        subexpressions: bool = False,
        vars: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get the current internal state of the underlying brian2 network

        Parameters
        ----------
        read_only_variables : bool, optional
            whether to include read-only variables, by default False
        subexpressions : bool, optional
            whether to include subexpressions, by default False
        vars : list[str] | None, optional
            list of variables to display. lists all if None, by default None

        Returns
        -------
        dict[str, Any]
            Dictionary representing the internal state of the network, including a lot of brian2 internals

        Raises
        ------
        AttributeError
            When the network has not yet been assembled
        """
        return self.network.get_states(
            read_only_variables=read_only_variables,
            subexpressions=subexpressions,
            vars=vars,
        )

    def get_network_objects(self) -> set[Any]:
        """
        Return the set of currently defined objects in the model

        Returns
        -------
        set[Any]
            The network objects
        """
        return self.network.objects

    def _make_timed_arrays_valid(self):
        """
        Ensure any TimedArrays have the correct shape

        For now, we just ensure that they have the same number of neurons as the largest NeuronGroup in the model.
        """
        target_n_neurons = 0
        for neuron_group in self.neurongroups.values():
            target_n_neurons = max(
                target_n_neurons, neuron_group.get("neuronnumber", 0)
            )
        for name, timed_array in self.parameters.items():
            if isinstance(timed_array, b2.TimedArray) and timed_array.values.ndim > 1:
                n_neurons = timed_array.values.shape[1]
                if n_neurons != target_n_neurons:
                    if n_neurons < target_n_neurons:
                        self.logger.info(
                            f"Increasing neurons in TimedArray {name} from {n_neurons} to {target_n_neurons}"
                        )
                        new_array = np.pad(
                            timed_array.values,
                            pad_width=((0, 0), (0, target_n_neurons - n_neurons)),
                        )
                    else:
                        self.logger.info(
                            f"Decreasing neurons in TimedArray {name} from {n_neurons} to {target_n_neurons}"
                        )
                        new_array = timed_array.values[:, :target_n_neurons]
                    new_timed_array = utils.timed_array_with_new_data(
                        timed_array, new_array
                    )
                    self.define_parameter(name, new_timed_array)

    def pad_timed_arrays(self, total_time_seconds: float):
        """
        Pad all TimedArrays in the model to a length of total_time_seconds
        """
        for name, timed_array in self.parameters.items():
            if isinstance(timed_array, b2.TimedArray) and timed_array.values.ndim > 1:
                n_timestamps = timed_array.values.shape[0]
                array_time_secs = timed_array.dt * n_timestamps
                padding_required_secs = total_time_seconds - array_time_secs
                if padding_required_secs > 0:
                    n_pad = math.ceil(padding_required_secs / timed_array.dt)
                    new_array = np.pad(
                        timed_array.values, pad_width=((0, n_pad), (0, 0))
                    )
                    self.logger.info(
                        f"Adding {padding_required_secs}s padding to {name}: {timed_array.values.shape} -> {new_array.shape}"
                    )
                    new_timed_array = utils.timed_array_with_new_data(
                        timed_array, new_array
                    )
                    self.define_parameter(name, new_timed_array)

    def trim_timed_arrays(self, total_time_seconds: float):
        """
        Trim all TimedArrays in the model to a length of total_time_seconds
        """
        for name, timed_array in self.parameters.items():
            if isinstance(timed_array, b2.TimedArray) and timed_array.values.ndim > 1:
                n_timestamps = timed_array.values.shape[0]
                new_n_timestamps = int(total_time_seconds / timed_array.dt)
                if new_n_timestamps < n_timestamps:
                    new_array = timed_array.values[:new_n_timestamps, :]
                    self.logger.info(
                        f"Reducing {name} number of timestamps: {n_timestamps} -> {new_n_timestamps}"
                    )
                    new_timed_array = utils.timed_array_with_new_data(
                        timed_array, new_array
                    )
                    self.define_parameter(name, new_timed_array)

    def trim_timed_array_would_delete_data(self, total_time_seconds: float) -> bool:
        """
        Check whether trimming all TimedArrays in the model to a length of total_time_seconds would delete data
        """
        for timed_array in self.parameters.values():
            if isinstance(timed_array, b2.TimedArray) and timed_array.values.ndim > 1:
                new_n_timestamps = int(total_time_seconds / timed_array.dt)
                if np.sum(timed_array.values[new_n_timestamps:, :]) > 0:
                    return True
        return False

    def run_simulation(
        self,
        tfinal: b2.Quantity,
        savepath: str | Path | None = None,
        report: Callable | None = None,
        timed_array_padding_seconds: int | None = None,
    ) -> None:
        """
        run_simulation Run the fully defined simulation

        Only networks with a fully defined system (neurongroups, possibly synapses, model equations and parameters) can be run.

        Parameters
        ----------
        tfinal : b2.Quantity
            The runtime of the system
        savepath : str, optional
            Path to save the data at, by default None
        """
        if self.must_be_rebuilt:
            self.network = b2.Network(name=self.network.name)
            self.assemble_network()

        if len(self.checkpoints) == 0:
            self.checkpoints.append("Initial state")
            self.network.store(self.checkpoints[-1])
        self.network.run(
            tfinal,
            report=report,
            report_period=0.5 * b2.second,
            namespace=self.parameters,
        )
        self.logger.info(f"run the system for {tfinal} with savepath = {savepath}")
        total_time_seconds = self.network.t / b2.second
        if timed_array_padding_seconds is not None:
            total_time_seconds += timed_array_padding_seconds
        self.pad_timed_arrays(total_time_seconds=total_time_seconds)
        if savepath is not None:
            self.save_to_h5(savepath)

    def store(self, checkpoint: str) -> None:
        """
        store Store a named checkpoint of the current network state.

        Parameters
        ----------
        checkpoint : str
            Name of the checkpoint to store. If the name already exists, the checkpoint will be overwritten.
        """
        self.logger.info(f"store network state at checkpoint {checkpoint}")
        self.network.store(checkpoint)
        if checkpoint not in self.checkpoints:
            self.checkpoints.append(checkpoint)

    def restore(self, checkpoint: str) -> None:
        """
        restore Restore the network to a previously stored named checkpoint.

        Parameters
        ----------
        checkpoint : str
            Name of the checkpoint to restore to.
        """
        self.logger.info(f"restore network to checkpoint {checkpoint}")
        try:
            self.network.restore(checkpoint)
        except Exception as e:
            message = f"Failed to restore checkpoint '{checkpoint}'"
            self.logger.critical(message)
            self.logger.exception(e)
            raise ValueError(message) from e

    def get_data_names(
        self, data_type: type[TraceData | SpikeData | SynapseData]
    ) -> list[str]:
        """
        Get names of all data of the given type

        Parameters
        ----------
        data_type : type[TraceData | SpikeData | SynapseData]
            Type of data to get names for

        Returns
        -------
        list[str]
            List of data names that can be passed to get_data()
        """
        data_names = []
        if data_type is TraceData:
            data_names.extend(self.get_simulation_trace_names())
        elif data_type is SpikeData:
            data_names.extend(self.get_simulation_spike_names())
        elif data_type is SynapseData:
            data_names.extend(self.get_simulation_synapse_names())
        for experiment_name, data_list in self.experiments.items():
            for datum in data_list:
                if isinstance(datum, data_type):
                    data_names.append(f"{experiment_name} :: {datum.name}")
        return data_names

    def get_data(
        self,
        name: str,
        data_type: type[DataType],
    ) -> DataType | None:
        """
        Get experimental data from the model

        Parameters
        ----------
        name : str
            Name of the data to get, as provided by get_data_names()
        data_type : type[TraceData | SpikeData | SynapseData]
            Type of data to get

        Returns
        -------
        TraceData | SpikeData | SynapseData | None
            The requested data, or None if not found
        """
        if name == "":
            return None
        self.logger.info(f"Getting '{name}' of type {data_type}")
        experiment_name, data_name = name.split(" :: ")
        if experiment_name == "Simulation":
            if data_type is TraceData:
                return self.get_simulation_trace_data(name)  # type: ignore
            if data_type is SpikeData:
                return self.get_simulation_spike_data(name)  # type: ignore
            if data_type is SynapseData:
                return self.get_simulation_synapse_data(
                    name, self.max_ram_synapse_plot_matrix
                )  # type: ignore
        if experiment_name not in self.experiments:
            self.logger.info(f"  - experiment '{experiment_name}' not found")
            return None
        for data in self.experiments[experiment_name]:
            if data.name == data_name and isinstance(data, data_type):
                return data
        self.logger.info(f"  - data '{data_name}' not found")
        return None

    def add_experimental_data(
        self,
        experiment_name: str,
        new_data: Sequence[TraceData | SpikeData | SynapseData],
        append: bool = False,
    ):
        """
        Add experimental data to the model

        Parameters
        ----------
        experiment_name : str
            Name of the experiment
        new_data : Sequence[TraceData | SpikeData | SynapseData]
            List of data objects to add
        append : bool, optional
            Whether to append to existing data for the experiment, by default False
        """
        if not append:
            self.logger.info(
                f"Adding experimental data for experiment '{experiment_name}'"
            )
            self.experiments[experiment_name] = list(new_data)
            return
        if experiment_name not in self.experiments:
            self.logger.warning(
                f"Append requested but experiment '{experiment_name}' does not exist yet - creating"
            )
            self.experiments[experiment_name] = []
        self.logger.info(
            f"Appending experimental data to experiment '{experiment_name}'"
        )
        # (for now) automatically get time offset when appending from existing trace data for this experiment
        existing_trace_data = next(
            (d for d in self.experiments[experiment_name] if isinstance(d, TraceData)),
            None,
        )
        if existing_trace_data is not None:
            time_offset_seconds = (
                existing_trace_data.dt * existing_trace_data.traces.shape[1]
            )
            self.logger.info(
                f"Setting time offset to {time_offset_seconds}s from existing trace data '{existing_trace_data.name}'"
            )
        else:
            self.logger.warning(
                f"No existing trace data found for experiment '{experiment_name}', setting time offset to 0s"
            )
            time_offset_seconds = 0.0
        new_min_stim_array_length_seconds = time_offset_seconds
        # match data by type and name and append
        for data in new_data:
            if isinstance(data, TraceData):
                new_min_stim_array_length_seconds = max(
                    new_min_stim_array_length_seconds,
                    time_offset_seconds + data.dt * data.traces.shape[1],
                )
            existing_datum = next(
                (
                    d
                    for d in self.experiments[experiment_name]
                    if d.name == data.name and isinstance(d, type(data))
                ),
                None,
            )
            if existing_datum is not None:
                self.logger.info(f"  - appending experimental data to '{data.name}'")
                append_data(existing_datum, data, time_offset_seconds)
            else:
                self.logger.warning(
                    f"'{data.name}' not found when trying to append - addding new data instead of appending."
                )
                self.experiments[experiment_name].append(data)
        self.pad_timed_arrays(total_time_seconds=new_min_stim_array_length_seconds)

    def get_simulation_trace_names(self) -> list[str]:
        """
        Get the names of all recorded traces in the simulation

        Returns
        -------
        list[str]
            List of trace names
        """
        trace_names = []
        self.trace_name_to_monitor = {}
        for m in self.network.objects:
            if isinstance(m, b2.StateMonitor) and isinstance(m.source, b2.NeuronGroup):
                for variable in m.record_variables:
                    trace_name = f"Simulation :: {m.source.name} / {variable}"
                    trace_names.append(trace_name)
                    self.trace_name_to_monitor[trace_name] = m
        return trace_names

    def get_simulation_trace_data(self, trace_name: str) -> TraceData | None:
        """
        Get trace data for the given trace name

        Parameters
        ----------
        trace_name : str
            Name of the trace as returned by get_simulation_trace_names()

        Returns
        -------
        TraceData
            the recorded data
        """
        self.logger.info(f"Getting simulation trace data for '{trace_name}'")
        neuron_group, variable_name = trace_name.split(" / ")
        monitor = self.trace_name_to_monitor.get(trace_name)
        if monitor is None:
            return None
        timestamps = monitor.t / b2.second
        if timestamps.shape[0] < 2:
            return None
        dt = timestamps[1] - timestamps[0]
        return TraceData(
            trace_name, monitor.variables[variable_name].get_value().T, "unitsTODO", dt
        )

    def get_simulation_spike_names(self) -> list[str]:
        """
        Get the names of all recorded spikes in the simulation

        Returns
        -------
        list[str]
            List of spike names
        """
        spike_names = []
        self.spike_name_to_monitor = {}
        for m in self.network.objects:
            if isinstance(m, b2.SpikeMonitor):
                spike_name = f"Simulation :: {m.source.name}"
                self.spike_name_to_monitor[spike_name] = m
                spike_names.append(spike_name)
        return spike_names

    def get_simulation_spike_data(self, spike_name: str) -> SpikeData | None:
        """
        Get spike data for the given spike name

        Parameters
        ----------
        spike_name : str
            Name of the spike as returned by get_simulation_spike_names()

        Returns
        -------
        SpikeData
            the recorded data
        """
        monitor = self.spike_name_to_monitor[spike_name]
        spike_indices = monitor.variables["i"].get_value().T
        spike_times = monitor.t / b2.second
        return SpikeData(spike_name, spike_indices, spike_times)

    def get_simulation_synapse_names(self) -> list[str]:
        """
        Get the names of all recorded synapse data in the simulation

        Returns
        -------
        list[str]
            List of synapse names
        """
        synapse_names = []
        self.synapse_name_to_monitor = {}
        for m in self.network.objects:
            if isinstance(m, b2.StateMonitor) and isinstance(m.source, b2.Synapses):
                for variable in m.record_variables:
                    synapse_name = f"Simulation :: {m.source.name} / {variable}"
                    synapse_names.append(synapse_name)
                    self.synapse_name_to_monitor[synapse_name] = m
        return synapse_names

    def get_simulation_synapse_data(
        self, synapse_name: str, max_ram_mb: int
    ) -> SynapseData | None:
        """
        Get synapse data for the given name

        Parameters
        ----------
        synapse_name : str
            Name of the synapse as returned by get_simulation_synapse_names()

        Returns
        -------
        SynapseData
            the recorded data
        """
        monitor = self.synapse_name_to_monitor[synapse_name]
        synapse, variable_name = synapse_name.split(" / ")
        values, timestamps = construct_synapse_matrix(
            monitor, variable_name, max_ram_mb
        )
        return SynapseData(synapse_name, values, timestamps)

    def to_dict(self) -> dict[Any, Any]:
        states = self.network.get_states(read_only_variables=False)
        # remove monitors from state to avoid saving their recorded data
        for key in self.monitors:
            del states[key]
        states["current_time"] = self.network.t
        states["seed"] = self.seed
        states["parameters"] = self.parameters
        states["initials"] = self.initials
        states["checkpoints"] = self.checkpoints
        states["objects"] = {
            "neurongroups": list(self.neurongroups.keys()),
            "synapses": list(self.synapses.keys()),
            "monitors": list(self.monitors.keys()),
        }

        # add minimal defining data for monitors
        for key, n in self.monitors.items():
            states[key] = n

        # augment neurongroups and synapses
        for group in [self.neurongroups, self.synapses]:
            for key, n in group.items():
                if key not in states:
                    states[key] = {}
                for k, v in n.items():
                    states[key][k] = v

        return states

    def save_model(
        self,
        filepath: str | Path,
        include_simulation_data: bool = True,
        include_experimental_data: bool = True,
    ) -> None:
        """
        Saves the model to a zipfile which contains
            - a model.json file representing the model
            - an optional experimental_data.json file representing the experimental data
            - an optional pickle file for the stored_state containing the simulation checkpoints
            - npy files for each numpy array in any TimedArray, TraceData or SpikeData objects

        Parameters
        ----------
        filepath : str
            path to the file where the current model should be stored
        """

        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            model_json = json.dumps(
                self.to_dict(), cls=b2json_encoder_factory(zf), separators=(",", ":")
            )
            zf.writestr("model.json", model_json)
            if include_simulation_data:
                # store the current simulation data state to a temporary checkpoint to be able to restore it on load
                self.store(save_store_name)
                zf.writestr(
                    "stored_state.pkl",
                    pickle.dumps(self.network._stored_state, protocol=4),
                )
            if include_experimental_data:
                experimental_data_json = json.dumps(
                    self.experiments,
                    cls=b2json_encoder_factory(zf),
                    separators=(",", ":"),
                )
                zf.writestr("experimental_data.json", experimental_data_json)

    @classmethod
    def load_model(
        cls,
        filepath: str | Path,
        report: Callable[[int], None] | None = None,
    ) -> B2Model:
        """
        Load model from file

        Parameters
        ----------
        filepath : str
            path to the model file
        report : Callable[[int],None], optional
            function to report progress, receives percentage complete as integer from 0 to 100
        """
        if report is not None:
            report(1)

        constr = Reconstructor(filepath)
        if report is not None:
            report(10)

        model = cls(seed=constr.loaded_data["seed"])
        if report is not None:
            report(20)

        constr.reconstruct(model, report)

        if report is not None:
            report(100)

        return model
