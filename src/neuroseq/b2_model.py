from __future__ import annotations

import ast
import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import NoReturn

import brian2 as b2
import brian2.core.names
import brian2.devices.device
import h5py
import numpy as np

from neuroseq import utils
from neuroseq.b2_units import display_units
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.io import as_timed_array
from neuroseq.stimulation.stim import StimArray
from neuroseq.utils import construct_synapse_matrix
from neuroseq.utils import get_downsample_factor
from neuroseq.utils import save_store_name

_brian2_find_name = brian2.core.names.find_name


def _find_name(name: str, names: Any = None) -> str:
    """Brian2's find_name, but safe if an object is freed while collecting names.

    Brian2's own version dereferences each weak reference twice, so it raises an
    AttributeError if the object is freed in between, e.g. by the garbage
    collector. TODO: remove once this is fixed in Brian2, see
    https://github.com/brian-team/brian2/issues/1929
    """
    if names is None and name.endswith("*"):
        objects = [ref() for ref in set(brian2.core.names.Nameable.__instances__())]
        names = {
            obj.name for obj in objects if obj is not None and hasattr(obj, "name")
        }
    return _brian2_find_name(name, names)


brian2.core.names.find_name = _find_name
brian2.devices.device.find_name = _find_name


class B2Model:
    """
    Wrapper class that bundles brian2 capabilities for modelling neuronal networks. This creates a dict-based intermediate representation
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

    def _reconstruct_dynamic_object(
        self,
        assembler: str,
        name: str,
        data_of_group: dict[str, Any],
        augmented: list[tuple[str, Callable]],
    ) -> None:
        """
        Reconstruct synapses and neurongroups from their respective dict representations

        Parameters
        ----------
        assembler : str
            Function to assemble the object with: `_assemble_neurongroup` or `_assemble_synapse`
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
                message = f"Error when defining {key} for object {name}: {e}"
                raise RuntimeError(message) from e

        # assemble object
        try:
            getattr(self, assembler)(name)
        except Exception as e:
            message = f"Error when assembling object {name}: {e}"
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
            # check if there is a setter method
            if key in excluded:
                continue
            else:
                element = getattr(self.network[name], key)
                if not isinstance(element, property) or element.fset is None:
                    continue

                try:
                    setattr(self.network[name], key, value)
                except Exception as e:
                    raise RuntimeError(
                        f"Something went wrong when setting attribute {key} in object {name}: {e}"
                    ) from e

    def reconstruct_neurongroups(
        self, name: str, data_of_group: dict[str, Any]
    ) -> None:
        """
        Reconstruct a neurongroup from its dict representation

        Parameters
        ----------
        name : str
            name of the neurongroup to be reconstructed
        data_of_group : dict[str, Any]
            json node that contains the necessary data for reconstruction
        """
        augmented = list(
            zip(
                [
                    "N",
                    "neuronnumber",
                    "model",
                    "th",
                    "refractory",
                    "reset",
                    "method",
                    "dt",
                ],
                [
                    self.define_neuronnumber,
                    self.define_neuronnumber,
                    self.define_equation,
                    self.define_threshold,
                    self.define_refractory,
                    self.define_reset,
                    self.define_solver_method,
                    self.define_dt,
                ],
                strict=True,
            )
        )

        self._reconstruct_dynamic_object(
            "_assemble_neurongroup",
            name,
            data_of_group,
            augmented,
        )

    def reconstruct_neuron_subgroup(
        self, name: str, data_of_group: dict[str, Any]
    ) -> None:
        """
        Reconstruct a neuron subgroup from its dict representation. Requires that the
        target neurongroup has already been constructed.

        Parameters
        ----------
        name : str
            Name of the subgroup to be constructed
        data_of_group : dict[str, Any]
            dict containing the data to construct the subgroup
        """
        self.define_neuronsubgroup_target(name, data_of_group["target"])
        self.define_neuronsubgroup_idxs(name, data_of_group["idxs"])
        if "dt" in data_of_group:
            self.define_dt(name, data_of_group["dt"])

        self._assemble_neuronsubgroup(name)

    def reconstruct_synapses(self, name: str, data_of_group: dict[str, Any]) -> None:
        """
        Reconstruct a synapse from its dict representation.

        Parameters
        ----------
        name : str
            Name of the synapse to be reconstructed
        data_of_group : dict[str, Any]
            json node to reconstruct the synapse from
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
                    "connection_i",
                    "connection_j",
                    "connection_n",
                    "skip_if_invalid",
                    "method",
                    "dt",
                ],
                [
                    self.define_synapse_equation,
                    self.define_synapse_origin,
                    self.define_synapse_target,
                    self.define_synapse_pre,
                    self.define_synapse_post,
                    self.define_synapse_connection,
                    self.define_synapse_connection_probability,
                    self.define_synapse_i,
                    self.define_synapse_j,
                    self.define_synapse_n,
                    self.define_synapse_skip_if_invalid,
                    self.define_synapse_solver_method,
                    self.define_synapse_dt,
                ],
                strict=True,
            )
        )

        self._reconstruct_dynamic_object(
            "_assemble_synapse",
            name,
            data_of_group,
            augmented,
        )

    def reconstruct_monitors(self, name: str, data_of_group: dict[str, Any]) -> None:
        """
        Reconstruct a monitor for a model from its dict representation

        Parameters
        ----------
        name : str
            Name of the monitor to reconstruct
        data_of_group : dict[str, Any]
            json node to reconstruct the monitor from
        """
        self.define_monitor_type(name, data_of_group["type"])
        self.define_monitor_source(name, data_of_group["source"])
        self.define_monitor_variable(name, data_of_group.get("variable"))

        if "dt" in data_of_group:
            self.define_monitor_dt(name, data_of_group["dt"])
            self._assemble_monitor(name)
        else:
            self._assemble_monitor(name)
            self.define_monitor_dt(
                name, self.network[name].clock.dt
            )  # define the timestep from the clock of the constructed monitor to expose it in the intermediate representation

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

    @staticmethod
    def _parse_subgroup_idxs(idxs: Any) -> tuple[int, int]:
        """
        Normalize a subgroup ``idxs`` specification into a ``(start, stop)`` pair.

        Accepts an int (a single neuron), a one- or two-element list/tuple, or a
        string form of any of those (the model editor stores edited idxs as
        strings, e.g. ``"[0, 1]"`` or ``"3"``).

        Parameters
        ----------
        idxs : Any
            The subgroup index specification.

        Returns
        -------
        tuple[int, int]
            The ``(start, stop)`` range (``stop`` exclusive).

        Raises
        ------
        ValueError
            When ``idxs`` cannot be interpreted as an int or a (start, stop) pair.
        """
        if isinstance(idxs, str):
            try:
                idxs = ast.literal_eval(idxs)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"could not parse subgroup idxs {idxs!r}: "
                    "expected an int or a (start, stop) pair"
                ) from e

        if isinstance(idxs, tuple | list):
            if len(idxs) == 1:
                return int(idxs[0]), int(idxs[0]) + 1
            if len(idxs) == 2:
                return int(idxs[0]), int(idxs[1])
            raise ValueError(
                "subgroups have to be contiguous: give an int or a (start, stop) pair"
            )
        if isinstance(idxs, int):
            return idxs, idxs + 1
        raise ValueError(
            f"invalid subgroup idxs {idxs!r}: expected an int or a (start, stop) pair"
        )

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

        try:
            start, stop = self._parse_subgroup_idxs(self.neurongroups[name]["idxs"])
        except ValueError as e:
            self._handle_exception(e, f"Error for subgroup {name}: {e}")

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

    @staticmethod
    def _build_connect_kwargs(args: dict[str, Any]) -> dict[str, Any]:
        """
        Build the keyword arguments for `brian2.Synapses.connect` from a synapse
        definition dict. Values ("-", "", None) are ignored.

        If explicit source/target indices `i`/`j` are given they take precedence
        over `condition`/`p`.

        Index values for `i`/`j` may be a literal index (`"3"`), a literal array
        (`"[0, 1, 2]"`) or a generator expression (`"i"`, `"k for k in range(...)"`).
        Literals are parsed into ints/lists; everything else is left as a string.

        Parameters
        ----------
        args : dict[str, Any]
            Synapse definition dict (an entry of `self.synapses`).

        Returns
        -------
        dict[str, Any]
            Keyword arguments to pass to `synapse.connect(...)`.
        """

        def _clean(value: Any) -> Any:
            if isinstance(value, str):
                return None if value in ("", "-") else value
            return value

        def _as_index(value: Any) -> Any:
            # turn a literal index ("3") or array ("[0, 1, 2]") into the int/list
            # brian2 expects; leave generator/condition expressions as strings.
            if not isinstance(value, str):
                return value
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
            return parsed if isinstance(parsed, (list, tuple, int)) else value

        kwargs: dict[str, Any] = {}

        n = _clean(args.get("connection_n"))
        if n is not None:
            kwargs["n"] = n

        skip = args.get("skip_if_invalid", False)
        if isinstance(skip, str):
            skip = skip.strip().lower() in ("true", "1", "yes")
        if skip:
            kwargs["skip_if_invalid"] = True

        i = _clean(args.get("connection_i"))
        j = _clean(args.get("connection_j"))
        if i is not None or j is not None:
            if i is not None:
                kwargs["i"] = _as_index(i)
            if j is not None:
                kwargs["j"] = _as_index(j)
        else:
            kwargs["condition"] = args.get("connection_condition")
            kwargs["p"] = args.get("p", 1.0)

        return kwargs

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
            synapse.connect(**self._build_connect_kwargs(args))
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
        args = self.neurongroups.get(name)
        self.logger.info(f"assembling neurongroup '{name}'")
        if args is None:
            message = (
                f"Error in assembling neuron group {name}: neuron group is not defined"
            )
            self.logger.critical(message)
            raise RuntimeError(message)

        try:
            ng = b2.NeuronGroup(
                args.get("neuronnumber", None),
                args.get("model", None),
                threshold=args.get("th", None),
                refractory=args.get("refractory", False),
                method=args.get("method", None),
                namespace=self.parameters,
                reset=args.get("reset", None),
                clock=b2.Clock(
                    dt=args.get("dt", b2.defaultclock.dt),
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
        self.network = b2.Network()
        self.parameters: dict[str, Any] = {}
        # optional one-line descriptions per parameter name, shown in the editor
        # and persisted in the .nseq archive
        self.parameter_descriptions: dict[str, str] = {}
        self.neurongroups: dict[str, dict[str, Any]] = {}
        self.synapses: dict[str, dict[str, Any]] = {}
        self.monitors: dict[str, dict[str, Any]] = {}
        self.initials: dict[str, dict[str, Any]] = {}

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
        self.cancel_requested = False
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model creation complete")

    def set_seed(self, new_seed: int) -> None:
        """
        Set the seed for the brian2 RNG system.

        Parameters
        ----------
        seed : int
            Seed value to set
        """
        self.seed = new_seed
        b2.seed(new_seed)
        self.logger.info(f"Set new seed: {new_seed}")

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

    def define_parameter(
        self,
        name: str,
        value: str | b2.Quantity | b2.TimedArray,
        description: str = "",
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
        description : str, optional
            One-line description shown in the model editor and persisted in the
            ``.nseq`` archive. Takes precedence over the shared registry.
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
        if description:
            self.parameter_descriptions[name] = description

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

    def _rescale_subgroups(self, parent: str, old_n: Any, new_n: Any) -> None:
        """
        Proportionally rescale the ``idxs`` of every subgroup that targets
        ``parent`` when the parent's neuron count changes from ``old_n`` to
        ``new_n``. This preserves each subgroup's fraction of the parent (e.g.
        an excitatory/inhibitory split), so resizing a network from the model
        editor keeps the subgroups consistent.

        The rule is ``new_idx = round(old_idx / old_n * new_n)``, clamped to
        ``[0, new_n]`` and kept non-empty for subgroups that were non-empty.

        Parameters
        ----------
        parent : str
            Name of the (full) neurongroup whose size changed.
        old_n, new_n : Any
            Previous and new neuron counts. Non-integer values are ignored.
        """
        try:
            old_n = int(old_n)
            new_n = int(new_n)
        except (TypeError, ValueError):
            return
        if old_n <= 0 or new_n <= 0 or old_n == new_n:
            return

        for sub_name, group in self.neurongroups.items():
            if "idxs" not in group or group.get("target") != parent:
                continue
            try:
                start, stop = self._parse_subgroup_idxs(group["idxs"])
            except ValueError:
                # leave un-parseable idxs untouched; assembly will report it
                continue

            new_start = max(0, min(round(start / old_n * new_n), new_n))
            new_stop = max(new_start, min(round(stop / old_n * new_n), new_n))
            # keep a previously non-empty subgroup non-empty
            if stop > start and new_stop == new_start:
                if new_start < new_n:
                    new_stop = new_start + 1
                else:
                    new_start = new_n - 1

            group["idxs"] = (new_start, new_stop)
            self.logger.info(
                f"Rescaled subgroup {sub_name} idxs {(start, stop)} -> "
                f"{(new_start, new_stop)} (parent {parent}: {old_n} -> {new_n})"
            )

    def _modify_neuronnumber(self, name: str, new_value: Any) -> None:
        """
        Modify callable bound to a neurongroup's ``neuronnumber`` field in the
        model editor. Sets the new value and proportionally rescales the idxs
        of any subgroups targeting this group, then flags a rebuild.
        """
        old_value = self.neurongroups.get(name, {}).get("neuronnumber")
        self.neurongroups[name]["neuronnumber"] = new_value
        self._rescale_subgroups(name, old_value, new_value)
        self.logger.info(f"Neuron number modified: {name} {old_value} -> {new_value}")
        self.set_rebuilt(True)

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
            # custom modify so editing the count cascades to subgroup idxs
            "modify": lambda value, _name=name: self._modify_neuronnumber(_name, value),
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
        self.define_synapse_i(
            name,
            "-",
        )
        self.define_synapse_j(
            name,
            "-",
        )
        self.define_synapse_n(
            name,
            "1",
        )
        self.define_synapse_skip_if_invalid(
            name,
            "False",
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
        self._register_synapse_value(name, "dt", dt)
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
        self._register_synapse_value(name, "method", method)
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
        self._register_synapse_value(name, "target", target)

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
        self._register_synapse_value(name, "origin", origin)

    @_trigger_rebuild
    def define_synapse_equation(self, name: str, line: str) -> None:
        """
        define_synapse_equation Define a line in the model equation for a synapse called `synapse`, or the full equation all at once.
        Have a look at the brian2 documentation for synapses for more details.

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
            String representation of the connection condition.
        """
        self._register_synapse_value(name, "connection_condition", connection_condition)

    def _register_synapse_value(self, name: str, key: str, value: Any) -> None:
        """
        Register a simple scalar/string `value` under `key` for the synapse `name`,
        wiring up the Modifier so the GUI can edit/delete/rename it. Used by the
        `define_synapse_*` methods that just store a value on the synapse dict.

        Parameters
        ----------
        name : str
            Name of the synapse to set the value for.
        key : str
            Key under which the value is stored in the synapse dict.
        value : Any
            Value to store.
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
            key,
            self.synapses[name],
            self.functions["synapses"][name],
            lambda: self.set_rebuilt(True),
        )

        self.functions["synapses"][name][key] = {
            "delete": element_modifier.delete,
            "rename": element_modifier.rename,
            "modify": element_modifier.modify,
        }

        utils.update_nested_dict(self.synapses, {name: {key: value}})

    @_trigger_rebuild
    def define_synapse_i(self, name: str, i: Any) -> None:
        """
        define_synapse_i Define the source indices `i` for a synapse called `name`.

        This is an alternative to a connection condition: `i` may be an explicit
        index (array) or a generator/string expression. When `i`/`j` are set they take
        precedence over `connection_condition`.

        Parameters
        ----------
        name : str
            Name of the synapse to add the source indices for.
        i : Any
            Source index expression, e.g. an int, an array, or a string expression.
        """
        self._register_synapse_value(name, "connection_i", i)

    @_trigger_rebuild
    def define_synapse_j(self, name: str, j: Any) -> None:
        """
        define_synapse_j Define the source indices `j` for a synapse called `name`.

        This is an alternative to a connection condition: `j` may be an explicit
        index (array) or a generator/string expression. When `i`/`j` are set they take
        precedence over `connection_condition`.

        Parameters
        ----------
        name : str
            Name of the synapse to add the target indices for.
        j : Any
            Target index expression, e.g. an int, an array, or a string expression.
        """
        self._register_synapse_value(name, "connection_j", j)

    @_trigger_rebuild
    def define_synapse_n(self, name: str, n: Any) -> None:
        """
        define_synapse_n Define the number of synapses to create per connected pair
        (multapses) for a synapse called `name`.

        Parameters
        ----------
        name : str
            Name of the synapse to set the number of synapses per pair for.
        n : Any
            Number of synapses per pair. May be an int or a string expression that
            depends on `i`/`j`. Please refer to the brian2 documentation.
        """
        self._register_synapse_value(name, "connection_n", n)

    @_trigger_rebuild
    def define_synapse_skip_if_invalid(self, name: str, skip_if_invalid: Any) -> None:
        """
        define_synapse_skip_if_invalid Define whether out-of-range indices produced
        when connecting a synapse called `name` should be silently skipped instead
        of raising an error.

        Parameters
        ----------
        name : str
            Name of the synapse to set the flag for.
        skip_if_invalid : Any
            Whether to skip invalid indices. Accepts a bool or a "True"/"False"
            string (as produced by the GUI).
        """
        self._register_synapse_value(name, "skip_if_invalid", skip_if_invalid)

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
        self._register_synapse_value(name, "delay", delay)

    @_trigger_rebuild
    def define_synapse_connection_probability(self, name: str, p: float | str) -> None:
        """
        define_synapse_connection_probability Define the connection probability for a given synapse called `name`.

        Parameters
        ----------
        name : str Name of the synapse to define the connection probability for.

        p : float | str
            Probability value. Floats must be in [0, 1]. Strings are
            treated as expressions (e.g. ``"exp(-(x_pre - x_post)**2)"``).
        """
        if not isinstance(p, str) and not 0.0 <= p <= 1.0:
            raise ValueError(
                f"Error, supplied numerical value {p} is not a probability."
            )

        self._register_synapse_value(name, "p", p)

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
    def define_monitor_variable(self, name: str, variable: str) -> None:
        """
        define the variable a monitor can be attached to.

        Parameters
        ----------
        name : str
            name of the monitor to define the variable for
        variable : str
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

    def _prune_orphaned_definitions(self) -> None:
        """
        Remove definitions that reference network elements which no longer
        exist. After deleting a neurongroup or synapse in the editor, its
        dependents (subgroups of it, synapses/monitors pointing at it, and its
        initial conditions) would otherwise remain and make the next
        ``assemble_network`` crash on a stale reference. Pruning them keeps the
        model self-consistent (e.g. removing an inhibitory neuron group also
        drops the inhibitory synapse that originated from it).
        """

        def _exists(ref: Any) -> bool:
            return ref in self.neurongroups or ref in self.synapses

        # subgroups whose target neurongroup was removed
        for name in list(self.neurongroups):
            group = self.neurongroups.get(name, {})
            target = group.get("target")
            if (
                "idxs" in group
                and target not in (None, "", "-")
                and target not in self.neurongroups
            ):
                self.logger.warning(
                    f"Pruning subgroup '{name}': target '{target}' no longer exists"
                )
                self._delete_dict_element("neurongroups", name)

        # synapses whose origin/target neurongroup was removed
        for name in list(self.synapses):
            args = self.synapses.get(name, {})
            for endpoint in ("origin", "target"):
                ref = args.get(endpoint)
                if ref not in (None, "", "-") and ref not in self.neurongroups:
                    self.logger.warning(
                        f"Pruning synapse '{name}': {endpoint} '{ref}' no longer exists"
                    )
                    self._delete_dict_element("synapses", name)
                    break

        # monitors whose source was removed
        for name in list(self.monitors):
            source = self.monitors.get(name, {}).get("source")
            if source not in (None, "", "-") and not _exists(source):
                self.logger.warning(
                    f"Pruning monitor '{name}': source '{source}' no longer exists"
                )
                self._delete_dict_element("monitors", name)

        # initial conditions for a deleted element
        for name in list(self.initials):
            if not _exists(name):
                self.logger.warning(
                    f"Pruning initial conditions for '{name}': element no longer exists"
                )
                self._delete_dict_element("initials", name)

    def assemble_network(self) -> None:
        """
        Assemble the network from its predefined components.
        """
        self._prune_orphaned_definitions()

        network_name = self.name
        self.trace_name_to_monitor.clear()
        self.spike_name_to_monitor.clear()
        self.synapse_name_to_monitor.clear()
        self.network = b2.Network(name=network_name)
        self.checkpoints.clear()

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

    def get_network_objects(self) -> set[Any]:
        """
        Return the set of currently defined objects in the model

        Returns
        -------
        set[Any]
            The network objects
        """
        return self.network.objects

    def run_simulation(
        self,
        time_seconds: float,
        stim_arrays: dict[str, StimArray],
        report: Callable | None = None,
        report_period: b2.Quantity = 1.0 * b2.second,
    ) -> None:
        """
        Run the fully defined simulation

        Only networks with a fully defined system (neurongroups, possibly synapses, model equations and parameters) can be run.

        Parameters
        ----------
        time_seconds : float
            The runtime of the system in seconds
        stim_arrays: dict[str, StimArray]
            A dictionary of stimulus arrays to be used in the simulation
        report : Callable | None, optional
            A callable function to report progress, by default None
        report_period : b2.Quantity, optional
            Minimum wall-clock interval between ``report`` calls, by default 1s.
            Pass a smaller value to report progress more frequently.
        """
        self.cancel_requested = False

        if self.must_be_rebuilt:
            self.assemble_network()

        if len(self.checkpoints) == 0:
            self.checkpoints.append("Initial state")
            self.network.store(self.checkpoints[-1])
        namespace = self.parameters | {
            name: as_timed_array(stim_array) for name, stim_array in stim_arrays.items()
        }
        if self.cancel_requested:
            self.logger.info("simulation cancelled before run started")
            return
        self.network.run(
            time_seconds * b2.second,
            report=report,
            report_period=report_period,
            namespace=namespace,
        )
        self.logger.info(f"run the system for {time_seconds}s")

    def request_cancel(self) -> None:
        """Request that an in-progress simulation stop early."""
        self.logger.info("simulation cancellation requested")
        self.cancel_requested = True
        if self.network is not None:
            self.network.stop()

    def store(self, checkpoint: str) -> None:
        """
        Store a named checkpoint of the current network state.

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
        Restore the network to a previously stored named checkpoint.

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

    def get_simulation_trace_data(
        self, trace_name: str, max_ram_mb: int = 0
    ) -> TraceData | None:
        """
        Get trace data for the given trace name

        Parameters
        ----------
        trace_name : str
            Name of the trace as returned by get_simulation_trace_names()
        max_ram_mb : int
            Maximum RAM (MB) for the returned trace matrix. When the recorded
            (neuron x time) matrix would exceed this, it is downsampled along
            the time axis to fit -- mirroring ``construct_synapse_matrix`` for
            the synapse-plot path. ``0`` (the default) means no limit.

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
        quantity = monitor.variables[variable_name]
        values = quantity.get_value().T.astype(np.float32, copy=False)
        n_floats = int(values.shape[0]) * int(values.shape[1])
        downsample_factor = get_downsample_factor(
            n_floats, max_ram_mb, bits_per_float=32
        )
        if downsample_factor > 1:
            self.logger.info(
                f"Downsampling {values.shape[0]}x{values.shape[1]} trace matrix by "
                f"factor {downsample_factor} due to max RAM limit of {max_ram_mb} MB"
            )
            # Copy to a contiguous array so the full-resolution monitor buffer
            # is not kept alive by the strided view we hand to the widget/GPU.
            values = np.ascontiguousarray(values[:, ::downsample_factor])
            timestamps = timestamps[::downsample_factor]
        dt = timestamps[1] - timestamps[0]
        units = display_units(b2.Quantity(1, dim=quantity.dim))
        # Voltage traces are stored in SI volts but conventionally displayed
        # in millivolts throughout the trace widget.
        if quantity.dim == b2.volt.dim:
            units = "mV"
        return TraceData(
            trace_name,
            values,
            monitor.record,
            units,
            dt,
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
        max_ram_mb: int
            Maximum RAM in MB to use when constructing the synapse data matrix

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

    def get_sim_time_secs(self) -> float:
        if self.network is not None:
            return float(self.network.t / b2.second)
        return 0.0

    def has_simulation_data(self) -> bool:
        return self.get_sim_time_secs() > 0 or len(self.checkpoints) > 0

    def get_max_neurons(self) -> int:
        max_neurons = 0
        for obj in self.network.objects:
            if isinstance(obj, b2.NeuronGroup):
                max_neurons = max(max_neurons, obj.N)
        return max_neurons

    def get_defined_neuron_count(self) -> int:
        """
        Largest neuron count across the full neurongroups in the *definition*
        (intermediate representation), independent of whether the underlying
        network has been (re)assembled yet.

        Subgroups (entries with ``idxs``) are skipped, as their size derives
        from their parent group. ``neuronnumber`` values that are not plain
        integers (e.g. symbolic expressions) are ignored.

        Returns
        -------
        int
            The maximum ``neuronnumber`` over all full neurongroups, or 0 if
            none is defined.
        """
        max_neurons = 0
        for name, group in self.neurongroups.items():
            if "idxs" in group:
                continue
            try:
                max_neurons = max(max_neurons, int(group.get("neuronnumber", 0)))
            except (TypeError, ValueError):
                self.logger.debug(
                    f"neurongroup {name} has a non-integer neuronnumber "
                    f"{group.get('neuronnumber')!r}; skipping for stim-array sizing"
                )
        return max_neurons

    def to_dict(self, include_simulation_data: bool = True) -> dict[Any, Any]:
        states = self.network.get_states(read_only_variables=False)
        # remove monitors from state to avoid saving their recorded data
        for key in self.monitors:
            if key in states:
                del states[key]
        states["current_time"] = self.network.t
        states["seed"] = self.seed
        states["parameters"] = self.parameters
        states["parameter_descriptions"] = self.parameter_descriptions
        states["initials"] = self.initials
        states["checkpoints"] = self.checkpoints if include_simulation_data else []
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

    @classmethod
    def from_dict(
        cls,
        b2_model_data: dict,
        b2_stored_state: dict,
        report: Callable[[int], None] | None = None,
    ):
        """
        Create a B2Model from its dict representation.

        Parameters
        ----------
        b2_model_data : dict
            The serialized model data
        b2_stored_state : dict
            The stored simulation state of the model
        report : Callable[[int],None], optional
            function to report progress, receives percentage complete as integer from 0 to 100
        """
        if report is not None:
            report(1)

        if not b2_model_data:
            # Projects that contain no model (e.g. stim-array-only archives)
            # have no serialized b2_model; return a default empty model.
            if report is not None:
                report(100)
            return cls()

        model = cls(seed=b2_model_data.get("seed", 123))
        model.checkpoints = b2_model_data.get("checkpoints", [])

        if report is not None:
            report(20)

        # reconstruct parameters, timestep, seed.
        param_descriptions = b2_model_data.get("parameter_descriptions", {})
        for k, v in b2_model_data["parameters"].items():
            model.define_parameter(k, v, param_descriptions.get(k, ""))

        if report is not None:
            report(30)

        # reconstruct neurongroups
        for n in b2_model_data["objects"]["neurongroups"]:
            if "idxs" not in b2_model_data[n]:  # reconstruct full groups first...
                model.reconstruct_neurongroups(n, b2_model_data[n])
        if report is not None:
            report(40)

        for n in b2_model_data["objects"]["neurongroups"]:
            if "idxs" in b2_model_data[n]:  # ... then define the subgroups
                model.reconstruct_neuron_subgroup(n, b2_model_data[n])
        if report is not None:
            report(50)

        # reconstruct synapses
        for s in b2_model_data["objects"]["synapses"]:
            model.reconstruct_synapses(s, b2_model_data[s])
        if report is not None:
            report(60)

        # reconstruct monitors
        for m in b2_model_data["objects"]["monitors"]:
            model.reconstruct_monitors(m, b2_model_data[m])
        if report is not None:
            report(70)

        for n, init in b2_model_data["initials"].items():
            for k, v in init.items():
                model.define_init(n, k, v)
            model._assemble_initial_conditions(n)

        # restore checkpoints; drop names whose state was not loaded (e.g. the
        # model was imported without simulation data) so they are not offered
        # for restore and "Initial state" is stored again on the next run.
        model.network._stored_state = b2_stored_state
        model.checkpoints = [
            name for name in model.checkpoints if name in b2_stored_state
        ]
        # restore state when saved if provided, then remove this temporary checkpoint
        if save_store_name in model.network._stored_state:
            model.network.restore(save_store_name)
            del model.network._stored_state[save_store_name]

        model.must_be_rebuilt = False

        if report is not None:
            report(100)

        return model
