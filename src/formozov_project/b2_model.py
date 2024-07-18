from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import brian2 as b2
import h5py

from formozov_project import b2json
from formozov_project import utils
from formozov_project.reconstructor import Reconstructor


class B2Model:
    """
    Wrapper class that bundles brian2 capabilities for modelling neuronal networks. This creates a dict-based intermediate represenation
    of the model. All function attributes that start with `define_` act on this intermediate representation and add elements to it,
    all function attributes that are called `delete_` delete elements from it. Function attributes starting with `assemble_` create
    elements in the underlying brian2 network from the intermediate dict, while attributes starting with `erase_` remove them again.
    """

    def _trigger_rebuild(func: Callable) -> Callable:  # type: ignore
        def do_with_rebuild(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.must_be_rebuilt = True

        return do_with_rebuild

    def _add_equation_to_group(self, group: dict, line: str) -> None:
        r"""
        _add_equation_to_group Add an equation to a neurongroup or synapse dictionary that holds arguments to construct the respective objects.

        Parameters
        ----------
        group : dict
            Argument dictionary for constructing a brian2 group object - neurongroup or synapse.
        line : str
            An equation of the form v = f(y,x): unit, possibly containing multiple lines for multiple variables.

        Raises
        ------
        ValueError
            When there is no ': unit' block in the line given
        ValueError
            When there is no '=' block in the line given
        """
        if ":" not in line and "=" not in line:
            message = "Equation must have a colon that separates equation and unit or an equal sign"
            self.logger.critical(message)
            raise ValueError(message)

        if "model" in group:
            group["model"] += line + "\n"
        else:
            group["model"] = line + "\n"

        self.logger.info(f"Equation defined for group {group}")
        self.logger.debug(f"full equation: {line}")

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
        self.parameters: dict[str, Any] | None = None
        self.neurongroups: dict[str, dict[str, Any]] = {}
        self.synapses: dict[str, dict[str, Any]] = {}
        self.monitors: dict[str, dict[str, Any]] = {}
        self.initials: dict[str, dict[str, Any]] = {}
        self.dt: b2.Quantity | None = None
        self.checkpoints: list[str] = []
        self.must_be_rebuilt = False
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model creation complete")

    @property
    def name(self) -> str:
        return self.network.name

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

        self._add_equation_to_group(self.neurongroups[name], line)

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
        if name not in self.initials:
            self.initials[name] = {}

        utils.update_nested_dict(
            self.initials[name],
            {
                varname: value,
            },
        )

    @_trigger_rebuild
    def define_dt(self, dt: b2.Quantity) -> None:
        """
        define_dt Define the global timestep for the default clock. currently, All elements of the network use the same clock with the same timestep.

        Parameters
        ----------
        dt : b2.Quantity
            timestep to be used. Must be given as value * unit.
        """
        self.dt = dt
        b2.defaultclock.dt = self.dt
        self.logger.info(f"dt defined: {dt}")

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

    @_trigger_rebuild
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
        if self.parameters is None:
            self.parameters = {}

        self.parameters[name] = value

        self.logger.info(f"Parameter defined: {name}")
        self.logger.debug(f"Parameter value: {value}")

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
        utils.update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "n": n,
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
        if target_name not in self.neurongroups:
            raise KeyError(
                f"Error, target {target_name} could not be found in neurongroups"
            )

        self.logger.info(f"defining target {target_name} for neuron subgroup {name}")
        if name not in self.neurongroups:
            self.neurongroups[name] = {"name": name}

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

        Raises
        ------
        ValueError
            When noncontiguous indices are passed
        """
        self.logger.info(f"defining indices {idxs} for neuron subgroup {name}")
        if name not in self.neurongroups:
            self.neurongroups[name] = {"name": name}

        utils.update_nested_dict(self.neurongroups[name], {"idxs": idxs})

    @_trigger_rebuild
    def assemble_neuronsubgroup(
        self,
        name: str,
    ) -> None:
        """
        assemble_neuronsubgroup Define a subgroup of a neurongroup as a separate element that can be referred to by name.

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
            message = f"Error when trying to create a neuronsubgroup {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        self.logger.info(f"Neuron subgroup defined: {name}")

        if any(obj.name == name for obj in self.network.objects):
            try:
                self.network.remove(sng)
            except Exception as e:
                message = f"Error when remogin existing subgroup {name}: {e}"
                self.logger.critical(message)
                raise RuntimeError(message) from e

        try:
            self.network.add(sng)
        except Exception as e:
            message = f"Error when adding subgroup {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

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
        utils.update_nested_dict(
            self.synapses,
            {
                name: {
                    "origin": origin,
                }
            },
        )

    def _delete_dict_element(
        self, dictionary: dict[Any, Any], groupname: str, varname: str
    ) -> None:
        """
        Delete an element 'varname' in the subdict behind the 'groupname' key in the dictionary.
        Intended to use with the `neurongroups`, `synapses` or `parameters`members of this class.
        Parameters
        ----------
        dictionary : dict[Any, Any]
            dictionary to modify
        groupname : str
            name of the subdictionary
        varname : str
            element name to remove
        """
        try:
            del dictionary[groupname][varname]
        except Exception as e:
            raise KeyError(f"Error when deleting {varname} in {groupname}: {e}") from e

    @_trigger_rebuild
    def delete_neurongroup(self, name: str) -> None:
        """
        delete a neurongroup definition from the intermediate dict

        Parameters
        ----------
        name : str
            Name of the neurongroup to delete from the intermediat dict
        """
        del self.neurongroups[name]

    @_trigger_rebuild
    def delete_synapse(self, name: str) -> None:
        """
        delete a synapse definition from the intermediate dict

        Parameters
        ----------
        name : str
            Name of the synapse to delete from the intermediat dict
        """
        del self.synapses[name]

    @_trigger_rebuild
    def delete_monitor(self, name: str) -> None:
        """
        delete a monitor definition from the intermediate dict

        Parameters
        ----------
        name : str
            Name of the monitor to delete from the intermediat dict
        """
        del self.monitors[name]

    @_trigger_rebuild
    def delete_neurongroup_element(self, groupname: str, varname: str) -> None:
        """
        Delete an element form the definition of a neurongroup. Note that if it has been assembled already,
        you have to remove it using `remove_neurongroup` and redefine it completetly.

        Parameters
        ----------
        groupname : str
            name of the neurongroup for which a parameter should be deleted
        varname : str
            variable name that should be deleted
        """
        self._delete_dict_element(self.neurongroups, groupname, varname)

    @_trigger_rebuild
    def delete_syanpse_element(self, groupname: str, varname: str) -> None:
        """
        Delete an element form the definition of a synapse. Note that if it has been assembled already,
        you have to remove it using `remove_synapse` and redefine it completetly.

        Parameters
        ----------
        groupname : str
            name of the synapse for which a parameter should be deleted
        varname : str
            variable name that should be deleted
        """
        self._delete_dict_element(self.synapses, groupname, varname)

    @_trigger_rebuild
    def delete_monitor_element(self, groupname: str, varname: str) -> None:
        """
        Delete an element form the definition of a monitor. Note that if it has been assembled already,
        you have to remove it using `remove_monitor` and redefine it completetly.

        Parameters
        ----------
        groupname : str
            name of the monitor for which a parameter should be deleted
        varname : str
            variable name that should be deleted
        """
        self._delete_dict_element(self.monitors, groupname, varname)

    def delete_parameter(self, name: str) -> None:
        """
        Delete a parameter from the set of parameters currently available

        Parameters
        ----------
        name : str
            name of the parameter to remove
        """
        try:
            del self.parameters[name]  # type: ignore
        except Exception as e:
            raise KeyError(f"Error when deleting {name}: {e}") from e

    def delete_initial_condition(self, name: str) -> None:
        """
        Delete an element `name` from the initial condition dictionary. Note that if
        initial conditions have already been defined, you have to redefine it in the
        network.

        Parameters
        ----------
        name : str
            Name of the initial condition to remove

        Raises
        ------
        KeyError
            When `name` is not in the initial condition dictionary
        """
        try:
            del self.initials[name]
        except Exception as e:
            raise KeyError(f"Error when deleting {name}: {e}") from e

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
        if name not in self.synapses or "on_pre" not in self.synapses[name]:
            utils.update_nested_dict(
                self.synapses,
                {
                    name: {
                        "on_pre": line + "\n",
                    },
                },
            )
        else:
            self.synapses[name]["on_pre"] += line + "\n"

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
        if name not in self.synapses or "on_post" not in self.synapses[name]:
            utils.update_nested_dict(
                self.synapses,
                {
                    name: {
                        "on_post": line + "\n",
                    },
                },
            )
        else:
            self.synapses[name]["on_post"] += line + "\n"

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
        utils.update_nested_dict(
            self.synapses,
            {
                name: {
                    "delay": delay,
                },
            },
        )

    @_trigger_rebuild
    def define_synapse_connection_probablitiy(self, name: str, p: float) -> None:
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

        utils.update_nested_dict(
            self.synapses,
            {
                name: {
                    "p": p,
                },
            },
        )

    @_trigger_rebuild
    def assemble_synapse(
        self,
        name: str,
    ) -> None:
        """
        assemble_synapse Assemble a synapse called `name` from its set of predefined values.

        You need to define the respective variables first via the `define_synapse_*` functions.

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
                name=name,
                namespace=self.parameters,
            )

        except Exception as e:
            message = f"Error in defining synapse {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        try:
            synapse.connect(
                args.get("connection_condition", None), p=args.get("p", 1.0)
            )
        except Exception as e:
            message = f"Error in connecting synapse {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        if any(obj.name == name for obj in self.network.objects):
            try:
                self.network.remove(self.network[name])
            except Exception as e:
                message = (
                    f"Error when removing existing synapse {name} from the network: {e}"
                )
                self.logger.critical(message)
                raise RuntimeError(message) from e

        try:
            self.network.add(synapse)
        except Exception as e:
            message = f"Error when adding {name} to the network: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

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
            self.monitors[name] = {
                "name": name,
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
            self.monitors[name] = {
                "name": name,
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
            self.monitors[name] = {
                "name": name,
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
    def assemble_monitor(
        self,
        name: str,
    ) -> None:
        """
        assemble_monitor Assemble a new monitor from predefined elements. When a monitor exists that
        in the network that has the same name, it is removed and replaced by the newly assembled one

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
            monitor = getattr(b2, self.monitors[name]["type"])(
                self.network[self.monitors[name]["source"]],
                self.monitors[name].get("variable", None),
                record=True,
                name=self.monitors[name]["name"],
            )
        except Exception as e:
            message = f"Error in defining monitor {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        if any(obj.name == name for obj in self.network.objects):
            try:
                self.network.remove(self.network[name])
            except Exception as e:
                message = f"Error while removing monitor {name} from network: {e}"
                self.logger.critical(message)
                raise RuntimeError(message) from e

        try:
            self.network.add(monitor)
        except Exception as e:
            message = f"Error in adding monitor {name} to network: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

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
        utils.update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "reset": reset,
                },
            },
        )

    @_trigger_rebuild
    def assemble_neurongroup(
        self,
        name: str = "neurongroup",
    ) -> None:
        """
        assemble_neurongroup Assemble a neurongroup called `name` from its set of predefined values.
        When a neurongroup of the same name already exists in the network, it will be removed and replaced with the newly assembled one.
        You need to define the respective variables first via the `define_*` functions.

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
                self.neurongroups[name].get("n", None),
                self.neurongroups[name].get("model", None),
                threshold=self.neurongroups[name].get("th", None),
                refractory=self.neurongroups[name].get("refractory", False),
                method=self.neurongroups[name].get("method", None),
                namespace=self.parameters,
                reset=self.neurongroups[name].get("reset", None),
                name=name,
            )
        except Exception as e:
            message = f"Error in assembling neuron group {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        if any(obj.name == name for obj in self.network.objects):
            try:
                self.network.remove(
                    self.network[name]
                )  # cannot use self._remove_object because we still need the intermediat definition
            except Exception as e:
                message = f"Error when removing neuron group {name} from network: {e}"
                self.logger.critical(message)
                raise RuntimeError(message) from e

        try:
            self.network.add(ng)
        except Exception as e:
            message = f"Error when adding new neuron group {name} to network: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

    def assemble_inital_condition(self, name: str, key: str) -> None:
        """
        Apply the initial condition identified by `key` to the network element called `name`.
        The element to set the initial condition for must be assembled first.

        Parameters
        ----------
        name : str
            name of network element to set the initial condition for, e.g., name of a neurongroup
        key : str
            name of the initial condition as defined by `define_init`.

        Raises
        ------
        ValueError
            When something goes wrong during the initial condition assignment.
        """
        try:
            setattr(self.network[name], key, self.initials[name][key])

        except Exception as e:
            raise ValueError(
                f"Error when initializing {key} of element {name} as {self.initials[name][key]}: {e}"
            ) from e

    def assemble_initial_conditions(self, name: str) -> None:
        """
        Set all the initial conditions that have been defined via 'define_init'.

        Parameters
        ----------
        name : str
            name of the element of the network for which to define an initial condition.

        Raises
        ------
        ValueError
            When something goes wrong during assignment of the initial condition to the variable given
        """
        for k in self.initials[name]:
            self.assemble_inital_condition(name, k)

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
                        group.create_dataset(
                            v, shape=getattr(monitor, v).shape, data=getattr(monitor, v)
                        )
        except FileNotFoundError as e:
            message = f"Path {pathname} does not exist"
            self.logger.critical(message)
            raise ValueError(message) from e

    def assemble_network(self) -> None:
        """
        Assemble the network from its predefined components.
        """
        self.network = b2.Network(name=self.name)

        for n in self.neurongroups:
            if "idxs" not in self.neurongroups[n]:
                self.assemble_neurongroup(n)

        for n in self.neurongroups:
            if "idxs" in self.neurongroups[n]:
                self.assemble_neuronsubgroup(n)

        for s in self.synapses:
            self.assemble_synapse(s)

        for m in self.monitors:
            self.assemble_monitor(m)

        for n in self.initials:
            self.assemble_initial_conditions(n)

        self.must_be_rebuilt = False

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
        return {"definition": self.neurongroups[name], "object": self.network[name]}

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
        return {"definition": self.synapses[name], "object": self.network[name]}

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
        return {"definition": self.monitors[name], "object": self.network[name]}

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
            return list(self.network[name].variables.keys())
        except Exception as e:
            raise RuntimeError(
                f"Error, something went wrong when getting variables for {name}: {e} "
            ) from e

    def get_equation_variables_of(
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
            raise RuntimeError(
                f"Error, something went wrong when getting equation variables for {name}: {e}"
            ) from e

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

    def _remove_object(self, name: str) -> None:
        """
        Remove an object from the neuron network model

        Parameters
        ----------
        name : str
            name of object to be removed
        """

        self.network.remove(self.network[name])

    def erase_neurongroup(self, name: str) -> None:
        """
        Remove a neurongroup from the network

        Parameters
        ----------
        name : str
            name of the neurongroup to remove
        """
        if name not in self.neurongroups:
            raise KeyError(
                f"Error when removing object {name}: no neurongroup of that name exists"
            )

        del self.neurongroups[name]
        self._remove_object(name)

    def erase_synapse(self, name: str) -> None:
        """
        Remove a synapse from the network

        Parameters
        ----------
        name : str
            name of the synapse to remove
        """
        if name not in self.synapses:
            raise KeyError(
                f"Error when removing object {name}: no synapse of that name exists"
            )

        del self.synapses[name]
        self._remove_object(name)

    def erase_monitor(self, name: str) -> None:
        """
        Remove a monitor from the network

        Parameters
        ----------
        name : str
            name of the synapse to remove
        """
        if name not in self.monitors:
            raise KeyError(
                f"Error when removing object {name}: no monitor of that name exists"
            )

        del self.monitors[name]
        self._remove_object(name)

    def run_simulation(
        self,
        tfinal: b2.Quantity,
        savepath: str | Path | None = None,
        report: Callable | None = None,
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
            logging.exception(e)
            raise ValueError(message) from e

    def to_dict(self) -> dict[Any, Any]:
        states = self.network.get_states()
        states["current_time"] = self.network.t

        if self.dt is not None:
            states["timestep"] = self.dt

        states["seed"] = self.seed

        states["objects"] = {}
        states["objects"]["neurongroups"] = list(self.neurongroups.keys())
        states["objects"]["synapses"] = list(self.synapses.keys())
        states["objects"]["monitors"] = list(self.monitors.keys())
        states["parameters"] = self.parameters
        states["initials"] = self.initials

        # augment neurongroups and synapses
        for group in [self.neurongroups, self.synapses]:
            for key, n in group.items():
                if key not in states:
                    states[key] = {}
                for k, v in n.items():
                    states[key][k] = v

        # add minimal defining data for monitors to avoid data writing
        for key, n in self.monitors.items():
            if key not in states:
                states[key] = {}

            for k, v in n.items():
                states[key][k] = v

        return states

    def save_model(self, filepath: str | Path) -> None:
        """
        Save the current state of the network to disk

        Parameters
        ----------
        filepath : str
            path to the yaml file where the current state should be stored
        """
        states = self.to_dict()

        with open(filepath, "w") as file:
            b2json.dump(states, file)

    @classmethod
    def load_model(
        cls, filepath: str | Path, report: Callable[[int], None] | None = None
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
