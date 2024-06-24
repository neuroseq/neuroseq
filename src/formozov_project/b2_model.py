from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Callable

import brian2 as b2
import h5py


class B2Model:
    """
    Wrapper class that bundles brian2 capabilities for modelling neuronal networks.
    """

    def _update_nested_dict(self, original: dict, updates: dict) -> None:
        """
        _update_nested_dict Update nested dictionary with another dictionary.
        This helper function is mainly useful to maintain the argument
        dictionaries for the different network elements.

        Parameters
        ----------
        original : dict
            dictionary to be updated
        updates : dict
            dictionary to update with
        """
        for key, value in updates.items():
            if (
                isinstance(value, dict)
                and key in original
                and isinstance(original[key], dict)
            ):
                self._update_nested_dict(original[key], value)
            else:
                original[key] = value

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
        if ":" not in line:
            message = "Equation must have a colon that separates equation and unit"
            self.logger.critical(message)
            raise ValueError(message)

        if "=" not in line:
            message = "Equation must have an equal sign that separates variable and expression"
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
        self.dt: b2.Quantity | None = None
        self.solver = None
        self.checkpoints: list[str] = []
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model creation complete")

    @property
    def name(self) -> str:
        return self.network.name

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

    def define_name(self, name: str) -> None:
        """
        define_name Define the name of the model

        Parameters
        ----------
        name : str
            Name of the model
        """
        self.network.name = self.name

    def define_parameter(self, name: str, value: str | b2.Quantity) -> None:
        """
        define_parameter Define a parameter for *any* model equation in the network. All elements will get the same set of parameters,
        so make sure to not include doublings. This is *NOT* verified.

        Parameters
        ----------
        name : str
            parameter name

        value : str | b2.Quantity
            value to be used. Can be an expression in the form of a string or a quantity with units.
        """
        if self.parameters is None:
            self.parameters = {}

        self.parameters[name] = value

        self.logger.info(f"Parameter defined: {name}")
        self.logger.debug(f"Parameter value: {value}")

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
        """
        define_solver_method Define the solver method for the neurongroup indicated by `name`.

        Parameters
        ----------
        name : str
            name of the neurongroup to define the solver method for
        method : str
            name of the method to use. Refer to the brian2 documentation for the currently available set of solvers.
        """
        self._update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "method": method,
                },
            },
        )
        self.logger.info(f"Solver method defined: {method}")

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
        """
        define_neuronnumber Define the number of neurons in a group indicated by name.

        Parameters
        ----------
        name : str
            Name of the neurongroup to define the neuronumber for.
        n : int
            Number of neurons the neurongroup should have.
        """
        self._update_nested_dict(
            self.neurongroups,
            {
                name: {
                    "n": n,
                },
            },
        )
        self.logger.info(f"Neuron number defined: {n}")

    def define_neuron_subgroup(
        self, target_name: str, identifier: str, idxs: slice | int | list | tuple
    ) -> None:
        """
        define_neuron_subgroup Define a subgroup of a neurongroup as a separate element that can be referred to by name.

        Parameters
        ----------
        target_name : str
            Name of the neurongroup the subgroup should be defined for.
        identifier : str
            Name that the subgroup should be referred to. Note that the name
            of the subgroup in the network will be different becasue it is
            automatically determined brian2. This name can be accessed by
            accessing the `neurongroup[identifier]['name']` element of the
            calling instance.
        idxs : slice | int | list |"tuple
            Indices of the target neurongroup neurons to add to the subgroup.
        """
        sng = None
        if isinstance(idxs, tuple):
            sng = self.network[target_name][idxs[0] : idxs[1]]
        else:
            sng = self.network[target_name][idxs]

        self.logger.info(f"Neuron subgroup defined: {identifier}")

        try:
            self.network.add(sng)
        except Exception as e:
            raise RuntimeError(
                f"Error when adding subgroup {identifier} of neurongroup {target_name}: {e}"
            ) from e

        self._update_nested_dict(
            self.neurongroups,
            {
                identifier: {
                    "idx": idxs,
                    "name": sng.name,
                    "identifier": identifier,
                    "target": target_name,
                }
            },
        )

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
        self._update_nested_dict(
            self.synapses,
            {
                name: {
                    "target": target,
                }
            },
        )

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
        self._update_nested_dict(
            self.synapses,
            {
                name: {
                    "origin": origin,
                }
            },
        )

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
            self._update_nested_dict(
                self.synapses,
                {
                    name: {
                        "on_pre": line + "\n",
                    },
                },
            )
        else:
            self.synapses[name]["on_pre"] += line + "\n"

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
            self._update_nested_dict(
                self.synapses,
                {
                    name: {
                        "on_post": line + "\n",
                    },
                },
            )
        else:
            self.synapses[name]["on_post"] += line + "\n"

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
        self._update_nested_dict(
            self.synapses,
            {
                name: {
                    "connection_condition": connection_condition,
                },
            },
        )

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

        self._update_nested_dict(
            self.synapses,
            {
                name: {
                    "p": p,
                },
            },
        )

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

        self.network.add(synapse)

    def define_monitor(
        self, name: str, cls: type, group: str, variable: str | None = None
    ) -> None:
        """
        define_monitor Define a monitor for a variable `variable` in the network element `group`.

        Parameters
        ----------
        name : str
            Name of the monitor as referred to in the governing network.
        group : str
            Name of the element of the network the monitor should grab data from.
        variable : str, optional
            Variable of the `group` to monitor, by default None

        Raises
        ------
        RuntimeError
            When something goes wrong while building the monitor
        """
        monitor = None
        self.logger.info(f"adding monitor '{name}'")

        try:
            monitor = cls(
                self.network[group],
                variable,
                record=True,
                name=name,
            )
        except Exception as e:
            message = f"Error in defining monitor {name}: {e}"
            self.logger.critical(message)
            raise RuntimeError(message) from e

        self.network.add(monitor)
        self.monitors[name] = {
            "name": name,
            "type": cls.__name__,
            "source": group,
            "record": True,
        }

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
        self._update_nested_dict(
            self.neurongroups,
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
        """
        assemble_neurongroup Assemble a neurongroup called `name` from its set of predefined values.

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

        self.network.add(ng)

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
        if len(self.checkpoints) == 0:
            self.checkpoints.append("Initial state")
            self.network.store(self.checkpoints[-1])
        self.network.run(tfinal, report=report, report_period=0.5 * b2.second)
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
