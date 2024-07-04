from typing import Any
from typing import Callable
from typing import cast

import brian2 as b2
import yaml
from brian2 import _numpy as np


def process_arrays(data: np.ndarray) -> list:
    """
    Process an array that can be zero-dimensional into a list that then can be processed with YAML

    Parameters
    ----------
    data : np.ndarray
        data array to process

    Returns
    -------
    list
        List version of the input array
    """
    # Adjusted to ensure it returns a list suitable for YAML representation
    if data.shape == ():
        return [
            data.item(),
        ]

    else:
        return data.tolist()


def quantity_representer(
    dumper: yaml.SafeDumper, data: b2.Quantity
) -> yaml.MappingNode:
    """
    Serialize a brian2 Quantity object to yaml

    Parameters
    ----------
    dumper : yaml.SafeDumper
        yaml dumper to use for saving the serialized data
    data : b2.Quantity
        The input quantity to serialize

    Returns
    -------
    yaml.MappingNode
        yaml mappingnode representing the quantity as a value - unit dict under the tag !Quantity
    """
    value = float(data.item()) if data.shape == () else data.tolist()

    return dumper.represent_mapping(
        "!Quantity",
        {
            "value": value,
            "unit": b2.get_dimensions(data)._str_representation(python_code=True),
        },
    )


def timedarray_representer(
    dumper: yaml.SafeDumper, data: b2.TimedArray
) -> yaml.MappingNode:
    """
    Serialize a brian2 TimedArray object to yaml

    Parameters
    ----------
    dumper : yaml.SafeDumper
        yaml dumper object to use for saving serialized data
    data : b2.TimedArray
        The TimedArray to serialize

    Returns
    -------
    yaml.MappingNode
        yaml MappingNode representing the TimedArray as a dict containing [values, unit, dt, name]
    """
    return dumper.represent_mapping(
        "!TimedArray",
        {
            "values": process_arrays(data.values),
            "unit": b2.get_dimensions(data)._str_representation(python_code=True),
            "dt": data.dt,
            "name": data.name,
        },
    )


def numpy_representer(dumper: yaml.SafeDumper, data: np.ndarray) -> yaml.SequenceNode:
    """
    Represent pure numpy arrays incl. 0-dimensional ones in a way that they can be loaded back into numpy directly

    Parameters
    ----------
    dumper : yaml.SafeDumper
        yaml dumper object to use for saving serialized data
    data : np.ndarray
        numpy array to serialize

    Returns
    -------
    yaml.SequenceNode
        YAMl node that represents the serialized array
    """
    treated_data = process_arrays(data)
    node_data = [dumper.represent_data(item) for item in treated_data]

    return yaml.nodes.SequenceNode(tag="!Numpy", value=node_data, flow_style=False)


def quantity_constructor(
    loader: yaml.SafeLoader, node: yaml.MappingNode
) -> b2.Quantity:
    """
    Reconstruct a brian2 quantity from its serialized representation

    Parameters
    ----------
    loader : yaml.SafeLoader
        loader to get the serialized data
    node : yaml.MappingNode
        MappingNode read from file that contains the serialized data

    Returns
    -------
    b2.Quantity
        Reconstructed Quantity
    """
    value = loader.construct_mapping(node)
    return b2.Quantity(value["value"], eval(value["unit"]))


def timedarray_constructor(
    loader: yaml.SafeLoader, node: yaml.MappingNode
) -> b2.TimedArray:
    """
    Reconstruct a brian2 TimedArray from its serialized representation

    Parameters
    ----------
    loader : yaml.SafeLoader
        loader to get the serialized data
    node : yaml.MappingNode
        MappingNode read from file that contains the serialized data

    Returns
    -------
    b2.TimedArray
        Reconstructed TimedArray
    """
    read_data = loader.construct_mapping(node)

    return b2.TimedArray(
        b2.Quantity(read_data["values"], eval(read_data["unit"])),
        read_data["dt"] * b2.second,
        name=read_data["name"],
    )


def numpy_constructor(loader: yaml.SafeLoader, node: yaml.SequenceNode) -> np.ndarray:
    """
    Reconstruct a numpy array from its serialized representation

    Parameters
    ----------
    loader : yaml.SafeLoader
        loader to get the serialized data
    node : yaml.SequenceNode
        Yaml node read from file that contains the serialized data

    Returns
    -------
    np.ndarray
        Reconstructed array
    """
    read_data = loader.construct_sequence(node)
    return np.array(read_data)


"""
Add representers and constructors to the safe dumpers and loaders of yaml
"""
yaml.add_representer(b2.Quantity, quantity_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.ndarray, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(b2.TimedArray, timedarray_representer, Dumper=yaml.SafeDumper)


yaml.add_constructor(
    "!Quantity",
    cast(Callable[[yaml.SafeLoader, yaml.Node], Any], quantity_constructor),
    Loader=yaml.SafeLoader,
)
yaml.add_constructor(
    "!TimedArray",
    cast(Callable[[yaml.SafeLoader, yaml.Node], Any], timedarray_constructor),
    Loader=yaml.SafeLoader,
)
yaml.add_constructor(
    "!Numpy",
    cast(Callable[[yaml.SafeLoader, yaml.Node], Any], numpy_constructor),
    Loader=yaml.SafeLoader,
)
