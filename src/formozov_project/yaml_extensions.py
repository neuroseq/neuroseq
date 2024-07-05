from typing import Any

import yaml
from brian2 import Quantity
from brian2 import TimedArray
from brian2 import amp  # noqa: F401
from brian2 import get_dimensions
from brian2 import kelvin  # noqa: F401
from brian2 import kilogram  # noqa: F401
from brian2 import metre  # noqa: F401
from brian2 import numpy_
from brian2 import second


def process_arrays(data: numpy_.ndarray) -> list:
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
        return numpy_.array(data).tolist()


def quantity_representer(dumper: yaml.SafeDumper, data: Quantity) -> yaml.MappingNode:
    """
    Serialize a brian2 Quantity object to yaml

    Parameters
    ----------
    dumper : yaml.SafeDumper
        yaml dumper to use for saving the serialized data
    data : Quantity
        The input quantity to serialize

    Returns
    -------
    yaml.MappingNode
        yaml mappingnode representing the quantity as a value - unit dict under the tag !Quantity
    """

    value = (
        numpy_.array(data).item() if data.shape == () else numpy_.array(data).tolist()
    )

    return dumper.represent_mapping(
        "!Quantity",
        {
            "value": value,
            "unit": get_dimensions(data)._str_representation(python_code=True),
        },
    )


def timedarray_representer(
    dumper: yaml.SafeDumper, data: TimedArray
) -> yaml.MappingNode:
    """
    Serialize a brian2 TimedArray object to yaml

    Parameters
    ----------
    dumper : yaml.SafeDumper
        yaml dumper object to use for saving serialized data
    data : TimedArray
        The TimedArray to serialize

    Returns
    -------
    yaml.MappingNode
        yaml MappingNode representing the TimedArray as a dict containing [values, unit, dt, name]
    """

    values = (
        numpy_.array(data.values).item()
        if data.values.shape == ()
        else numpy_.array(data.values).tolist()
    )
    return dumper.represent_mapping(
        "!TimedArray",
        {
            "values": values,
            "unit": get_dimensions(data)._str_representation(python_code=True),
            "dt": data.dt,
            "name": data.name,
        },
    )


def numpy_representer(
    dumper: yaml.SafeDumper, data: numpy_.ndarray
) -> yaml.SequenceNode:
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


def quantity_constructor(loader: yaml.SafeLoader, node: Any) -> Quantity:
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
    Quantity
        Reconstructed Quantity
    """
    data = loader.construct_mapping(node, deep=True)

    unit = eval(data["unit"])

    return data["value"] * unit


def timedarray_constructor(loader: yaml.SafeLoader, node: Any) -> TimedArray:
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
    TimedArray
        Reconstructed TimedArray
    """
    read_data = loader.construct_mapping(node, deep=True)

    return TimedArray(
        read_data["values"] * eval(read_data["unit"]),
        read_data["dt"] * second,
        name=read_data["name"],
    )


def numpy_constructor(loader: yaml.SafeLoader, node: Any) -> numpy_.ndarray:
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
    numpy_.ndarray
        Reconstructed array
    """
    read_data = loader.construct_sequence(node)
    return numpy_.array(read_data)


"""
Add representers and constructors to the safe dumpers and loaders of yaml
"""
yaml.add_representer(Quantity, quantity_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(numpy_.ndarray, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(numpy_.ndarray, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(TimedArray, timedarray_representer, Dumper=yaml.SafeDumper)


yaml.add_constructor(
    "!Quantity",
    quantity_constructor,
    Loader=yaml.SafeLoader,
)
yaml.add_constructor(
    "!TimedArray",
    timedarray_constructor,
    Loader=yaml.SafeLoader,
)
yaml.add_constructor(
    "!Numpy",
    numpy_constructor,
    Loader=yaml.SafeLoader,
)
