import yaml
from brian2 import Quantity


def quantity_representer(dumper, data):
    return dumper.represent_mapping(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        {"value": float(data), "unit": str(data.units)},
    )


def quantity_constructor(loader, node):
    value = loader.construct_mapping(node)
    return Quantity(value["value"], value["unit"])


yaml.add_representer(Quantity, quantity_representer)
yaml.add_constructor("!Quantity", quantity_constructor)
