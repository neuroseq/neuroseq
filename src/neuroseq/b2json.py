import json

from brian2 import Quantity
from brian2 import TimedArray

# these imports are needed so that we can do eval("units expression as python code"):
from brian2 import amp  # noqa: F401
from brian2 import asarray
from brian2 import get_dimensions
from brian2 import kelvin  # noqa: F401
from brian2 import kilogram  # noqa: F401
from brian2 import metre  # noqa: F401
from brian2 import numpy_
from brian2 import second


def _obj_unitless_json_value(
    obj: numpy_.ndarray | Quantity | TimedArray,
) -> list | float | int:
    a = asarray(obj)
    return a.item() if a.shape == () else a.tolist()


def _obj_units_as_str(obj: Quantity | TimedArray) -> str:
    return get_dimensions(obj)._str_representation(python_code=True)


class _B2JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TimedArray):
            return {
                "!TimedArray": True,
                "value": _obj_unitless_json_value(obj.values),
                "unit": _obj_units_as_str(obj),
                "dt": obj.dt,
                "name": obj.name,
            }
        elif isinstance(obj, Quantity):
            return {
                "!Quantity": True,
                "value": _obj_unitless_json_value(obj),
                "unit": _obj_units_as_str(obj),
            }
        elif isinstance(obj, numpy_.ndarray):
            return {"!Ndarray": True, "value": _obj_unitless_json_value(obj)}
        return super().default(obj)


def _b2_json_object_hook(dct: dict):
    if "!Ndarray" in dct:
        return numpy_.array(dct["value"])
    elif "!Quantity" in dct:
        return dct["value"] * eval(dct["unit"])
    elif "!TimedArray" in dct:
        return TimedArray(
            dct["value"] * eval(dct["unit"]), dct["dt"] * second, name=dct["name"]
        )
    return dct


def dump(obj: dict, fp):
    json.dump(obj, fp, cls=_B2JSONEncoder, separators=(",", ":"))


def load(fp):
    return json.load(fp, object_hook=_b2_json_object_hook)
