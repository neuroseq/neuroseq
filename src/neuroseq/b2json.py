import json
import zipfile
from io import BytesIO
from pathlib import Path

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


def b2json_encoder_factory(zf: zipfile.ZipFile):
    """
    Factory function that creates a JSON encoder class that can handle Brian2 objects.

    The encoder saves any TimedArray it encounters as .npy files in the provided zipfile.
    """

    class B2JSONEncoder(json.JSONEncoder):
        def save_array(self, name: str, array: numpy_.ndarray):
            npy_buffer = BytesIO()
            numpy_.save(npy_buffer, array)
            npy_buffer.seek(0)
            zf.writestr(name, npy_buffer.getvalue())

        def default(self, obj):
            if isinstance(obj, TimedArray):
                self.save_array(obj.name, obj.values)
                return {
                    "!TimedArray": True,
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

    return B2JSONEncoder


def b2_object_hook_factory(zf: zipfile.ZipFile | None):
    """
    Factory function that creates an object_hook that can reconstruct Brian2 objects.

    If a zipfile is provided, the numpy arrays for any TimedArrays are loaded from the
    corresponding .npy file in the provided zipfile, otherwise they are read from the
    dict directly (to support older models where the arrays were included directly in the json).
    """

    def b2_object_hook(dct: dict, _zf=zf):
        if "!Ndarray" in dct:
            return numpy_.array(dct["value"])
        elif "!Quantity" in dct:
            return dct["value"] * eval(dct["unit"])
        elif "!TimedArray" in dct:
            if _zf is not None:
                # get the array from the npy file in the zipfile
                with _zf.open(dct["name"]) as fp:
                    array = numpy_.load(fp)
            else:
                # get the array from the json directly (for backwards compatibility)
                array = numpy_.array(dct["value"])
            return TimedArray(
                array * eval(dct["unit"]), dct["dt"] * second, name=dct["name"]
            )
        return dct

    return b2_object_hook


def dump(obj: dict, filepath: str | Path):
    """
    Dumps a dict representing the model to a zipfile containing a model.json file and a npy file for each TimedArray.
    """
    with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        model_json = json.dumps(
            obj, cls=b2json_encoder_factory(zf), separators=(",", ":")
        )
        zf.writestr("model.json", model_json)


def load(filepath: str | Path) -> dict:
    """
    Returns a dict representing the model from a zipfile containing a model.json file and a npy file for each TimedArray.
    """
    if Path(filepath).suffix == ".json":
        # for backwards compatibility, support loading models from json files
        with open(filepath) as fp:
            return json.load(fp, object_hook=b2_object_hook_factory(None))
    else:
        # default is to load the model from a zipfile
        with (
            zipfile.ZipFile(filepath, "r", zipfile.ZIP_DEFLATED) as zf,
            zf.open("model.json") as fp,
        ):
            return json.load(fp, object_hook=b2_object_hook_factory(zf))
