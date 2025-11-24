import json
import zipfile
from io import BytesIO

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

from neuroseq.experiment import SpikeData
from neuroseq.experiment import SynapseData
from neuroseq.experiment import TraceData


def _obj_unitless_json_value(
    obj: numpy_.ndarray | Quantity | TimedArray,
) -> list | float | int:
    a = asarray(obj)
    return a.item() if a.shape == () else a.tolist()


def _obj_units_as_str(obj: Quantity | TimedArray) -> str:
    return get_dimensions(obj)._str_representation(python_code=True)


def _save_array(zf: zipfile.ZipFile, name: str, array: numpy_.ndarray) -> str:
    npy_buffer = BytesIO()
    numpy_.save(npy_buffer, array)
    npy_buffer.seek(0)
    # check if name already exists in the zipfile and if so add a suffix until it is unique
    original_name = name
    suffix_counter = 1
    while name in zf.namelist():
        name = f"{original_name}_{suffix_counter}"
        suffix_counter += 1
    zf.writestr(name, npy_buffer.getvalue())
    return name


def b2json_encoder_factory(zf: zipfile.ZipFile):
    """
    Factory function that creates a JSON encoder class that can handle Brian2 objects.

    The encoder saves any numpy arrays included in TimedArray, TraceData, SynapseData or SpikeData objects it encounters as .npy files in the provided zipfile.
    Any other numpy arrays are included directly in the json.
    """

    class B2JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            match obj:
                case SynapseData():
                    return {
                        "!SynapseData": True,
                        "name": obj.name,
                        "values": _save_array(zf, f"_{obj.name}_values", obj.values),
                        "timestamps": _save_array(
                            zf, f"_{obj.name}_timestamps", obj.timestamps
                        ),
                    }
                case TraceData():
                    return {
                        "!TraceData": True,
                        "name": obj.name,
                        "traces": _save_array(zf, f"_{obj.name}_traces", obj.traces),
                        "units": obj.units,
                        "dt": obj.dt,
                    }
                case SpikeData():
                    return {
                        "!SpikeData": True,
                        "name": obj.name,
                        "spike_indices": _save_array(
                            zf, f"_{obj.name}_spike_indices", obj.spike_indices
                        ),
                        "spike_times": _save_array(
                            zf, f"_{obj.name}_spike_times", obj.spike_times
                        ),
                    }
                case TimedArray():
                    return {
                        "!TimedArray": True,
                        "values": _save_array(zf, obj.name, obj.values),
                        "unit": _obj_units_as_str(obj),
                        "dt": obj.dt,
                        "name": obj.name,
                    }
                case Quantity():
                    return {
                        "!Quantity": True,
                        "value": _obj_unitless_json_value(obj),
                        "unit": _obj_units_as_str(obj),
                    }
                case numpy_.ndarray():
                    return {"!Ndarray": True, "value": _obj_unitless_json_value(obj)}
                case _:
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
        match dct:
            case {"!Ndarray": True, "value": value}:
                return numpy_.array(value)
            case {"!Quantity": True, "value": value, "unit": unit}:
                return value * eval(unit)
            case {"!TimedArray": True}:
                if _zf is not None:
                    # get the array from the npy file in the zipfile
                    # the filename is stored in the "values" key, or in the previous file format the "name" key
                    with _zf.open(dct.get("values", dct.get("name"))) as fp:
                        array = numpy_.load(fp)
                else:
                    # get the array from the json directly (for backwards compatibility with original file format)
                    array = numpy_.array(dct["value"])
                return TimedArray(
                    array * eval(dct["unit"]), dct["dt"] * second, name=dct["name"]
                )
            case {
                "!SpikeData": True,
                "name": name,
                "spike_indices": spike_indices,
                "spike_times": spike_times,
            }:
                if _zf is None:
                    raise RuntimeError(
                        "Cannot load SpikeData without zipfile containing the numpy arrays."
                    )
                with _zf.open(spike_indices) as fp:
                    loaded_spike_indices = numpy_.load(fp)
                with _zf.open(spike_times) as fp:
                    loaded_spike_times = numpy_.load(fp)
                return SpikeData(name, loaded_spike_indices, loaded_spike_times)
            case {
                "!TraceData": True,
                "name": name,
                "traces": traces,
                "units": units,
                "dt": dt,
            }:
                if _zf is None:
                    raise RuntimeError(
                        "Cannot load TraceData without zipfile containing the numpy arrays."
                    )
                with _zf.open(traces) as fp:
                    loaded_traces = numpy_.load(fp)
                return TraceData(name, loaded_traces, units, dt)
            case {
                "!SynapseData": True,
                "name": name,
                "values": values,
                "timestamps": timestamps,
            }:
                if _zf is None:
                    raise RuntimeError(
                        "Cannot load SynapseData without zipfile containing the numpy arrays."
                    )
                with _zf.open(values) as fp:
                    loaded_values = numpy_.load(fp)
                with _zf.open(timestamps) as fp:
                    loaded_timestamps = numpy_.load(fp)
                return SynapseData(name, loaded_values, loaded_timestamps)
            case _:
                return dct

    return b2_object_hook
