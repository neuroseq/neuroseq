import json
import zipfile
from io import BytesIO
from pathlib import Path

import brian2 as b2
import numpy as np

from neuroseq.b2_units import quantity_to_value_unit_str
from neuroseq.b2_units import value_unit_str_to_quantity
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.stim import StimArray


def _save_array(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> str:
    npy_buffer = BytesIO()
    np.save(npy_buffer, array)
    npy_buffer.seek(0)
    # check if name already exists in the zipfile and if so add a suffix until it is unique
    original_name = name
    suffix_counter = 1
    while name in zf.namelist():
        name = f"{original_name}_{suffix_counter}"
        suffix_counter += 1
    zf.writestr(name, npy_buffer.getvalue())
    return name


def _json_encoder_factory(zf: zipfile.ZipFile):
    """
    Factory function that creates a JSON encoder class that can handle neuroseq and Brian2 objects.

    The encoder saves any numpy it encounters as .npy files in the provided zipfile.
    """

    class NeuroseqJSONEncoder(json.JSONEncoder):
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
                        "indices": _save_array(zf, f"_{obj.name}_indices", obj.indices),
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
                case StimArray():
                    return {
                        "!StimArray": True,
                        "name": obj.name,
                        "stim_type": obj.stim_type,
                        "values": _save_array(zf, obj.name, obj.values),
                        "units": obj.units,
                        "dt": obj.dt,
                    }
                case CellposeImportResult():
                    segmentation_path = _save_array(
                        zf, "cellpose_segmentation", obj.segmentation_array
                    )
                    overlay_path = _save_array(
                        zf, "cellpose_overlay", obj.overlay_image
                    )
                    original_ids_path = None
                    if obj.original_cellpose_ids is not None:
                        original_ids_path = _save_array(
                            zf, "cellpose_original_ids", obj.original_cellpose_ids
                        )
                    return {
                        "!CellposeImportResult": True,
                        "image_path": str(obj.image_path),
                        "segmentation_array": segmentation_path,
                        "overlay_image": overlay_path,
                        "trace_data": obj.trace_data,
                        "spike_data": obj.spike_data,
                        "original_cellpose_ids": original_ids_path,
                        "stimulus_to_cellpose_mapping": obj.stimulus_to_cellpose_mapping,
                    }
                case b2.Quantity():
                    value, unit = quantity_to_value_unit_str(obj)
                    if isinstance(value, np.ndarray):
                        # save the array to a .npy file in the zipfile and store the filename in the json instead
                        value = _save_array(zf, "quantity_array", value)
                    return {
                        "!Quantity": True,
                        "value": value,
                        "unit": unit,
                    }
                case np.ndarray():
                    return {"!Ndarray": True, "value": _save_array(zf, "ndarray", obj)}
                case _:
                    return super().default(obj)

    return NeuroseqJSONEncoder


def _object_hook_factory(zf: zipfile.ZipFile):
    """
    Factory function that creates an object_hook that can reconstruct neuroseq and Brian2 objects.

    Any numpy arrays are loaded from the corresponding .npy file in the provided zipfile.
    """

    def neuroseq_object_hook(dct: dict, _zf=zf):
        match dct:
            case {"!Ndarray": True, "value": value}:
                with _zf.open(value) as fp:
                    return np.load(fp)
            case {"!Quantity": True, "value": value, "unit": unit}:
                if isinstance(value, str):
                    with _zf.open(value) as fp:
                        value = np.load(fp)
                return value_unit_str_to_quantity(value, unit)
            case {
                "!StimArray": True,
                "name": name,
                "stim_type": stim_type,
                "values": values,
                "units": units,
                "dt": dt,
            }:
                with _zf.open(values) as fp:
                    loaded_values = np.load(fp)
                return StimArray(name, stim_type, loaded_values, units, dt)
            case {
                "!SpikeData": True,
                "name": name,
                "spike_indices": spike_indices,
                "spike_times": spike_times,
            }:
                with _zf.open(spike_indices) as fp:
                    loaded_spike_indices = np.load(fp)
                with _zf.open(spike_times) as fp:
                    loaded_spike_times = np.load(fp)
                return SpikeData(name, loaded_spike_indices, loaded_spike_times)
            case {
                "!TraceData": True,
                "name": name,
                "traces": traces,
                "indices": indices,
                "units": units,
                "dt": dt,
            }:
                with _zf.open(traces) as fp:
                    loaded_traces = np.load(fp)
                with _zf.open(indices) as fp:
                    loaded_indices = np.load(fp)
                return TraceData(name, loaded_traces, loaded_indices, units, dt)
            case {
                "!SynapseData": True,
                "name": name,
                "values": values,
                "timestamps": timestamps,
            }:
                with _zf.open(values) as fp:
                    loaded_values = np.load(fp)
                with _zf.open(timestamps) as fp:
                    loaded_timestamps = np.load(fp)
                return SynapseData(name, loaded_values, loaded_timestamps)
            case {
                "!CellposeImportResult": True,
                "image_path": image_path,
                "segmentation_array": segmentation_array,
                "overlay_image": overlay_image,
                "trace_data": trace_data,
                "spike_data": spike_data,
                "original_cellpose_ids": original_cellpose_ids,
                "stimulus_to_cellpose_mapping": stimulus_to_cellpose_mapping,
            }:
                with _zf.open(segmentation_array) as fp:
                    loaded_segmentation = np.load(fp)
                with _zf.open(overlay_image) as fp:
                    loaded_overlay = np.load(fp)
                loaded_original_ids = None
                if original_cellpose_ids is not None:
                    with _zf.open(original_cellpose_ids) as fp:
                        loaded_original_ids = np.load(fp)

                return CellposeImportResult(
                    image_path=Path(image_path),
                    segmentation_array=loaded_segmentation,
                    overlay_image=loaded_overlay,
                    trace_data=trace_data,
                    spike_data=spike_data,
                    original_cellpose_ids=loaded_original_ids,
                    stimulus_to_cellpose_mapping=stimulus_to_cellpose_mapping,
                )
            case _:
                return dct

    return neuroseq_object_hook


def append_to_zip(dct: dict, name: str, zf: zipfile.ZipFile) -> None:
    """
    Serialize a dictionary to a JSON file and write this to a zipfile using the NeuroseqJSONEncoder.

    Args:
        dct: The dictionary to serialize.
        name: The name of the dictionary (used for naming the JSON file).
        zf: The zipfile to save the dictionary to.
    """
    json_str = json.dumps(
        dct,
        cls=_json_encoder_factory(zf),
        separators=(",", ":"),
    )
    zf.writestr(f"{name}.json", json_str)


def load_from_zip(name: str, zf: zipfile.ZipFile) -> dict:
    """
    Load and deserialize a dictionary from a JSON file in a zipfile using the neuroseq_object_hook.

    Args:
        name: The name of the dictionary (used for naming the JSON file).
        zf: The zipfile to load the dictionary from.
    Returns:
        The deserialized dictionary or an empty dict if the JSON file does not exist in the zipfile.
    """
    json_filename = f"{name}.json"
    if json_filename not in zf.namelist():
        return {}
    with zf.open(f"{name}.json") as fp:
        return json.load(fp, object_hook=_object_hook_factory(zf))


def as_timed_array(stim_array: StimArray) -> b2.TimedArray:
    """Convert a StimArray to a brian2 TimedArray for use in simulations.

    Returns:
        TimedArray: A brian2 TimedArray representing the stimuli.
    """
    return b2.TimedArray(
        values=value_unit_str_to_quantity(stim_array.values, stim_array.units),
        dt=stim_array.dt * b2.second,
        name=stim_array.name,
    )
