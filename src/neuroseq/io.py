import json
import zipfile
from io import BytesIO
from pathlib import Path

import brian2 as b2
import numpy as np

from neuroseq.b2_units import quantity_to_value_unit_str
from neuroseq.b2_units import value_unit_str_to_quantity
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.importer_cellpose import AOIRole
from neuroseq.importer_cellpose import AreasOfInterestConfig
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import CircleAOI
from neuroseq.importer_cellpose import SquareAOI
from neuroseq.stim import StimArray
from neuroseq.stim_timing import ensure_2d_stim_values


def _to_seconds(value: float | int) -> float:
    return float(value)


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


def _aoi_to_json(config: AreasOfInterestConfig | None) -> dict | None:
    if config is None:
        return None
    return {
        "square": {
            "enabled": bool(config.square.enabled),
            "role": config.square.role.value,
            "x": float(config.square.x),
            "y": float(config.square.y),
            "side": float(config.square.side),
        },
        "circle": {
            "enabled": bool(config.circle.enabled),
            "role": config.circle.role.value,
            "cx": float(config.circle.cx),
            "cy": float(config.circle.cy),
            "diameter": float(config.circle.diameter),
        },
        "show_shapes": bool(config.show_shapes),
        "coordinate_mode": str(config.coordinate_mode),
    }


def _aoi_from_json(config: dict | None) -> AreasOfInterestConfig | None:
    if config is None:
        return None
    square = config["square"]
    circle = config["circle"]
    return AreasOfInterestConfig(
        square=SquareAOI(
            enabled=bool(square["enabled"]),
            role=AOIRole(square["role"]),
            x=float(square["x"]),
            y=float(square["y"]),
            side=float(square["side"]),
        ),
        circle=CircleAOI(
            enabled=bool(circle["enabled"]),
            role=AOIRole(circle["role"]),
            cx=float(circle["cx"]),
            cy=float(circle["cy"]),
            diameter=float(circle["diameter"]),
        ),
        show_shapes=bool(config.get("show_shapes", True)),
        coordinate_mode=str(config.get("coordinate_mode", "pixels")),
    )


def _json_encoder_factory(zf: zipfile.ZipFile):
    """
    Factory function that creates a JSON encoder class that can handle neuroseq and Brian2 objects.

    The encoder saves any numpy it encounters as .npy files in the provided zipfile.
    """

    class NeuroseqJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            match obj:
                case Experiment():
                    return {
                        "!Experiment": True,
                        "name": obj.name,
                        "duration": obj.duration,
                        "timepoints": obj.timepoints,
                        "data": obj.data,
                    }
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
                    raw_cellpose_ids_path = (
                        None
                        if obj.raw_cellpose_ids is None
                        else _save_array(
                            zf, f"_{obj.name}_raw_cellpose_ids", obj.raw_cellpose_ids
                        )
                    )
                    return {
                        "!TraceData": True,
                        "name": obj.name,
                        "traces": _save_array(zf, f"_{obj.name}_traces", obj.traces),
                        "indices": _save_array(zf, f"_{obj.name}_indices", obj.indices),
                        "units": obj.units,
                        "dt": obj.dt,
                        "raw_cellpose_ids": raw_cellpose_ids_path,
                    }
                case SpikeData():
                    raw_cellpose_ids_path = (
                        None
                        if obj.raw_cellpose_ids is None
                        else _save_array(
                            zf,
                            f"_{obj.name}_spike_raw_cellpose_ids",
                            obj.raw_cellpose_ids,
                        )
                    )
                    return {
                        "!SpikeData": True,
                        "name": obj.name,
                        "spike_indices": _save_array(
                            zf, f"_{obj.name}_spike_indices", obj.spike_indices
                        ),
                        "spike_times": _save_array(
                            zf, f"_{obj.name}_spike_times", obj.spike_times
                        ),
                        "raw_cellpose_ids": raw_cellpose_ids_path,
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
                        "mapping_strategy": obj.mapping_strategy,
                        "linear_offset": obj.linear_offset,
                        "aoi_config": _aoi_to_json(obj.aoi_config),
                        "segmentation_path": (
                            None
                            if obj.segmentation_path is None
                            else str(obj.segmentation_path)
                        ),
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
                    loaded_values = ensure_2d_stim_values(np.load(fp))
                return StimArray(
                    name,
                    stim_type,
                    loaded_values,
                    units,
                    _to_seconds(dt),
                )
            case {
                "!SpikeData": True,
                "name": name,
                "spike_indices": spike_indices,
                "spike_times": spike_times,
                **rest,
            }:
                with _zf.open(spike_indices) as fp:
                    loaded_spike_indices = np.load(fp)
                with _zf.open(spike_times) as fp:
                    loaded_spike_times = np.load(fp)
                raw_cellpose_ids_path = rest.get("raw_cellpose_ids")
                loaded_raw_cellpose_ids = None
                if raw_cellpose_ids_path is not None:
                    with _zf.open(raw_cellpose_ids_path) as fp:
                        loaded_raw_cellpose_ids = np.load(fp)
                return SpikeData(
                    name,
                    loaded_spike_indices,
                    loaded_spike_times,
                    loaded_raw_cellpose_ids,
                )
            case {
                "!TraceData": True,
                "name": name,
                "traces": traces,
                "indices": indices,
                "units": units,
                "dt": dt,
                **rest,
            }:
                with _zf.open(traces) as fp:
                    loaded_traces = np.load(fp)
                with _zf.open(indices) as fp:
                    loaded_indices = np.load(fp)
                raw_cellpose_ids_path = rest.get("raw_cellpose_ids")
                loaded_raw_cellpose_ids = None
                if raw_cellpose_ids_path is not None:
                    with _zf.open(raw_cellpose_ids_path) as fp:
                        loaded_raw_cellpose_ids = np.load(fp)
                return TraceData(
                    name,
                    loaded_traces,
                    loaded_indices,
                    units,
                    _to_seconds(dt),
                    loaded_raw_cellpose_ids,
                )
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
                "mapping_strategy": mapping_strategy,
                "linear_offset": linear_offset,
                "aoi_config": aoi_config,
                **rest,
            }:
                with _zf.open(segmentation_array) as fp:
                    loaded_segmentation = np.load(fp)
                with _zf.open(overlay_image) as fp:
                    loaded_overlay = np.load(fp)
                loaded_original_ids = None
                if original_cellpose_ids is not None:
                    with _zf.open(original_cellpose_ids) as fp:
                        loaded_original_ids = np.load(fp)
                segmentation_path = rest.get("segmentation_path")

                return CellposeImportResult(
                    image_path=Path(image_path),
                    segmentation_array=loaded_segmentation,
                    overlay_image=loaded_overlay,
                    trace_data=trace_data,
                    spike_data=spike_data,
                    original_cellpose_ids=loaded_original_ids,
                    stimulus_to_cellpose_mapping=(
                        None
                        if stimulus_to_cellpose_mapping is None
                        else {
                            int(k): int(v)
                            for k, v in stimulus_to_cellpose_mapping.items()
                        }
                    ),
                    mapping_strategy=mapping_strategy,
                    linear_offset=(
                        None if linear_offset is None else int(linear_offset)
                    ),
                    aoi_config=_aoi_from_json(aoi_config),
                    segmentation_path=(
                        None if segmentation_path is None else Path(segmentation_path)
                    ),
                )
            case {
                "!CellposeImportResult": True,
                "image_path": image_path,
                "segmentation_array": segmentation_array,
                "overlay_image": overlay_image,
                "trace_data": trace_data,
                "spike_data": spike_data,
                "original_cellpose_ids": original_cellpose_ids,
                "stimulus_to_cellpose_mapping": stimulus_to_cellpose_mapping,
                "mapping_strategy": mapping_strategy,
                "linear_offset": linear_offset,
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
                    stimulus_to_cellpose_mapping=(
                        None
                        if stimulus_to_cellpose_mapping is None
                        else {
                            int(k): int(v)
                            for k, v in stimulus_to_cellpose_mapping.items()
                        }
                    ),
                    mapping_strategy=mapping_strategy,
                    linear_offset=(
                        None if linear_offset is None else int(linear_offset)
                    ),
                )
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
                    stimulus_to_cellpose_mapping=(
                        None
                        if stimulus_to_cellpose_mapping is None
                        else {
                            int(k): int(v)
                            for k, v in stimulus_to_cellpose_mapping.items()
                        }
                    ),
                )
            case {
                "!Experiment": True,
                "name": name,
                "duration": duration,
                "timepoints": timepoints,
                "data": data,
            }:
                return Experiment(
                    name,
                    _to_seconds(duration),
                    [_to_seconds(t) for t in timepoints],
                    data,
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
