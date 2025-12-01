from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brian2.numpy_ as np
from PIL import Image
from PIL import ImageSequence

from neuroseq.experiment import SpikeData
from neuroseq.experiment import TraceData
from neuroseq.utils import jet_colors_for_neurons


@dataclass
class CellposeImportResult:
    image_path: Path
    segmentation_array: np.ndarray
    overlay_image: np.ndarray
    trace_data: TraceData | None
    spike_data: SpikeData | None


def prepare_cellpose_traces(
    traces: np.ndarray,
    *,
    name: str,
    dt: float,
    units: str,
) -> TraceData:
    if traces.ndim != 2:
        raise ValueError(
            "Cellpose traces must be a 2D array of shape (neurons, timepoints)"
        )
    if traces.shape[0] == 0 or traces.shape[1] == 0:
        raise ValueError(
            "Cellpose traces must contain at least one neuron and one timepoint"
        )
    if dt <= 0:
        raise ValueError("Cellpose trace dt must be positive")

    return TraceData(name, traces.astype(np.float32), units, dt)


def prepare_cellpose_spikes(
    spike_indices: np.ndarray,
    spike_times: np.ndarray,
    *,
    name: str,
) -> SpikeData:
    if spike_indices.ndim != 1 or spike_times.ndim != 1:
        raise ValueError("Spike indices and times must be 1D arrays")
    if spike_indices.shape[0] != spike_times.shape[0]:
        raise ValueError("Spike indices and times must have matching lengths")
    if spike_indices.shape[0] == 0:
        raise ValueError("Spike data must contain at least one event")
    if np.any(spike_indices < 0):
        raise ValueError("Spike indices must be non-negative")
    if np.any(spike_times < 0):
        raise ValueError("Spike times must be non-negative")

    return SpikeData(
        name,
        spike_indices.astype(np.int32),
        spike_times.astype(np.float32),
    )


class CellPoseImporter:
    """Handles loading and processing of Cellpose segmentation data."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.image_data: np.ndarray | None = None
        self.segmentation_data: np.ndarray | None = None
        self.overlay_image: np.ndarray | None = None
        self.trace_data: TraceData | None = None
        self.spike_data: SpikeData | None = None
        self._neuron_ids: np.ndarray | None = None
        self._neuron_colors: np.ndarray | None = None

    def load_image_file(
        self,
        filename: str | Path,
    ) -> None:
        """load PNG or TIFF image file. TIFF stacks use mean projection."""
        path = Path(filename)
        suffix = path.suffix.lower()
        try:
            with Image.open(path) as pil_image:
                if suffix in {".tif", ".tiff"}:
                    self.image_data = self._load_mean_tiff(pil_image)
                else:
                    self.image_data = self._pil_to_rgb_array(pil_image)

            self.logger.info("Loaded image data: %s", self.image_data.shape)
        except Exception as exc:
            self.logger.exception("Error loading image file")
            raise ValueError(f"Could not load {filename}: {exc}") from exc

    @staticmethod
    def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
        """Return RGB numpy array for a PIL image."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)

    def _load_mean_tiff(self, image: Image.Image) -> np.ndarray:
        """Average all frames of a TIFF stack per pixel, then normalize to display range."""
        # Load all frames into memory
        frames = []
        for frame in ImageSequence.Iterator(image):
            frame_array = self._frame_to_rgb_float(frame)
            frames.append(frame_array)

        if not frames:
            raise ValueError("TIF file contains no frames")

        # Ensure all frames have same shape
        first_shape = frames[0].shape
        for i, frame in enumerate(frames):
            if frame.shape != first_shape:
                raise ValueError(
                    f"Frame {i} has shape {frame.shape}, expected {first_shape}"
                )

        frame_stack = np.stack(frames, axis=0)
        result = np.mean(frame_stack, axis=0)

        return self._scale_float_image_to_uint8(result)

    def _frame_to_rgb_float(self, frame: Image.Image) -> np.ndarray:
        """Return frame data as float32 RGB array preserving raw intensities."""
        raw_array = np.asarray(frame)
        array = raw_array.astype(np.float32)

        if array.ndim == 2:
            array = array[:, :, np.newaxis]

        channels = array.shape[2]
        if channels == 1:
            array = np.repeat(array, 3, axis=2)
        elif channels == 2:
            array = np.concatenate([array, array[:, :, :1]], axis=2)
        elif channels > 3:
            array = array[:, :, :3]

        return array

    @staticmethod
    def _scale_float_image_to_uint8(image_array: np.ndarray) -> np.ndarray:
        """Scale arbitray float RGB array into uint8 range using min/max normalization."""
        finite_values = image_array[np.isfinite(image_array)]
        if finite_values.size == 0:
            raise ValueError("Image contains no finite pixels")

        min_val = float(np.min(finite_values))
        max_val = float(np.max(finite_values))

        if max_val > min_val:
            scaled = (image_array - min_val) / (max_val - min_val)
        else:
            scaled = np.zeros_like(image_array, dtype=np.float32)

        scaled = np.clip(scaled * 255.0, 0, 255)
        return scaled.astype(np.uint8)

    def load_segmentation_file(self, filename: str | Path) -> None:
        """Load segmentation file saved as CSV or NPY."""
        path = Path(filename)
        try:
            if path.suffix.lower() == ".npy":
                loaded = np.load(path, allow_pickle=True)

                # Handle cellpose output format (dict with 'masks' key)
                if isinstance(loaded, np.ndarray) and loaded.dtype == object:
                    if loaded.ndim == 0:
                        dict_data = loaded.item()
                        if isinstance(dict_data, dict) and "masks" in dict_data:
                            self.segmentation_data = np.asarray(
                                dict_data["masks"], dtype=np.int32
                            )
                            self.logger.info(
                                "Loaded cellpose segmentation with masks from dict format: %s",
                                self.segmentation_data.shape,
                            )
                        else:
                            raise ValueError("Dict format but no 'masks' key found")
                    else:
                        # Array of dicts (stat.npy format) - not supported for segmentation
                        raise ValueError(
                            "Array of dict format not supported for segmentation"
                        )
                else:
                    # Standard numpy array format
                    self.segmentation_data = np.asarray(loaded, dtype=np.int32)
            else:
                loaded = np.loadtxt(path, delimiter=",", dtype=np.int32)
                self.segmentation_data = np.asarray(loaded, dtype=np.int32)

            if self.segmentation_data.ndim != 2:
                raise ValueError(
                    f"Expected 2D array, got {self.segmentation_data.ndim}D array"
                )

            if np.any(self.segmentation_data < 0):
                raise ValueError("Segmentation data contains negative values")

            self.logger.info(
                "Loaded segmentation from %s: %s",
                path.suffix.lower(),
                self.segmentation_data.shape,
            )
        except Exception as exc:
            self.logger.exception("Error loading segmentation file")
            raise ValueError(f"Could not load {filename}: {exc}") from exc

    def set_trace_data(self, trace_data: TraceData) -> None:
        self._validate_trace_data(trace_data)
        self.trace_data = trace_data

    def set_spike_data(self, spike_data: SpikeData) -> None:
        self._validate_spike_data(spike_data)
        self.spike_data = spike_data

    def validate_dimensions(self) -> None:
        """check if image and segmentation dimensions match."""
        if self.image_data is None or self.segmentation_data is None:
            raise ValueError("Both image and segmentation data must be loaded first")

        if (
            self.image_data.shape[0] != self.segmentation_data.shape[0]
            or self.image_data.shape[1] != self.segmentation_data.shape[1]
        ):
            raise ValueError(
                f"Image dimensions {self.image_data.shape[:2]} do not match "
                f"segmentation dimensions {self.segmentation_data.shape}"
            )

    def generate_overlay(self) -> None:
        """Generate color overlay image using neuron colormap."""
        if self.image_data is None or self.segmentation_data is None:
            raise ValueError("Both image and segmentation data must be loaded first")

        unique_ids = np.unique(self.segmentation_data)
        neuron_ids = unique_ids[unique_ids > 0]

        if neuron_ids.size > 0:
            self._neuron_ids = neuron_ids
            self._neuron_colors = self._compute_neuron_colors(neuron_ids)

            overlay = np.zeros((*self.segmentation_data.shape, 3), dtype=np.uint8)

            for i, seg_id in enumerate(neuron_ids):
                mask = self.segmentation_data == seg_id
                overlay[mask] = self._neuron_colors[i]

            self.overlay_image = (0.5 * self.image_data + 0.5 * overlay).astype(
                np.uint8
            )
        else:
            self.overlay_image = self.image_data.copy()
            self._neuron_ids = None
            self._neuron_colors = None

    def get_info(self) -> dict[str, Any]:
        """Get information about loaded data."""
        if self.image_data is None or self.segmentation_data is None:
            return {}

        unique_ids = np.unique(self.segmentation_data)
        n_background = np.sum(self.segmentation_data == 0)
        n_neurons = len(unique_ids[unique_ids > 0])

        info = {
            "image_shape": self.image_data.shape,
            "segmentation_shape": self.segmentation_data.shape,
            "unique_ids": list(unique_ids),
            "background_pixels": n_background,
            "n_neurons": n_neurons,
        }

        if n_neurons > 0:
            info["neuron_id_range"] = (
                unique_ids[unique_ids > 0].min(),
                unique_ids[unique_ids > 0].max(),
            )
        if self.trace_data is not None:
            info["trace_neurons"] = self.trace_data.traces.shape[0]
            info["trace_timepoints"] = self.trace_data.traces.shape[1]
        if self.spike_data is not None:
            info["spike_events"] = self.spike_data.spike_indices.shape[0]

        return info

    def get_legend_data(self) -> list[tuple[int, tuple[int, int, int]]]:
        """Get legend data with cell ID and RGB color tuples."""
        if self.segmentation_data is None:
            return []

        if self._neuron_ids is None or self._neuron_colors is None:
            return []

        legend_data: list[tuple[int, tuple[int, int, int]]] = []
        for seg_id, color in zip(self._neuron_ids, self._neuron_colors, strict=True):
            rgb = (int(color[0]), int(color[1]), int(color[2]))
            legend_data.append((int(seg_id), rgb))

        return legend_data

    def create_result(self, image_path: Path) -> CellposeImportResult:
        """Create import result after successful processing."""
        if self.segmentation_data is None or self.overlay_image is None:
            raise ValueError("Data must be processed before creating result")

        return CellposeImportResult(
            image_path=image_path,
            segmentation_array=self.segmentation_data,
            overlay_image=self.overlay_image,
            trace_data=self.trace_data,
            spike_data=self.spike_data,
        )

    def _compute_neuron_colors(self, neuron_indices: np.ndarray | None) -> np.ndarray:
        if neuron_indices is None or neuron_indices.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        zero_based = neuron_indices - neuron_indices.min()
        return jet_colors_for_neurons(zero_based)

    @staticmethod
    def _validate_trace_data(trace_data: TraceData) -> None:
        if trace_data.traces.ndim != 2:
            raise ValueError("TraceData.traces must be 2D")
        if trace_data.traces.shape[0] == 0 or trace_data.traces.shape[1] == 0:
            raise ValueError(
                "TraceData.traces must contain at least one neuron and one timepoint"
            )
        if trace_data.dt <= 0:
            raise ValueError("TraceData.dt must be positive")

    @staticmethod
    def _validate_spike_data(spike_data: SpikeData) -> None:
        if spike_data.spike_indices.ndim != 1 or spike_data.spike_times.ndim != 1:
            raise ValueError("SpikeData arrays must be 1D")
        if spike_data.spike_indices.shape[0] != spike_data.spike_times.shape[0]:
            raise ValueError("SpikeData indices and times must have matching lengths")
        if spike_data.spike_indices.shape[0] == 0:
            raise ValueError("SpikeData must contain at least one spike")
        if np.any(spike_data.spike_indices < 0):
            raise ValueError("Spike indices must be non-negative")
        if np.any(spike_data.spike_times < 0):
            raise ValueError("Spike times must be non-negative")
