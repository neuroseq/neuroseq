from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Literal
from typing import TypedDict

import brian2.numpy_ as np
import cmap as cmap_lib
import numpy.typing as npt
from PIL import Image
from PIL import ImageSequence

from neuroseq.utils import cmap_transform_neuron_indices

__all__ = [
    "Suite2pTracePayload",
    "Suite2pSpikePayload",
    "prepare_suite2p_traces",
    "prepare_suite2p_spikes",
    "compute_normalized_suite2p_traces",
    "compute_normalized_suite2p_spikes",
    "CellposeImportResult",
    "CellPoseImporter",
]


class Suite2pTracePayload(TypedDict):
    widget: Literal["trace"]
    timestamps: npt.NDArray[Any]
    traces: npt.NDArray[Any]
    dt: float
    sampling_rate: float


class Suite2pSpikePayload(TypedDict):
    widget: Literal["spike"]
    timestamps: npt.NDArray[Any]
    spikes: npt.NDArray[Any]
    dt: float
    threshold: float


class CellposeImportResult(TypedDict, total=False):
    image_path: Path
    segmentation_array: np.ndarray
    overlay_image: np.ndarray


def _normalize_array_per_neuron(
    data: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Return per-neuron min/max normalized copy of the array along with the raw bounds."""
    if data.size == 0:
        raise ValueError("Cannot normalize empty array")

    if data.ndim != 2:
        raise ValueError("Data must be 2D array (neurons x timepoints)")

    n_neurons, n_timepoints = data.shape
    normalized = np.zeros_like(data)
    raw_mins = np.zeros(n_neurons, dtype=float)
    raw_maxs = np.zeros(n_neurons, dtype=float)

    for i in range(n_neurons):
        neuron_data = data[i, :]
        raw_min = float(np.min(neuron_data))
        raw_max = float(np.max(neuron_data))

        raw_mins[i] = raw_min
        raw_maxs[i] = raw_max

        if raw_max > raw_min:
            normalized[i, :] = (neuron_data - raw_min) / (raw_max - raw_min)
        else:
            normalized[i, :] = np.zeros_like(neuron_data)

    return normalized, raw_mins, raw_maxs


def compute_normalized_suite2p_traces(
    f_data: npt.NDArray[Any],
    fneu_data: npt.NDArray[Any] | None = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Contrast Suite2p fluorescence data (optional neuropil subtraction) and normalize per-neuron.
    Returns normalized traces along with the per-neuron pre-normalization min/max values.
    """
    if fneu_data is not None and f_data.shape != fneu_data.shape:
        raise ValueError(
            f"F.npy shape {f_data.shape} does not match Fneu.npy shape {fneu_data.shape}"
        )

    contrasting_traces = f_data - 0.7 * fneu_data if fneu_data is not None else f_data
    return _normalize_array_per_neuron(contrasting_traces)


def compute_normalized_suite2p_spikes(
    spike_data: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Normalize Suite2p spiking activity to [0, 1] per-neuron."""
    return _normalize_array_per_neuron(spike_data)


def prepare_suite2p_traces(
    f_data: npt.NDArray[Any],
    fneu_data: npt.NDArray[Any] | None,
    *,
    max_neurons: int,
    sampling_rate: float = 30.0,
) -> Suite2pTracePayload:
    """Create Neuroseq trace widget payload from Suite2p fluorescence data."""

    if max_neurons <= 0:
        raise ValueError("max_neurons must be positive")

    dt = 1.0 / float(sampling_rate)
    n_neurons, n_timepoints = f_data.shape
    n_neurons_to_import = min(max_neurons, n_neurons)
    normalized_traces, _, _ = compute_normalized_suite2p_traces(f_data, fneu_data)

    return {
        "widget": "trace",
        "timestamps": np.arange(n_timepoints) * dt,
        "traces": normalized_traces[:n_neurons_to_import, :],
        "dt": dt,
        "sampling_rate": float(sampling_rate),
    }


def prepare_suite2p_spikes(
    spike_data: npt.NDArray[Any],
    *,
    dt: float,
    threshold: float,
    max_neurons: int,
) -> Suite2pSpikePayload:
    """
    Create Neuroseq spike widget payload from Suite2p spks.npy data.
    """
    if max_neurons <= 0:
        raise ValueError("max_neurons must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    n_neurons, n_timesteps = spike_data.shape
    n_neurons_to_import = min(max_neurons, n_neurons)
    normalized_spikes, _, _ = compute_normalized_suite2p_spikes(spike_data)
    trimmed_data = normalized_spikes[:n_neurons_to_import, :]

    mask = trimmed_data >= threshold
    neuron_indices, time_indices = np.nonzero(mask)
    spike_times = time_indices * dt

    return {
        "widget": "spike",
        "timestamps": spike_times,
        "spikes": neuron_indices,
        "dt": dt,
        "threshold": threshold,
    }


class CellPoseImporter:
    """Handles loading and processing of Cellpose segmentation data."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.image_data: np.ndarray | None = None
        self.segmentation_data: np.ndarray | None = None
        self.overlay_image: np.ndarray | None = None
        self._neuron_ids: np.ndarray | None = None
        self._neuron_colors: np.ndarray | None = None

    def load_image_file(self, filename: str | Path) -> None:
        """Load PNG or TIFF image file. TIFF stacks are averaged across frames."""
        path = Path(filename)
        suffix = path.suffix.lower()
        try:
            with Image.open(path) as pil_image:
                if suffix in {".tif", ".tiff"}:
                    self.image_data = self._load_mean_tiff(pil_image)
                else:
                    self.image_data = self._pil_to_rgb_array(pil_image)

            self.logger.info(f"Loaded image: {self.image_data.shape}")
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
        accumulator: np.ndarray | None = None
        n_frames = 0

        for frame in ImageSequence.Iterator(image):
            frame_array = self._frame_to_rgb_float(frame)

            if accumulator is None:
                accumulator = np.zeros_like(frame_array, dtype=np.float64)
            elif frame_array.shape != accumulator.shape:
                raise ValueError("All TIF frames must share identical dimensions")

            accumulator += frame_array
            n_frames += 1

        if accumulator is None or n_frames == 0:
            raise ValueError("TIF file contains no frames")

        mean_frame = accumulator / float(n_frames)
        return self._scale_float_image_to_uint8(mean_frame)

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
        """Scale arbitrary float RGB array into uint8 range using min/max normalization."""
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
        """Load CSV segmentation file."""
        try:
            self.segmentation_data = np.loadtxt(filename, delimiter=",", dtype=np.int32)

            if self.segmentation_data.ndim != 2:
                raise ValueError(
                    f"Expected 2D array, got {self.segmentation_data.ndim}D array"
                )

            if np.any(self.segmentation_data < 0):
                raise ValueError("Segmentation data contains negative values")

            self.logger.info(f"Loaded segmentation: {self.segmentation_data.shape}")
        except Exception as exc:
            self.logger.exception("Error loading segmentation file")
            raise ValueError(f"Could not load {filename}: {exc}") from exc

    def validate_dimensions(self) -> None:
        """Check if image and segmentation dimensions match."""
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
            self._neuron_colors = self._compute_neuron_colors(neuron_ids.size)

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

        return {
            "image_path": image_path,
            "segmentation_array": self.segmentation_data,
            "overlay_image": self.overlay_image,
        }

    def _compute_neuron_colors(self, n_neurons: int) -> np.ndarray:
        if n_neurons <= 0:
            return np.zeros((0, 3), dtype=np.uint8)

        # Use exactly the same color mapping as trace/spike widgets
        # The graphics library normalizes cmap_transform to [0,1] then maps to colormap
        transform = cmap_transform_neuron_indices(n_neurons).astype(np.float32)
        normalized = self._normalize_transform(transform)

        # Map normalized values to colormap indices
        lut = self._get_jet_lut()
        indices = (normalized * (lut.shape[0] - 1)).astype(int)
        indices = np.clip(indices, 0, lut.shape[0] - 1)
        colors = lut[indices, :3]
        return (colors * 255.0).astype(np.uint8)

    @staticmethod
    def _normalize_transform(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values

        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if max_val > min_val:
            return (values - min_val) / (max_val - min_val)
        return np.zeros_like(values)

    @staticmethod
    def _get_jet_lut() -> np.ndarray:
        if not hasattr(CellPoseImporter, "_JET_LUT"):
            CellPoseImporter._JET_LUT = cmap_lib.Colormap("jet").lut(256)
        return CellPoseImporter._JET_LUT
