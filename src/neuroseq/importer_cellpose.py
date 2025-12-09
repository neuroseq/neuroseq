from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brian2.numpy_ as np
from PIL import Image
from PIL import ImageSequence

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.utils import neuron_indices_to_colors


@dataclass
class CellposeImportResult:
    image_path: Path
    segmentation_array: np.ndarray
    overlay_image: np.ndarray
    trace_data: TraceData | None
    spike_data: SpikeData | None
    original_cellpose_ids: np.ndarray | None = None
    stimulus_to_cellpose_mapping: dict[int, int] | None = None


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

    indices = np.arange(traces.shape[0], dtype=np.uint32)
    return TraceData(name, traces.astype(np.float32), indices, units, dt)


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
        self._original_cellpose_ids: np.ndarray | None = None

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
            segmentation = self._read_segmentation_file(path)
            self.segmentation_data = self._validate_segmentation(segmentation)

            unique_ids = np.unique(self.segmentation_data)
            self._original_cellpose_ids = unique_ids[unique_ids > 0]

            self.logger.info(
                "Loaded segmentation from %s: %s (background kept at 0, cells preserved)",
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

    def generate_overlay(self, opacity: float = 0.5) -> None:
        """Generate color overlay image using neuron colormap.

        Opacity parameter exists to make tests more certain/easier. There were issues with different color transforms/
        implementations, so doing it this way lets us test colors exactly for an exact match between the legend and the
        overlay itself.

        Args:
            opacity: Opacity for cell overlays (0.0 = transparent, 1.0 = full opacity)
        """
        if self.image_data is None or self.segmentation_data is None:
            raise ValueError("Both image and segmentation data must be loaded first")

        unique_ids = np.unique(self.segmentation_data)
        neuron_ids = unique_ids[unique_ids > 0]

        if neuron_ids.size > 0:
            # neuron_ids are already 0-based after loading
            self._neuron_ids = neuron_ids
            self._neuron_colors = self._compute_neuron_colors(neuron_ids)

            self.overlay_image = self.image_data.copy().astype(np.float32)

            for i, seg_id in enumerate(neuron_ids):
                mask = self.segmentation_data == seg_id
                cell_color = self._neuron_colors[i].astype(np.float32)
                self.overlay_image[mask] = (1 - opacity) * self.image_data[
                    mask
                ] + opacity * cell_color

            self.overlay_image = self.overlay_image.astype(np.uint8)
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

    def create_result(
        self,
        image_path: Path,
        stimulus_to_cellpose_mapping: dict[int, int] | None = None,
    ) -> CellposeImportResult:
        """Create import result after successful processing."""
        if self.segmentation_data is None or self.overlay_image is None:
            raise ValueError("Data must be processed before creating result")

        return CellposeImportResult(
            image_path=image_path,
            segmentation_array=self.segmentation_data,
            overlay_image=self.overlay_image,
            trace_data=self.trace_data,
            spike_data=self.spike_data,
            original_cellpose_ids=self._original_cellpose_ids,
            stimulus_to_cellpose_mapping=stimulus_to_cellpose_mapping,
        )

    def _compute_neuron_colors(self, neuron_indices: np.ndarray | None) -> np.ndarray:
        if neuron_indices is None or neuron_indices.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        # neuron_indices are already 0-based, so use them directly
        colors_rgba_float = neuron_indices_to_colors(neuron_indices)
        return (colors_rgba_float[:, :3] * 255).astype(
            np.uint8
        )  # convert from decimal to RGB int8 integers

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
            raise ValueError(
                "SpikeData indices and times must have matching lengths - one set of times per"
            )
        if spike_data.spike_indices.shape[0] == 0:
            raise ValueError("SpikeData must contain at least one spike data point")
        if np.any(spike_data.spike_indices < 0):
            raise ValueError("Spike indices must be non-negative")
        if np.any(spike_data.spike_times < 0):
            raise ValueError("Spike times must be non-negative")

    def _load_suite2p_segmentation(
        self, stat_data: np.ndarray, stat_path: Path
    ) -> np.ndarray:
        # experimental support - not rigourously tested or tested in our tests.
        """Load suite2p stat.npy format and convert to segmentation mask."""
        # Check if this looks like suite2p stat.npy format
        if not (stat_data.dtype == object and len(stat_data) > 0):
            raise ValueError(
                "Array of dict format not supported for segmentation (not suite2p)"
            )

        first_roi = stat_data[0]
        if not isinstance(first_roi, dict) or "ypix" not in first_roi:
            raise ValueError(
                "Array of dict format not supported for segmentation (not suite2p)"
            )

        # Look for corresponding iscell.npy file
        iscell_path = stat_path.parent / "iscell.npy"
        if not iscell_path.exists():
            raise ValueError(
                "Suite2p stat.npy detected but iscell.npy not found in same directory"
            )

        iscell_data = np.load(iscell_path)
        if iscell_data.shape[0] != len(stat_data):
            raise ValueError("Mismatch between stat.npy and iscell.npy lengths")

        # Determine image dimensions from ROI coordinates
        all_ypix = []
        all_xpix = []
        for roi in stat_data:
            all_ypix.extend(roi["ypix"])
            all_xpix.extend(roi["xpix"])

        image_height = max(all_ypix) + 1
        image_width = max(all_xpix) + 1

        # Create segmentation mask
        seg_mask = np.zeros((image_height, image_width), dtype=np.int32)
        cell_id = 1  # Start with 1-based IDs; background stays 0

        for i, roi in enumerate(stat_data):
            # Only include cells that are marked as real (iscell == 1)
            if iscell_data[i, 0] == 1:
                ypix = roi["ypix"]
                xpix = roi["xpix"]
                seg_mask[ypix, xpix] = cell_id
                cell_id += 1

        n_cells = cell_id
        n_total_rois = len(stat_data)
        n_real_cells = np.sum(iscell_data[:, 0] == 1)

        self.logger.info(
            "Loaded suite2p segmentation: %s, %d cells from %d ROIs (%d real)",
            seg_mask.shape,
            n_cells,
            n_total_rois,
            n_real_cells,
        )

        return seg_mask

    def _read_segmentation_file(self, path: Path) -> np.ndarray:
        """Load segmentation data from CSV or NPY (cellpose or suite2p)."""
        if path.suffix.lower() != ".npy":
            return np.loadtxt(path, delimiter=",", dtype=np.int32)

        loaded = np.load(path, allow_pickle=True)

        # cellpose dict format
        if isinstance(loaded, np.ndarray) and loaded.dtype == object:
            if loaded.ndim == 0:
                dict_data = loaded.item()
                if isinstance(dict_data, dict) and "masks" in dict_data:
                    return np.asarray(dict_data["masks"], dtype=np.int32)
                raise ValueError("Dict format but no 'masks' key found")

            # suite2p stat.npy format
            return self._load_suite2p_segmentation(loaded, path)

        # plain numeric array
        return np.asarray(loaded, dtype=np.int32)

    @staticmethod
    def _validate_segmentation(segmentation: np.ndarray) -> np.ndarray:
        seg = np.asarray(segmentation, dtype=np.int32)

        if seg.ndim != 2:
            raise ValueError(f"Expected 2D array, got {seg.ndim}D array")

        if np.any(seg < 0):
            raise ValueError("Segmentation data contains negative values")

        if not np.isfinite(seg).all():
            raise ValueError("Segmentation data contains non-finite values")

        return seg
