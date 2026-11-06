from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import brian2.numpy_ as np
from PIL import Image
from PIL import ImageSequence

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.image_scaling import scale_to_uint8
from neuroseq.utils import neuron_indices_to_colors


class AOIRole(str, Enum):  # noqa: UP042
    PREFERENTIAL = "preferential"
    EXCLUSION = "exclusion"


@dataclass
class SquareAOI:
    enabled: bool
    role: AOIRole
    x: float
    y: float
    side: float


@dataclass
class CircleAOI:
    enabled: bool
    role: AOIRole
    cx: float
    cy: float
    diameter: float


@dataclass
class AreasOfInterestConfig:
    square: SquareAOI
    circle: CircleAOI
    show_shapes: bool = True
    coordinate_mode: str = "pixels"


@dataclass
class CellposeImportResult:
    image_path: Path
    segmentation_array: np.ndarray
    overlay_image: np.ndarray
    trace_data: TraceData | None
    spike_data: SpikeData | None
    original_cellpose_ids: np.ndarray | None = None
    stimulus_to_cellpose_mapping: dict[int, int] | None = None
    mapping_strategy: str | None = None
    linear_offset: int | None = None
    aoi_config: AreasOfInterestConfig | None = None
    segmentation_path: Path | None = None
    # The raw, uncolored image `overlay_image` was generated from - kept so
    # views can fall back to it (e.g. to make a "hide segmentation colors"
    # toggle actually change anything) without depending on image_path
    # still pointing at a reachable file.
    background_image: np.ndarray | None = None
    # (minimum, maximum) brightness/contrast display window over the 0-255
    # FOV intensity range, set from the Spatial Footprints panel. Display
    # only - None (or the full range) means the FOV is shown unadjusted.
    display_range: tuple[int, int] | None = None


def load_background_image(image_path: Path) -> np.ndarray | None:
    """Best-effort reload of the raw (uncolored) image at `image_path`.

    Used to backfill `CellposeImportResult.background_image` for projects
    saved before that field existed, so a "Show segmentation" toggle has a
    genuine raw background to fall back to instead of always showing the
    colored overlay - the same fallback `CellPoseImporter.restore_from_result`
    already relies on. Returns `None` if the file no longer exists or fails
    to load, rather than raising.
    """
    if not image_path.exists():
        return None
    importer = CellPoseImporter()
    try:
        importer.load_image_file(image_path)
    except ValueError:
        return None
    return importer.image_data


def clone_aoi_config(
    config: AreasOfInterestConfig | None,
) -> AreasOfInterestConfig | None:
    """Return a defensive copy of an AOI configuration."""
    if config is None:
        return None

    return AreasOfInterestConfig(
        square=SquareAOI(
            enabled=bool(config.square.enabled),
            role=AOIRole(config.square.role),
            x=float(config.square.x),
            y=float(config.square.y),
            side=float(config.square.side),
        ),
        circle=CircleAOI(
            enabled=bool(config.circle.enabled),
            role=AOIRole(config.circle.role),
            cx=float(config.circle.cx),
            cy=float(config.circle.cy),
            diameter=float(config.circle.diameter),
        ),
        show_shapes=bool(config.show_shapes),
        coordinate_mode=str(config.coordinate_mode),
    )


def default_aoi_config(image_shape: tuple[int, int]) -> AreasOfInterestConfig:
    """Return the default AOI config centred within the field of view."""
    h, w = image_shape[:2]
    short = min(h, w)
    circle_d = 0.095 * short
    square_s = 0.5 * short
    return AreasOfInterestConfig(
        square=SquareAOI(
            enabled=True,
            role=AOIRole.PREFERENTIAL,
            x=(w - square_s) / 2.0,
            y=(h - square_s) / 2.0,
            side=square_s,
        ),
        circle=CircleAOI(
            enabled=True,
            role=AOIRole.EXCLUSION,
            cx=w / 2.0,
            cy=h / 2.0,
            diameter=circle_d,
        ),
        show_shapes=True,
        coordinate_mode="percent",
    )


def point_in_square(x: float, y: float, sq: SquareAOI) -> bool:
    """Return whether an image-coordinate point lies inside the square."""
    return sq.x <= x <= sq.x + sq.side and sq.y <= y <= sq.y + sq.side


def point_in_circle(x: float, y: float, c: CircleAOI) -> bool:
    """Return whether an image-coordinate point lies inside the circle."""
    radius = c.diameter / 2.0
    return (x - c.cx) ** 2 + (y - c.cy) ** 2 <= radius * radius


def generate_aoi_stimulus_mapping(
    original_ids: np.ndarray | None,
    label_centers: dict[int, tuple[float, float]],
    config: AreasOfInterestConfig,
) -> dict[int, int]:
    """Map cellpose IDs to stimulus IDs using AOI role tiers.

    `label_centers` are (column, row) image coordinates, like the AOIs.
    """
    if original_ids is None:
        return {}

    cells: list[tuple[int, float, float]] = []
    for cellpose_id in original_ids:
        center = label_centers.get(int(cellpose_id))
        if center is None:
            continue
        cells.append((int(cellpose_id), float(center[0]), float(center[1])))

    preferential: list[int] = []
    neutral: list[int] = []
    exclusion: list[int] = []

    for cellpose_id, x, y in cells:
        in_square = config.square.enabled and point_in_square(x, y, config.square)
        in_circle = config.circle.enabled and point_in_circle(x, y, config.circle)

        is_exclusion = (
            config.square.enabled
            and config.square.role == AOIRole.EXCLUSION
            and in_square
        ) or (
            config.circle.enabled
            and config.circle.role == AOIRole.EXCLUSION
            and in_circle
        )
        is_preferential = not is_exclusion and (
            (
                config.square.enabled
                and config.square.role == AOIRole.PREFERENTIAL
                and in_square
            )
            or (
                config.circle.enabled
                and config.circle.role == AOIRole.PREFERENTIAL
                and in_circle
            )
        )

        if is_exclusion:
            exclusion.append(cellpose_id)
        elif is_preferential:
            preferential.append(cellpose_id)
        else:
            neutral.append(cellpose_id)

    preferential.sort()
    neutral.sort()
    exclusion.sort()
    ordered_ids = preferential + neutral + exclusion
    return {stim_id: cellpose_id for stim_id, cellpose_id in enumerate(ordered_ids)}


def clone_cellpose_result(
    result: CellposeImportResult | None,
) -> CellposeImportResult | None:
    """Return a defensive copy of a CellposeImportResult."""
    if result is None:
        return None

    return CellposeImportResult(
        image_path=result.image_path,
        segmentation_array=(
            None
            if result.segmentation_array is None
            else np.array(result.segmentation_array, copy=True)
        ),
        overlay_image=(
            None
            if result.overlay_image is None
            else np.array(result.overlay_image, copy=True)
        ),
        trace_data=result.trace_data,
        spike_data=result.spike_data,
        original_cellpose_ids=(
            None
            if result.original_cellpose_ids is None
            else np.array(result.original_cellpose_ids, copy=True)
        ),
        stimulus_to_cellpose_mapping=(
            None
            if result.stimulus_to_cellpose_mapping is None
            else dict(result.stimulus_to_cellpose_mapping)
        ),
        mapping_strategy=result.mapping_strategy,
        linear_offset=result.linear_offset,
        aoi_config=clone_aoi_config(result.aoi_config),
        segmentation_path=getattr(result, "segmentation_path", None),
        background_image=(
            None
            if result.background_image is None
            else np.array(result.background_image, copy=True)
        ),
        display_range=getattr(result, "display_range", None),
    )


def is_identity_mapping(mapping: dict[int, int] | None) -> bool:
    """Return True if all stimulus IDs map to the same cellpose IDs (identity)."""
    if not mapping:
        return False
    return all(stim_id == cellpose_id for stim_id, cellpose_id in mapping.items())


def generate_linear_stimulus_mapping(
    original_ids: np.ndarray | None, offset: int
) -> dict[int, int]:
    """Map stimulus IDs to cellpose IDs starting at the selected offset."""
    if original_ids is None:
        return {}
    ordered_ids = [int(cellpose_id) for cellpose_id in original_ids]
    first_ids = [cellpose_id for cellpose_id in ordered_ids if cellpose_id >= offset]
    wrapped_ids = [cellpose_id for cellpose_id in ordered_ids if cellpose_id < offset]
    mapped_ids = first_ids + wrapped_ids
    return {stim_id: cellpose_id for stim_id, cellpose_id in enumerate(mapped_ids)}


def resolve_stimulus_to_cellpose_mapping(
    cellpose_data: CellposeImportResult | None,
) -> dict[int, int] | None:
    """Resolve the mapping actually in effect for `cellpose_data`.

    The "Linear" mapping strategy never materializes a
    `stimulus_to_cellpose_mapping` dict - only its `linear_offset` is stored
    (see `CellposeImportDialog._import_data`) - so reading
    `stimulus_to_cellpose_mapping` directly for a Linear import looks
    indistinguishable from "no mapping at all" and gets silently treated as
    pure identity, dropping any nonzero offset. Reconstruct it here instead.
    """
    if cellpose_data is None:
        return None
    mapping = cellpose_data.stimulus_to_cellpose_mapping
    if mapping:
        return mapping
    strategy = cellpose_data.mapping_strategy or "Linear"
    if strategy != "Linear":
        return mapping
    return generate_linear_stimulus_mapping(
        cellpose_data.original_cellpose_ids,
        int(cellpose_data.linear_offset or 0),
    )


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
        self._csv_background_zero = False

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
                    self.image_data = self._load_tiff_with_strategy(pil_image)
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

    def _load_tiff_with_strategy(self, image: Image.Image) -> np.ndarray:
        """Project a TIFF stack with mean projection, then normalize to display range."""
        frames = []
        for frame in ImageSequence.Iterator(image):
            frames.append(self._frame_to_rgb_float(frame))

        if not frames:
            raise ValueError("TIF file contains no frames")

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
        """Scale a float RGB array onto uint8, ignoring the most extreme pixels so
        a few hot pixels do not darken the whole image (see image_scaling)."""
        return scale_to_uint8(image_array)

    def load_segmentation_file(self, filename: str | Path) -> None:
        """Load segmentation file saved as CSV or NPY."""
        path = Path(filename)
        try:
            if path.suffix.lower() == ".csv":
                segmentation = self._read_csv_segmentation(path)
                segmentation = self._validate_csv_segmentation(segmentation)
                self.segmentation_data = self._shift_to_zero_based(segmentation)
                self._csv_background_zero = False

                unique_ids = np.unique(self.segmentation_data)
                self._original_cellpose_ids = unique_ids[unique_ids >= 0]

                self.logger.info(
                    "Loaded CSV segmentation: %s (shifted to 0-based IDs, background is -1)",
                    self.segmentation_data.shape,
                )
            else:
                segmentation = self._read_segmentation_file(path)
                segmentation = self._validate_segmentation(segmentation)
                self.segmentation_data = self._shift_to_zero_based(segmentation)
                self._csv_background_zero = False

                unique_ids = np.unique(self.segmentation_data)
                self._original_cellpose_ids = unique_ids[unique_ids >= 0]

                self.logger.info(
                    "Loaded segmentation from %s: %s (shifted to 0-based IDs, background is -1)",
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
        ):  # Sometimes this error occurs but then reselecting the right file would get
            # "The truth value of an array with more than one element is ambigous ... Use a.any() or a.all()
            # it is a bit tricky to manage and avoid stale state without confusing the user this wya, therefore we wipe

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

        self._refresh_neuron_colors()

        if self._neuron_ids is not None and self._neuron_colors is not None:
            self.overlay_image = self.image_data.copy().astype(np.float32)

            for i, seg_id in enumerate(self._neuron_ids):
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
        if self._csv_background_zero:
            n_background = np.sum(self.segmentation_data == 0)
            n_neurons = len(unique_ids[unique_ids > 0])
        else:
            n_background = np.sum(self.segmentation_data < 0)
            n_neurons = len(unique_ids[unique_ids >= 0])

        info = {
            "image_shape": self.image_data.shape,
            "segmentation_shape": self.segmentation_data.shape,
            "unique_ids": list(unique_ids),
            "background_pixels": n_background,
            "n_neurons": n_neurons,
        }

        if n_neurons > 0:
            if self._csv_background_zero:
                info["neuron_id_range"] = (
                    unique_ids[unique_ids > 0].min(),
                    unique_ids[unique_ids > 0].max(),
                )
            else:
                info["neuron_id_range"] = (
                    unique_ids[unique_ids >= 0].min(),
                    unique_ids[unique_ids >= 0].max(),
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
        *,
        mapping_strategy: str = "Linear",
        linear_offset: int = 0,
        aoi_config: AreasOfInterestConfig | None = None,
        segmentation_path: Path | None = None,
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
            mapping_strategy=mapping_strategy,
            linear_offset=linear_offset,
            aoi_config=clone_aoi_config(aoi_config),
            segmentation_path=segmentation_path,
            background_image=(
                None
                if self.image_data is None
                else np.array(self.image_data, copy=True)
            ),
        )

    def restore_from_result(self, result: CellposeImportResult) -> None:
        """Restore importer state from a previously saved result."""
        self.trace_data = result.trace_data
        self.spike_data = result.spike_data
        self._original_cellpose_ids = result.original_cellpose_ids
        self.segmentation_data = (
            None
            if result.segmentation_array is None
            else np.array(result.segmentation_array, copy=True)
        )
        self.overlay_image = (
            None
            if result.overlay_image is None
            else np.array(result.overlay_image, copy=True)
        )

        # Attempt to reload the original image; fall back to the stored raw
        # background, then the (colored) overlay, if unavailable.
        self.image_data = None
        if result.image_path.exists():
            try:
                self.load_image_file(result.image_path)
            except ValueError:
                self.logger.warning(
                    "Could not reload original image at %s; using stored overlay instead",
                    result.image_path,
                )
        if self.image_data is None and result.background_image is not None:
            self.image_data = np.array(result.background_image, copy=True)
        if self.image_data is None and self.overlay_image is not None:
            self.image_data = np.array(self.overlay_image, copy=True)

        self._refresh_neuron_colors()

    def _compute_neuron_colors(self, neuron_indices: np.ndarray | None) -> np.ndarray:
        if neuron_indices is None or neuron_indices.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        # neuron_indices are already 0-based, so use them directly
        colors_rgba_float = neuron_indices_to_colors(neuron_indices)
        return (colors_rgba_float[:, :3] * 255).astype(
            np.uint8
        )  # convert from decimal to RGB int8 integers

    def _refresh_neuron_colors(self) -> None:
        """Recompute neuron IDs/colors based on the current segmentation data."""
        if self.segmentation_data is None:
            self._neuron_ids = None
            self._neuron_colors = None
            return

        unique_ids = np.unique(self.segmentation_data)
        if self._csv_background_zero:
            neuron_ids = unique_ids[unique_ids > 0]
        else:
            neuron_ids = unique_ids[unique_ids >= 0]  # treat negatives as background
        if neuron_ids.size == 0:
            self._neuron_ids = None
            self._neuron_colors = None
            return

        self._neuron_ids = neuron_ids
        if self._csv_background_zero:
            color_indices = neuron_ids - neuron_ids.min()
        else:
            color_indices = neuron_ids
        self._neuron_colors = self._compute_neuron_colors(color_indices)

    @staticmethod
    def _read_csv_segmentation(path: Path) -> np.ndarray:
        return np.loadtxt(path, delimiter=",", dtype=np.int32)

    @staticmethod
    def _validate_csv_segmentation(segmentation: np.ndarray) -> np.ndarray:
        seg = np.asarray(segmentation, dtype=np.int32)
        if seg.ndim != 2:
            raise ValueError(f"Expected 2D array, got {seg.ndim}D array")
        if np.any(seg < 0):
            raise ValueError("Segmentation data contains negative values")
        if not np.isfinite(seg).all():
            raise ValueError("Segmentation data contains non-finite values")
        return seg

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

        if np.any(seg < -1):
            raise ValueError(
                "Segmentation data contains values below -1 after shifting"
            )

        if not np.isfinite(seg).all():
            raise ValueError("Segmentation data contains non-finite values")

        return seg

    @staticmethod
    def _shift_to_zero_based(segmentation: np.ndarray) -> np.ndarray:
        """Convert 1-based cell IDs to 0-based, set background to -1."""
        seg = np.array(segmentation, copy=True, dtype=np.int32)
        seg[seg >= 0] -= 1
        return seg
