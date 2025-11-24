from __future__ import annotations

from typing import Any

import brian2.numpy_ as np
import cmap as cmap_lib
from PIL import Image
from PIL import ImageSequence

from neuroseq.utils import cmap_transform_neuron_indices


class CellposeImportResult(TypedDict, total=False):
    image_path: Path
    segmentation_array: np.ndarray
    overlay_image: np.ndarray


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
