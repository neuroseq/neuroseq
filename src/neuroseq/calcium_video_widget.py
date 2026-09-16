"""Raw calcium-imaging TIFF playback and stimulation overlay."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any
from typing import Protocol

import fastplotlib as fpl
import numpy as np
import tifffile
from PIL import Image
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.cell_overlay_widget import cellpose_to_stim_id
from neuroseq.cell_overlay_widget import compute_cell_centroids
from neuroseq.cell_overlay_widget import stim_id_to_cellpose_id
from neuroseq.fpl_widget import FPLWidget
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.stim import StimArray

ALL_STIM_ARRAYS_COMBINED = "All arrays combined"
HOVERED_CELL_BRIGHTNESS_BOOST = 1.8


class _LabelGraphic(Protocol):
    text: str
    offset: tuple[float, float, float]
    visible: bool


class CalciumVideoWidget(FPLWidget):
    """Display a raw TIFF movie and mark currently stimulated cells."""

    import_requested = QtCore.Signal()
    remove_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._pages: list[tuple[tifffile.TiffFile, int, int]] = []
        self._gif_images: list[Image.Image] = []
        self._cv_captures: list[Any] = []
        self._frame_refs: list[tuple[str, int, int]] = []
        super().__init__(parent=parent)
        self._paths: list[Path] = []
        self._fps: float | None = None
        self._cellpose_data: CellposeImportResult | None = None
        self._centroids: dict[int, tuple[float, float]] = {}
        self._stim_arrays: dict[str, StimArray] = {}
        self._current_frame_index: int | None = None
        self.current_frame_index: int | None = None
        self.current_frame: np.ndarray | None = None
        self._pages = []
        self._gif_images = []
        self._frame_refs = []
        self._frame_graphic = None
        self._cross_graphic: fpl.ScatterGraphic | None = None
        self.image_graphic = None
        self.cross_graphic = None
        self._label_graphics: dict[int, _LabelGraphic] = {}
        self._label_warning_shown = False
        self._video_warning_shown = False
        self._hovered_cellpose_id: int | None = None

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(4, 4, 4, 4)
        controls.addWidget(QtWidgets.QLabel("Stim source:"))
        self.stim_combo = QtWidgets.QComboBox(self)
        self.stim_combo.setMinimumWidth(180)
        self.stim_combo.currentIndexChanged.connect(self._refresh_current_frame)
        controls.addWidget(self.stim_combo)
        self.show_stimulation_labels_checkbox = QtWidgets.QCheckBox(
            "Show stimulation labels", self
        )
        self.show_stimulation_labels_checkbox.setEnabled(False)
        self.show_stimulation_labels_checkbox.toggled.connect(self._on_labels_toggled)
        controls.addStretch(1)
        self.save_button = QtWidgets.QPushButton("Save labeled video…", self)
        self.save_button.clicked.connect(self._on_save_labeled_video_clicked)
        self.remove_button = QtWidgets.QPushButton("Remove video", self)
        self.remove_button.clicked.connect(self.remove_requested.emit)
        self.video_actions_widget = QtWidgets.QWidget(self)
        actions = QtWidgets.QVBoxLayout(self.video_actions_widget)
        actions.setContentsMargins(4, 4, 4, 4)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.remove_button)
        button_row.addStretch(1)
        actions.addLayout(button_row)
        actions.addWidget(
            self.show_stimulation_labels_checkbox,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        self.import_button = QtWidgets.QPushButton("Import Video", self)
        self.import_button.clicked.connect(self.import_requested.emit)

        # Put the controls above the canvas created by FPLWidget.
        layout = self.layout()
        assert isinstance(layout, QtWidgets.QVBoxLayout)
        layout.insertLayout(0, controls)
        self.empty_state_widget = QtWidgets.QWidget(self)
        empty_layout = QtWidgets.QVBoxLayout(self.empty_state_widget)
        empty_layout.addStretch(1)
        label = QtWidgets.QLabel("No calcium video loaded.", self.empty_state_widget)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(label)
        empty_layout.addWidget(
            self.import_button, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        empty_layout.addStretch(1)
        layout.insertWidget(1, self.empty_state_widget)
        layout.addWidget(self.video_actions_widget)
        self.save_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.video_actions_widget.setVisible(False)
        self._set_canvas_visible(False)
        self.figure.renderer.add_event_handler(self._on_pointer_move, "pointer_move")

    def _set_canvas_visible(self, visible: bool) -> None:
        self.widget.setVisible(visible)
        self.save_button.setEnabled(visible)
        self.remove_button.setEnabled(visible)
        self.video_actions_widget.setVisible(visible)

    def _close_pages(self) -> None:
        for tif, _, _ in self._pages:
            with contextlib.suppress(Exception):
                tif.close()
        self._pages = []
        for image in self._gif_images:
            with contextlib.suppress(Exception):
                image.close()
        self._gif_images = []
        for capture in self._cv_captures:
            with contextlib.suppress(Exception):
                capture.release()
        self._cv_captures = []
        self._frame_refs = []

    def set_video_paths(
        self,
        paths: list[Path],
        fps: float | None,
        *,
        warn_if_missing_cellpose: bool = True,
    ) -> None:
        self._close_pages()
        self.clear_data()
        self._paths = list(paths)
        self._fps = float(fps) if fps is not None and fps > 0 else None
        self._current_frame_index = None
        self.current_frame_index = None
        self.current_frame = None
        if not self._paths or self._fps is None:
            self._video_warning_shown = False
            self._set_canvas_visible(False)
            self.empty_state_widget.setVisible(True)
            return
        try:
            pages: list[tuple[tifffile.TiffFile, int, int]] = []
            frame_refs: list[tuple[str, int, int]] = []
            for path in self._paths:
                if path.suffix.lower() == ".gif":
                    image = Image.open(path)
                    frame_count = int(getattr(image, "n_frames", 1))
                    gif_index = len(self._gif_images)
                    self._gif_images.append(image)
                    frame_refs.extend(
                        ("gif", gif_index, sub) for sub in range(frame_count)
                    )
                    continue
                if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                    import cv2

                    capture = cv2.VideoCapture(str(path))
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if not capture.isOpened() or frame_count <= 0:
                        capture.release()
                        raise ValueError("Video contains no readable frames")
                    capture_index = len(self._cv_captures)
                    self._cv_captures.append(capture)
                    frame_refs.extend(
                        ("cv2", capture_index, sub) for sub in range(frame_count)
                    )
                    continue
                tif = tifffile.TiffFile(path)
                if not tif.pages:
                    tif.close()
                    raise ValueError("TIFF contains no frames")
                for index, page in enumerate(tif.pages):
                    shape = tuple(page.shape)
                    if len(shape) == 3:
                        for sub in range(shape[0]):
                            pages.append((tif, index, sub))
                            frame_refs.append(("tiff", len(pages) - 1, sub))
                    else:
                        pages.append((tif, index, 0))
                        frame_refs.append(("tiff", len(pages) - 1, 0))
            self._pages = pages
            self._frame_refs = frame_refs
        except Exception:
            for tif, _, _ in pages:
                with contextlib.suppress(Exception):
                    tif.close()
            self._close_pages()
            self._set_canvas_visible(False)
            self.empty_state_widget.setVisible(True)
            return
        self.empty_state_widget.setVisible(False)
        self._set_canvas_visible(True)
        if not warn_if_missing_cellpose and self._cellpose_data is None:
            # Detached copies inherit an already-established video state and
            # must not present their initialization as a new upload.
            self._video_warning_shown = True
        if (
            warn_if_missing_cellpose
            and self._cellpose_data is None
            and not self._video_warning_shown
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Cell segmentation unavailable",
                "The video will play normally, but stimulation labels and "
                "crosses require uploaded Cellpose segmentation masks.",
            )
            self._video_warning_shown = True
        self.set_time_ms(0)

    def set_cellpose_data(self, cellpose_data: CellposeImportResult | None) -> None:
        self._cellpose_data = cellpose_data
        self._hovered_cellpose_id = None
        self.show_stimulation_labels_checkbox.setEnabled(cellpose_data is not None)
        if cellpose_data is None:
            self.show_stimulation_labels_checkbox.setChecked(False)
            self._label_warning_shown = False
        self._centroids = (
            compute_cell_centroids(cellpose_data.segmentation_array)
            if cellpose_data is not None
            else {}
        )
        self._refresh_current_frame()

    def set_stim_arrays(self, stim_arrays: dict[str, StimArray]) -> None:
        current = self.stim_combo.currentText()
        self._stim_arrays = dict(stim_arrays)
        self.stim_combo.blockSignals(True)
        self.stim_combo.clear()
        self.stim_combo.addItem(ALL_STIM_ARRAYS_COMBINED)
        self.stim_combo.addItems(list(self._stim_arrays))
        self.stim_combo.setCurrentText(
            current if current in self._stim_arrays else ALL_STIM_ARRAYS_COMBINED
        )
        self.stim_combo.blockSignals(False)
        self._refresh_current_frame()

    def _read_frame(self, index: int) -> np.ndarray:
        source, source_index, sub_index = self._frame_refs[index]
        if source == "gif":
            image = self._gif_images[source_index]
            image.seek(sub_index)
            frame = np.asarray(image.convert("L"))
        elif source == "cv2":
            import cv2

            capture = self._cv_captures[source_index]
            capture.set(cv2.CAP_PROP_POS_FRAMES, sub_index)
            ok, frame_data = capture.read()
            if not ok:
                raise ValueError("Could not read video frame")
            frame = np.asarray(cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY))
        else:
            tif, page_index, _ = self._pages[source_index]
            frame = np.asarray(tif.pages[page_index].asarray())
            if frame.ndim == 3:
                frame = frame[sub_index]
        while frame.ndim > 2:
            frame = frame[0]
        # CellposeImportDialog flips the reference image and masks vertically
        # when loading them. Match that in-app coordinate convention so
        # centroids and stimulation overlays land on the raw movie cells.
        return np.flipud(frame)

    @staticmethod
    def _to_rgb(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim != 2:
            frame = frame.squeeze()
        values = frame.astype(np.float32, copy=False)
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if hi <= lo:
            gray = np.zeros(values.shape, dtype=np.uint8)
        else:
            gray = np.clip((values - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    def _stimulated_cellpose_ids_at(self, time_ms: int) -> set[int]:
        if self._fps is None:
            return set()
        frame_index = max(0, int(float(time_ms) / 1000.0 * self._fps))
        start = frame_index / self._fps
        end = start + 1.0 / self._fps
        selected = self.stim_combo.currentText()
        arrays = (
            self._stim_arrays.values()
            if selected == ALL_STIM_ARRAYS_COMBINED
            else [self._stim_arrays[selected]]
            if selected in self._stim_arrays
            else []
        )
        result: set[int] = set()
        for array in arrays:
            dt = float(array.dt)
            if dt <= 0 or array.values.ndim != 2:
                continue
            first = max(0, int(np.floor(start / dt)))
            last = min(array.values.shape[0], int(np.ceil(end / dt)))
            if last <= first:
                continue
            for stim_id in np.flatnonzero(
                np.any(array.values[first:last] != 0, axis=0)
            ):
                cellpose_id = stim_id_to_cellpose_id(int(stim_id), self._cellpose_data)
                if cellpose_id is not None:
                    result.add(cellpose_id)
        return result

    def _set_crosses(self, cellpose_ids: set[int]) -> None:
        positions = [
            self._centroids[cell_id]
            for cell_id in cellpose_ids
            if cell_id in self._centroids
        ]
        if not positions:
            if self._cross_graphic is not None:
                with contextlib.suppress(Exception):
                    self.subplot.remove_graphic(self._cross_graphic)
                self._cross_graphic = None
                self.cross_graphic = None
            return
        data = np.column_stack(
            (
                np.asarray(positions, dtype=np.float32),
                np.zeros(len(positions), dtype=np.float32),
            )
        )
        if self._cross_graphic is not None:
            cross_data = self._cross_graphic.data
            cross_data_value = getattr(cross_data, "value", cross_data)
            cross_count = np.asarray(cross_data_value).shape[0]
        else:
            cross_count = 0
        if self._cross_graphic is not None and cross_count != len(data):
            with contextlib.suppress(Exception):
                self.subplot.remove_graphic(self._cross_graphic)
            self._cross_graphic = None
        if self._cross_graphic is None:
            self._cross_graphic = self.subplot.add_scatter(
                data,
                colors=(1.0, 0.0, 0.0, 1.0),
                sizes=16,
                size_space="screen",
                uniform_size=True,
                markers="x",
                uniform_marker=True,
                edge_width=1.0,
            )
            self.cross_graphic = self._cross_graphic
        else:
            self._cross_graphic.data = data
            self._cross_graphic.visible = True

    def _set_labels(self, cellpose_ids: set[int]) -> None:
        """Show stimulation IDs beside stimulated cells when enabled."""
        if not self.show_stimulation_labels_checkbox.isChecked() or not cellpose_ids:
            for label_graphic in self._label_graphics.values():
                label_graphic.visible = False
                with contextlib.suppress(Exception):
                    self.subplot.remove_graphic(label_graphic)
            self._label_graphics = {}
            return
        for label_graphic in self._label_graphics.values():
            label_graphic.visible = False
        for cellpose_id in cellpose_ids:
            point = self._centroids.get(cellpose_id)
            if point is None:
                continue
            stim_id = cellpose_to_stim_id(cellpose_id, self._cellpose_data)
            if stim_id is None:
                continue
            graphic: _LabelGraphic | None = self._label_graphics.get(cellpose_id)
            if graphic is None:
                graphic = self.subplot.add_text(
                    str(stim_id),
                    font_size=12,
                    offset=(point[0] + 4.0, point[1] - 4.0, 100.0),
                    face_color=(1.0, 0.0, 0.0),
                    outline_color=(1.0, 1.0, 1.0),
                    outline_thickness=2.0,
                    center=False,
                    screen_space=False,
                )
                self._label_graphics[cellpose_id] = graphic
            else:
                graphic.text = str(stim_id)
                graphic.offset = (point[0] + 4.0, point[1] - 4.0, 100.0)
            graphic.visible = True

    def _on_pointer_move(self, event: object) -> None:
        if self._cellpose_data is None or self.current_frame is None:
            return
        try:
            point = self.subplot.map_screen_to_world(event)
        except (AttributeError, TypeError, ValueError):
            return
        hovered: int | None = None
        if point is not None:
            x, y = int(point[0]), int(point[1])
            segmentation = self._cellpose_data.segmentation_array
            if 0 <= y < segmentation.shape[0] and 0 <= x < segmentation.shape[1]:
                cell_id = int(segmentation[y, x])
                if cell_id >= 0:
                    hovered = cell_id
        if hovered == self._hovered_cellpose_id:
            return
        self._hovered_cellpose_id = hovered
        self._update_frame_graphic()

    def _update_frame_graphic(self) -> None:
        if self._frame_graphic is None or self.current_frame is None:
            return
        frame = np.array(self.current_frame, copy=True)
        cell_id = self._hovered_cellpose_id
        if cell_id is not None and self._cellpose_data is not None:
            segmentation = self._cellpose_data.segmentation_array
            if segmentation.shape == frame.shape[:2]:
                mask = segmentation == cell_id
                frame[mask] = np.clip(
                    frame[mask].astype(np.float32) * HOVERED_CELL_BRIGHTNESS_BOOST,
                    0,
                    255,
                ).astype(np.uint8)
        self._frame_graphic.data = frame

    def _on_labels_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._label_warning_shown = False
        elif not self._label_warning_shown:
            problems: list[str] = []
            if self._cellpose_data is None:
                problems.append("Cell segmentation data is missing.")
            if not self._stim_arrays or not any(
                np.any(array.values != 0) for array in self._stim_arrays.values()
            ):
                problems.append("No nonzero stimulation samples are available.")
            if problems:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Stimulation labels unavailable",
                    "Stimulation labels require:\n\n" + "\n".join(problems),
                )
                self._label_warning_shown = True
        self._refresh_current_frame()

    def set_time_ms(self, time_ms: int) -> None:
        if not self._frame_refs or self._fps is None:
            return
        index = max(
            0, min(len(self._frame_refs) - 1, int(float(time_ms) / 1000.0 * self._fps))
        )
        frame = self._to_rgb(self._read_frame(index))
        self._current_frame_index = index
        self.current_frame_index = index
        self.current_frame = frame
        if self._frame_graphic is None:
            self._frame_graphic = self.subplot.add_image(frame, interpolation="nearest")
            self.image_graphic = self._frame_graphic
            self.subplot.axes.visible = False
            self.subplot.grids = False
            self.subplot.camera.maintain_aspect = True
            self.subplot.auto_scale(zoom=1.0, maintain_aspect=True)
        else:
            self._frame_graphic.data = frame
        self._update_frame_graphic()
        stimulated_ids = self._stimulated_cellpose_ids_at(int(time_ms))
        self._set_crosses(stimulated_ids)
        self._set_labels(stimulated_ids)
        self.figure.renderer.request_draw()

    def _refresh_current_frame(self) -> None:
        if self._current_frame_index is None or self._fps is None:
            return
        self.set_time_ms(round(self._current_frame_index * 1000.0 / self._fps))

    def _on_save_labeled_video_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save labeled video",
            "labeled_video.tif",
            "TIFF files (*.tif *.tiff);;GIF files (*.gif)",
        )
        if path:
            self.export_labeled_video(Path(path))

    def export_labeled_video(self, path: Path) -> None:
        if not self._frame_refs or self._fps is None:
            return
        if path.suffix.lower() == ".gif":
            from PIL import Image

            frames = [
                Image.fromarray(self._labeled_frame(index))
                for index in range(len(self._frame_refs))
            ]
            if frames:
                frames[0].save(
                    path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=max(1, round(1000.0 / self._fps)),
                    loop=0,
                )
            return
        with tifffile.TiffWriter(path) as writer:
            for index in range(len(self._frame_refs)):
                writer.write(self._labeled_frame(index), photometric="rgb")

    def _labeled_frame(self, index: int) -> np.ndarray:
        assert self._fps is not None
        time_ms = round(index * 1000.0 / self._fps)
        frame = self._to_rgb(self._read_frame(index))
        for cell_id in self._stimulated_cellpose_ids_at(time_ms):
            point = self._centroids.get(cell_id)
            if point is None:
                continue
            x, y = round(point[0]), round(point[1])
            if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
                continue
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                xx, yy = x + dx, y + dy
                if 0 <= yy < frame.shape[0] and 0 <= xx < frame.shape[1]:
                    frame[yy, xx] = (255, 0, 0)
        return frame

    def clear_data(self) -> None:
        super().clear_data()
        self._frame_graphic = None
        self._cross_graphic = None
        self.image_graphic = None
        self.cross_graphic = None
        self._label_graphics = {}
        self._hovered_cellpose_id = None

    def closeEvent(self, event) -> None:
        self._close_pages()
        super().closeEvent(event)
