"""Raw calcium-imaging TIFF playback and stimulation overlay."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import fastplotlib as fpl
import numpy as np
import tifffile
from PIL import Image
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_scanimage import compact_index
from neuroseq.save_imaging_dialog import SaveImagingDialog
from neuroseq.stimulation.stim import StimArray
from neuroseq.visualization.cell_overlay_widget import _pixel_at
from neuroseq.visualization.fpl_widget import FPLWidget
from neuroseq.visualization_support.cellpose_ids import cellpose_to_stim_id
from neuroseq.visualization_support.cellpose_ids import compute_cell_centroids
from neuroseq.visualization_support.cellpose_ids import stim_id_to_cellpose_id

ALL_STIM_ARRAYS_COMBINED = "All arrays combined"
NO_STIM_ARRAY = "None"
HOVERED_CELL_BRIGHTNESS_BOOST = 1.8
# Frames and percentiles that set the grey scale of saved imaging data.
EXPORT_SCALE_SAMPLE_FRAMES = 50
EXPORT_SCALE_PERCENTILES = (0.1, 99.9)


class CalciumVideoWidget(FPLWidget):
    """Display a raw TIFF movie and mark currently stimulated cells."""

    import_requested = QtCore.Signal()
    remove_requested = QtCore.Signal()
    # The movie's files were not found; the user wants to pick their new place.
    locate_requested = QtCore.Signal()
    # Interleaved channels in the movie and the one shown (0 = all).
    channels_changed = QtCore.Signal(int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._pages: list[tuple[tifffile.TiffFile, int, int]] = []
        self._gif_images: list[Image.Image] = []
        self._cv_captures: list[Any] = []
        self._frame_refs: list[tuple[str, int, int]] = []
        super().__init__(parent=parent)
        self._paths: list[Path] = []
        self.missing_paths: list[Path] = []
        self._fps: float | None = None
        # Frame rate of each channel and every page of the movie, before a
        # channel is picked; _fps and _frame_refs are what is played.
        self._channel_fps: float | None = None
        self._all_frame_refs: list[tuple[str, int, int]] = []
        self._channels = 1
        self._shown_channel = 1
        # The last "Save imaging data" choice of original or marked frames.
        self._save_original = False
        self._cellpose_data: CellposeImportResult | None = None
        self._centroids: dict[int, tuple[float, float]] = {}
        self._id_label_graphics: list[fpl.TextGraphic] = []
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
        self._video_warning_shown = False
        self._stim_warning_shown = False
        self._hovered_cellpose_id: int | None = None
        self._scanimage_index: dict | None = None
        self._frame_times: np.ndarray | None = None

        self.stim_combo = QtWidgets.QComboBox(self)
        self.stim_combo.setMinimumWidth(180)
        self.stim_combo.currentIndexChanged.connect(self._on_stim_source_changed)
        self.save_button = QtWidgets.QPushButton("Save imaging data…", self)
        self.save_button.clicked.connect(self._on_save_labeled_video_clicked)
        self.remove_button = QtWidgets.QPushButton("Remove imaging data", self)
        self.remove_button.clicked.connect(self.remove_requested.emit)
        self.ids_checkbox = QtWidgets.QCheckBox("Show ids", self)
        self.ids_checkbox.setToolTip("Show each cell's neuron id on the video")
        self.ids_checkbox.setChecked(True)
        self.ids_checkbox.toggled.connect(self._update_id_labels)
        self.channels_spin = QtWidgets.QSpinBox(self)
        self.channels_spin.setRange(1, 8)
        self.channels_spin.setToolTip(
            "Number of channels interleaved frame by frame in the movie"
        )
        self.channel_combo = QtWidgets.QComboBox(self)
        self.channel_combo.setToolTip(
            "Show one channel, or every frame as stored in the file"
        )
        self.channels_spin.valueChanged.connect(self._on_channel_controls_changed)
        self.channel_combo.activated.connect(self._on_channel_controls_changed)
        self._fill_channel_combo()
        self.channel_controls = QtWidgets.QWidget(self)
        channel_layout = QtWidgets.QHBoxLayout(self.channel_controls)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.addWidget(QtWidgets.QLabel("Channels:"))
        channel_layout.addWidget(self.channels_spin)
        channel_layout.addWidget(QtWidgets.QLabel("Show:"))
        channel_layout.addWidget(self.channel_combo)
        self.video_actions_widget = QtWidgets.QWidget(self)
        actions = QtWidgets.QHBoxLayout(self.video_actions_widget)
        actions.setContentsMargins(4, 4, 4, 4)
        # Stim source lives in the same bottom bar as the other video
        # actions, rather than a separate row above the canvas, so this tab
        # matches the Cell segmentation tab's shape (content, then one fixed
        # bottom bar) and switching between them doesn't shift anything.
        actions.addWidget(QtWidgets.QLabel("Stim source:"))
        actions.addWidget(self.stim_combo)
        actions.addWidget(self.channel_controls)
        actions.addWidget(self.ids_checkbox)
        actions.addStretch(1)
        actions.addWidget(self.save_button)
        actions.addWidget(self.remove_button)
        self.video_actions_widget.setFixedHeight(44)
        self.video_actions_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.import_button = QtWidgets.QPushButton("Import Imaging Data", self)
        self.import_button.clicked.connect(self.import_requested.emit)

        layout = self.layout()
        assert isinstance(layout, QtWidgets.QVBoxLayout)
        self.empty_state_widget = QtWidgets.QWidget(self)
        empty_layout = QtWidgets.QVBoxLayout(self.empty_state_widget)
        empty_layout.addStretch(1)
        label = QtWidgets.QLabel(
            "No calcium imaging data loaded.", self.empty_state_widget
        )
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(label)
        empty_layout.addWidget(
            self.import_button, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        empty_layout.addStretch(1)
        layout.insertWidget(0, self.empty_state_widget)
        # Shown instead of the video when its files are not where the project
        # says (e.g. a project opened on another machine).
        self.missing_state_widget = QtWidgets.QWidget(self)
        missing_layout = QtWidgets.QVBoxLayout(self.missing_state_widget)
        missing_layout.addStretch(1)
        self.missing_label = QtWidgets.QLabel(self.missing_state_widget)
        self.missing_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.missing_label.setWordWrap(True)
        self.missing_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        missing_layout.addWidget(self.missing_label)
        self.locate_button = QtWidgets.QPushButton(
            "Locate file…", self.missing_state_widget
        )
        self.locate_button.setToolTip(
            "Choose where the original imaging file is on this computer"
        )
        self.locate_button.clicked.connect(self.locate_requested.emit)
        self.missing_remove_button = QtWidgets.QPushButton(
            "Remove imaging data", self.missing_state_widget
        )
        self.missing_remove_button.clicked.connect(self.remove_requested.emit)
        missing_buttons = QtWidgets.QHBoxLayout()
        missing_buttons.addStretch(1)
        missing_buttons.addWidget(self.locate_button)
        missing_buttons.addWidget(self.missing_remove_button)
        missing_buttons.addStretch(1)
        missing_layout.addLayout(missing_buttons)
        missing_layout.addStretch(1)
        layout.insertWidget(1, self.missing_state_widget)
        self.missing_state_widget.setVisible(False)
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
        self._all_frame_refs = []

    def set_video_paths(
        self,
        paths: list[Path],
        fps: float | None,
        *,
        warn_if_missing_cellpose: bool = True,
        scanimage_index: dict | None = None,
        channel: int | None = None,
        channels: int = 1,
        shown_channel: int = 1,
    ) -> None:
        """Load a movie. ``fps`` is the frame rate of each channel; a movie
        with ``channels`` interleaved channels shows ``shown_channel``
        (counted from 1, 0 for every frame)."""
        self._close_pages()
        self.clear_data()
        self._scanimage_index = None
        self._frame_times = None
        self._paths = list(paths)
        self.missing_paths = [path for path in self._paths if not path.is_file()]
        self.missing_state_widget.setVisible(False)
        self._fps = float(fps) if fps is not None and fps > 0 else None
        self._channel_fps = self._fps
        self._channels = max(1, int(channels))
        self._shown_channel = (
            min(max(0, int(shown_channel)), self._channels) if self._channels > 1 else 1
        )
        self._sync_channel_controls()
        self.channel_controls.setVisible(True)
        self._current_frame_index = None
        self.current_frame_index = None
        self.current_frame = None
        if not self._paths or self._fps is None:
            self._video_warning_shown = False
            self._set_canvas_visible(False)
            self.empty_state_widget.setVisible(True)
            return
        if self.missing_paths:
            self._show_missing_files()
            return
        try:
            pages: list[tuple[tifffile.TiffFile, int, int]] = []
            frame_refs: list[tuple[str, int, int]] = []
            self._scanimage_page_refs = {}
            for file_index, path in enumerate(self._paths):
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
                        raise ValueError("Imaging file contains no readable frames")
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
                        self._scanimage_page_refs[(file_index, index)] = (
                            "tiff",
                            len(pages),
                            0,
                        )
                        pages.append((tif, index, 0))
                        frame_refs.append(("tiff", len(pages) - 1, 0))
            self._pages = pages
            self._all_frame_refs = frame_refs
            self._apply_channels()
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
                "The imaging data will play normally, but stimulation crosses require "
                "uploaded Cellpose segmentation masks.",
            )
            self._video_warning_shown = True
        if scanimage_index is not None:
            self.set_scanimage_index(scanimage_index, channel)
        self.set_time_ms(0)

    def _show_missing_files(self) -> None:
        heading = (
            "The imaging file of this project was not found:"
            if len(self.missing_paths) == 1
            else "These imaging files of this project were not found:"
        )
        names = "\n".join(str(path) for path in self.missing_paths)
        self.missing_label.setText(f"{heading}\n{names}")
        self._set_canvas_visible(False)
        self.empty_state_widget.setVisible(False)
        self.missing_state_widget.setVisible(True)

    def _apply_channels(self) -> None:
        if self._channels > 1 and self._shown_channel > 0:
            self._frame_refs = self._all_frame_refs[
                self._shown_channel - 1 :: self._channels
            ]
            self._fps = self._channel_fps
        else:
            # Every page in file order: the channels share each frame period.
            self._frame_refs = list(self._all_frame_refs)
            self._fps = (
                self._channel_fps * self._channels
                if self._channel_fps is not None
                else None
            )

    def set_channels(self, channels: int, shown_channel: int) -> None:
        """Choose how many channels the movie interleaves and which to show
        (counted from 1, 0 for every frame)."""
        channels = max(1, int(channels))
        shown_channel = min(max(0, int(shown_channel)), channels) if channels > 1 else 1
        if (channels, shown_channel) == (self._channels, self._shown_channel):
            return
        self._channels = channels
        self._shown_channel = shown_channel
        self._sync_channel_controls()
        if self._scanimage_index is None and self._all_frame_refs:
            time_ms = self._current_time_ms()
            self._apply_channels()
            self.set_time_ms(time_ms)

    def _current_time_ms(self) -> int:
        if self._current_frame_index is None or not self._fps:
            return 0
        return round(self._current_frame_index * 1000.0 / self._fps)

    def _fill_channel_combo(self) -> None:
        self.channel_combo.clear()
        if self._channels > 1:
            for channel in range(1, self._channels + 1):
                self.channel_combo.addItem(f"Channel {channel}", channel)
        self.channel_combo.addItem("All frames", 0)

    def _sync_channel_controls(self) -> None:
        self.channels_spin.blockSignals(True)
        self.channels_spin.setValue(self._channels)
        self.channels_spin.blockSignals(False)
        self._fill_channel_combo()
        self.channel_combo.setCurrentIndex(
            self.channel_combo.findData(
                self._shown_channel if self._channels > 1 else 0
            )
        )
        self.channel_combo.setEnabled(self._channels > 1)

    def _on_channel_controls_changed(self, *_args: object) -> None:
        channels = self.channels_spin.value()
        # A new channel count starts with its first channel.
        shown = 1 if channels != self._channels else self.channel_combo.currentData()
        self.set_channels(channels, shown)
        self.channels_changed.emit(self._channels, self._shown_channel)

    def set_scanimage_index(self, index: dict, channel: int | None = None) -> None:
        self._scanimage_index = compact_index(index)
        # ScanImage recordings choose their channel from their own metadata.
        self.channel_controls.setVisible(False)
        self.set_recorded_channel(channel or index["channels"][0])

    def set_recorded_channel(self, channel: int) -> None:
        if self._scanimage_index is None:
            return
        frames = self._scanimage_index["frames"]
        channels = self._scanimage_index["channels"]
        if channel not in channels:
            channel = channels[0]
        self.recorded_channel = channel
        selected = [i for i, c in enumerate(frames["channel"]) if c == channel]
        self._frame_refs = [
            self._scanimage_page_refs[(frames["file"][i], frames["page"][i])]
            for i in selected
        ]
        self._frame_times = np.asarray([frames["time_s"][i] for i in selected])
        self.set_time_ms(0)

    def set_cellpose_data(self, cellpose_data: CellposeImportResult | None) -> None:
        self._cellpose_data = cellpose_data
        self._hovered_cellpose_id = None
        self._centroids = (
            compute_cell_centroids(cellpose_data.segmentation_array)
            if cellpose_data is not None
            else {}
        )
        self._update_id_labels()
        self._refresh_current_frame()

    def _update_id_labels(self, *_args: object) -> None:
        """Label each mapped cell with its neuron id, as in the Spatial
        Footprints panel (issue #596)."""
        for graphic in self._id_label_graphics:
            with contextlib.suppress(Exception):
                self.subplot.remove_graphic(graphic)
        self._id_label_graphics = []
        if (
            not self.ids_checkbox.isChecked()
            or self._frame_graphic is None
            or self._cellpose_data is None
            # A frame that does not share the masks' pixel grid.
            or self._frame_graphic.data.value.shape[:2]
            != self._cellpose_data.segmentation_array.shape
        ):
            return
        for cell_id, (x, y) in self._centroids.items():
            neuron_id = cellpose_to_stim_id(cell_id, self._cellpose_data)
            if neuron_id is None:
                continue
            self._id_label_graphics.append(
                self.subplot.add_text(
                    str(neuron_id),
                    font_size=14,
                    face_color=(1.0, 1.0, 0.4, 1.0),
                    outline_color=(0.0, 0.0, 0.0, 1.0),
                    outline_thickness=0.3,
                    offset=(x, y, 150.0),
                    screen_space=True,
                    # Beside the centre, where the stimulation cross is drawn.
                    anchor="top-left",
                )
            )
        self.figure.renderer.request_draw()

    def set_stim_arrays(self, stim_arrays: dict[str, StimArray]) -> None:
        current = self.stim_combo.currentText()
        self._stim_arrays = dict(stim_arrays)
        self.stim_combo.blockSignals(True)
        self.stim_combo.clear()
        self.stim_combo.addItem(NO_STIM_ARRAY)
        self.stim_combo.addItem(ALL_STIM_ARRAYS_COMBINED)
        self.stim_combo.addItems(list(self._stim_arrays))
        selected = (
            NO_STIM_ARRAY
            if current == NO_STIM_ARRAY
            else current
            if current in self._stim_arrays
            else ALL_STIM_ARRAYS_COMBINED
        )
        self.stim_combo.setCurrentText(selected)
        self.stim_combo.blockSignals(False)
        self._refresh_current_frame()

    def _on_stim_source_changed(self) -> None:
        selected = self.stim_combo.currentText()
        if selected != NO_STIM_ARRAY and not self._has_nonzero_stimulation(selected):
            if not self._stim_warning_shown:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Stimulation data unavailable",
                    "The selected stimulation source is empty. The imaging data will "
                    "continue to play without stimulation crosses.",
                )
                self._stim_warning_shown = True
        else:
            self._stim_warning_shown = False
        self._refresh_current_frame()

    def _has_nonzero_stimulation(self, selected: str) -> bool:
        arrays = (
            self._stim_arrays.values()
            if selected == ALL_STIM_ARRAYS_COMBINED
            else [self._stim_arrays[selected]]
            if selected in self._stim_arrays
            else []
        )
        return any(np.any(array.values != 0) for array in arrays)

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
                raise ValueError("Could not read imaging frame")
            frame = np.asarray(cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY))
        else:
            tif, page_index, _ = self._pages[source_index]
            frame = np.asarray(tif.pages[page_index].asarray())
            if frame.ndim == 3:
                frame = frame[sub_index]
        while frame.ndim > 2:
            frame = frame[0]
        return frame

    @staticmethod
    def _to_rgb(
        frame: np.ndarray, value_range: tuple[float, float] | None = None
    ) -> np.ndarray:
        """Grey RGB of a frame, scaled to ``value_range`` or else to the
        frame's own minimum and maximum."""
        frame = np.asarray(frame)
        if frame.ndim != 2:
            frame = frame.squeeze()
        values = frame.astype(np.float32, copy=False)
        lo, hi = (
            value_range
            if value_range is not None
            else (float(np.nanmin(values)), float(np.nanmax(values)))
        )
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
        if selected == NO_STIM_ARRAY:
            return result
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
                markers="x",
                edge_width=1.0,
            )
            self.cross_graphic = self._cross_graphic
        else:
            self._cross_graphic.data = data
            self._cross_graphic.visible = True

    def _on_pointer_move(self, event: object) -> None:
        if self._cellpose_data is None or self.current_frame is None:
            return
        try:
            point = self.subplot.map_screen_to_world(event)
        except (AttributeError, TypeError, ValueError):
            return
        hovered: int | None = None
        if point is not None:
            x, y = _pixel_at(float(point[0]), float(point[1]))
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

    def frame_index_at(self, time_ms: float) -> int | None:
        """Index of the movie frame shown at ``time_ms``; None without a movie."""
        if not self._frame_refs or self._fps is None:
            return None
        if self._frame_times is not None:
            index = (
                int(np.searchsorted(self._frame_times, time_ms / 1000, side="right"))
                - 1
            )
        else:
            index = int(float(time_ms) / 1000.0 * self._fps)
        return max(0, min(len(self._frame_refs) - 1, index))

    def frame_count(self) -> int:
        return len(self._frame_refs)

    def set_time_ms(self, time_ms: int) -> None:
        index = self.frame_index_at(time_ms)
        if index is None:
            return
        frame = self._to_rgb(self._read_frame(index))
        self._current_frame_index = index
        self.current_frame_index = index
        self.current_frame = frame
        self._draw_current_frame()
        # ScanImage recordings have no assumed alignment with the stimulus.
        stimulated_ids = (
            self._stimulated_cellpose_ids_at(int(time_ms))
            if self._scanimage_index is None
            else set()
        )
        self._set_crosses(stimulated_ids)
        self.figure.renderer.request_draw()

    def _draw_current_frame(self) -> None:
        frame = self.current_frame
        if frame is None:
            return
        if (
            self._frame_graphic is not None
            and self._frame_graphic.data.value.shape != frame.shape
        ):
            # Movies of another size; fastplotlib images cannot change shape
            # in place.
            with contextlib.suppress(Exception):
                self.subplot.remove_graphic(self._frame_graphic)
            self._frame_graphic = None
        if self._frame_graphic is None:
            self._frame_graphic = self.subplot.add_image(frame, interpolation="nearest")
            self.image_graphic = self._frame_graphic
            self.subplot.axes.visible = False
            self.subplot.grids = False
            self.subplot.camera.maintain_aspect = True
            self.show_images_top_down()
            self.subplot.auto_scale(zoom=1.0, maintain_aspect=True)
            self._update_id_labels()
        else:
            self._frame_graphic.data = frame
        self._update_frame_graphic()
        self.figure.renderer.request_draw()

    def _refresh_current_frame(self) -> None:
        if self._current_frame_index is None or self._fps is None:
            return
        time_s = (
            self._frame_times[self._current_frame_index]
            if self._frame_times is not None
            else self._current_frame_index / self._fps
        )
        self.set_time_ms(round(time_s * 1000))

    def _on_save_labeled_video_clicked(self) -> None:
        if not self._frame_refs or self._fps is None:
            return
        current = self._current_frame_index or 0
        options = SaveImagingDialog(
            len(self._frame_refs),
            current + 1,
            original=self._save_original,
            parent=self,
        )
        if options.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._save_original = options.original
        first, last = options.frame_range
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save imaging data",
            "imaging.tif",
            "TIFF files (*.tif *.tiff)"
            if options.original
            else "TIFF files (*.tif *.tiff);;GIF files (*.gif)",
        )
        if path:
            try:
                self.export_labeled_video(
                    Path(path), first=first, last=last, original=options.original
                )
            except Exception as exc:
                self.logger.exception("Could not save labeled imaging data")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Could not save imaging data",
                    f"The imaging data could not be saved:\n{exc}",
                )

    def export_labeled_video(
        self,
        path: Path,
        *,
        first: int = 0,
        last: int | None = None,
        original: bool = False,
    ) -> None:
        """Save frames ``first`` to ``last`` (counted from 0, both included,
        all by default) of the shown channel. ``original`` keeps the recorded
        pixel values as a TIFF stack; otherwise the frames are saved as
        displayed, with the stimulated cells marked."""
        if not self._frame_refs or self._fps is None:
            return
        last = len(self._frame_refs) - 1 if last is None else last
        indices = range(max(0, first), min(len(self._frame_refs) - 1, last) + 1)
        if not indices:
            raise ValueError("No frames in the chosen range")
        if original:
            self._export_original_tiff(path, indices)
            return
        value_range = self._export_value_range(indices)
        if path.suffix.lower() == ".gif":
            self._export_gif_streaming(path, indices, value_range)
            return
        with tifffile.TiffWriter(path) as writer:
            for index in indices:
                writer.write(self._labeled_frame(index, value_range), photometric="rgb")

    def _export_value_range(self, indices: range) -> tuple[float, float]:
        """One grey scale for every saved frame. Scaling each frame to its own
        minimum and maximum made the background flicker in the saved movie,
        as noise and hot pixels change those from frame to frame (issue
        #618). Robust percentiles of up to 50 frames spread over the range
        set the scale instead."""
        step = max(1, len(indices) // EXPORT_SCALE_SAMPLE_FRAMES)
        sample = np.stack(
            [
                np.asarray(self._read_frame(index), dtype=np.float32)
                for index in indices[::step][:EXPORT_SCALE_SAMPLE_FRAMES]
            ]
        )
        lo, hi = np.nanpercentile(sample, EXPORT_SCALE_PERCENTILES)
        if hi <= lo:
            lo, hi = float(np.nanmin(sample)), float(np.nanmax(sample))
        return float(lo), float(hi)

    def _export_original_tiff(self, path: Path, indices: range) -> None:
        """Write the recorded frames as one stack, like Fiji's substack."""
        sample = np.asarray(self._read_frame(indices[0]))
        frames = (np.asarray(self._read_frame(index)) for index in indices)
        # ImageJ hyperstacks (frame interval, Fiji's frame axis) support only
        # these sample types.
        imagej = sample.dtype in (np.uint8, np.uint16, np.float32)
        tifffile.imwrite(
            path,
            data=frames,
            shape=(len(indices), *sample.shape),
            dtype=sample.dtype,
            imagej=imagej,
            metadata={"axes": "TYX", "finterval": 1.0 / self._fps}
            if imagej and self._fps
            else None,
        )

    def _export_gif_streaming(
        self, path: Path, indices: range, value_range: tuple[float, float]
    ) -> None:
        """Write a GIF one frame at a time to avoid buffering the movie."""
        from PIL import GifImagePlugin

        duration_ms = max(1, round(1000.0 / self._fps)) if self._fps else 1
        with path.open("wb") as file:
            for index in indices:
                if self._frame_times is not None and index + 1 < len(self._frame_times):
                    duration_ms = max(
                        1,
                        round(
                            1000
                            * (self._frame_times[index + 1] - self._frame_times[index])
                        ),
                    )
                frame = GifImagePlugin._normalize_mode(  # type: ignore[attr-defined]
                    Image.fromarray(self._labeled_frame(index, value_range), mode="RGB")
                )
                params = {
                    "duration": duration_ms,
                    "disposal": 2,
                    # Keep a local palette per frame; this avoids collecting
                    # all frames just to calculate one global palette.
                    "include_color_table": index > indices[0],
                }
                if index == indices[0]:
                    for header in GifImagePlugin._get_global_header(  # type: ignore[attr-defined]
                        frame, {"duration": duration_ms, "loop": 0}
                    ):
                        file.write(header)
                GifImagePlugin._write_frame_data(  # type: ignore[attr-defined]
                    file, frame, (0, 0), params
                )
            file.write(b";")

    def _labeled_frame(
        self, index: int, value_range: tuple[float, float] | None = None
    ) -> np.ndarray:
        assert self._fps is not None
        time_ms = round(index * 1000.0 / self._fps)
        frame = self._to_rgb(self._read_frame(index), value_range)
        if self._scanimage_index is not None:
            return frame
        height, width = frame.shape[:2]
        # An "x" like the one in the Imaging tab, scaled with the image so it
        # stays visible in a full-size recording (issue #618).
        arm = max(2, round(min(height, width) / 64))
        thickness = max(1, round(arm / 3))
        steps = np.arange(-arm, arm + 1)
        widths = np.arange(thickness) - (thickness - 1) // 2
        for cell_id in self._stimulated_cellpose_ids_at(time_ms):
            point = self._centroids.get(cell_id)
            if point is None:
                continue
            x, y = round(point[0]), round(point[1])
            if not (0 <= y < height and 0 <= x < width):
                continue
            for direction in (1, -1):
                for offset in widths:
                    xx = x + steps + offset
                    yy = y + direction * steps
                    inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                    frame[yy[inside], xx[inside]] = (255, 0, 0)
        return frame

    def clear_data(self) -> None:
        super().clear_data()
        self._frame_graphic = None
        self._cross_graphic = None
        self._id_label_graphics = []
        self.image_graphic = None
        self.cross_graphic = None
        self._stim_warning_shown = False
        self._hovered_cellpose_id = None

    def closeEvent(self, event) -> None:
        self._close_pages()
        super().closeEvent(event)
