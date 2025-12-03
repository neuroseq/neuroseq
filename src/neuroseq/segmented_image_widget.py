from __future__ import annotations

import logging

import brian2.numpy_ as np
import fastplotlib
import fastplotlib as fpl
import pygfx
from cmap import Colormap
from fastplotlib.layouts import Subplot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.utils import QtABCMeta


class SegmentedImageWidget(QtWidgets.QWidget, metaclass=QtABCMeta):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure: fpl.Figure = fpl.Figure(canvas_kwargs={"parent": self})
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.widget = self.figure.show()
        layout.addWidget(self.widget)
        self.labels: np.ndarray | None = None
        self.text: fastplotlib.Graphic | None = None
        self.show_text = True
        self.setLayout(layout)
        self.subplot: Subplot = self.figure[0, 0]
        self.setMinimumSize(QtCore.QSize(100, 100))
        self.resize(800, 600)
        self.logger = logging.getLogger(__name__)
        self.mouse_pos: np.ndarray | None = None
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")

    def add_image(self, array: np.ndarray | None = None) -> None:
        if array is None:
            return
        self.subplot.add_image(
            array,
            cmap="matlab:gray",
            interpolation="nearest",
        )
        self.subplot.auto_scale()

    def add_labels(self, array: np.ndarray | None = None) -> None:
        if array is None:
            return
        self.labels = array
        # use a cmap to convert label indices to rgba colors
        cmap = Colormap("jet")
        norm = float(np.max(array))
        if norm == 0.0:
            norm = 1.0
        rgb = np.array([[np.asarray(cmap(v / norm)) for v in row] for row in array])
        rgb[:, :, 3] = 0.4  # set alpha to 0.4 for labels
        rgb[array == 0, 3] = 0  # make background labels with label 0 fully transparent
        self.subplot.add_image(
            rgb,
            interpolation="nearest",
        )
        # an example of adding some fixed black text with a white outline, with a size in data space:
        for i, offset in enumerate(
            [(6.0, 3.0, 200.0), (15.0, 12.0, 200.0), (24.0, 24.0, 200.0)]
        ):
            self.subplot.add_text(
                f"{i}",
                font_size=1,
                offset=offset,
                face_color=(0.0, 0.0, 0.0),
                outline_color=(1.0, 1.0, 1.0),
                outline_thickness=3.0,
                center=True,
                screen_space=False,
            )
        # dynamic text wiht a size in screen space that we'll make follow the mouse pointer:
        self.text = self.subplot.add_text(
            "",
            font_size=24,
            offset=(8.0, 4.0, 200.0),
            face_color=(0.0, 0.0, 0.0),
            outline_color=(1.0, 1.0, 1.0),
            outline_thickness=3.0,
            center=False,
            anchor="bottom-right",
            screen_space=True,
        )
        self.subplot.auto_scale()

    def clear_data(self) -> None:
        self.subplot.clear()
        self.labels = None

    def pointer_move(self, ev: pygfx.PointerEvent) -> None:
        if self.text is None:
            return
        self.mouse_pos = self.subplot.map_screen_to_world(ev)
        if self.text is None or self.mouse_pos is None or self.labels is None:
            self.text.visible = False
            return
        x = int(self.mouse_pos[0])
        y = int(self.mouse_pos[1])
        if x < 0 or y < 0 or x >= self.labels.shape[1] or y >= self.labels.shape[0]:
            self.text.visible = False
            return
        neuron_index = self.labels[y, x]
        if neuron_index == 0:
            # background pixel
            self.text.visible = False
            return
        self.text.visible = True
        self.text.offset = (self.mouse_pos[0], self.mouse_pos[1], 200)
        self.text.text = f"{neuron_index}"

    def closeEvent(self, event):
        if self.figure is not None:
            # stop the renderer before closing the widget otherwise it may try to render to the deleted object
            self.figure.close()
            self.figure = None
        super().closeEvent(event)
