from __future__ import annotations

import numpy as np
from pytestqt.qtbot import QtBot

from neuroseq.segmented_image_widget import SegmentedImageWidget


def test_segmented_image_widget(qtbot: QtBot) -> None:
    widget = SegmentedImageWidget()
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitUntil(lambda: widget.isVisible())

    # grayscale image
    n_rows = 30
    n_cols = 30
    image = np.zeros((n_rows, n_cols), dtype=np.float32)
    image[2:5, 3:7] = 0.4
    image[3:4, 4:6] = 0.6
    image[6:9, 10:14] = 0.7
    image[0:2, 0:3] = 1.0
    widget.add_image(image)
    qtbot.wait(2000)

    # labels
    labels = np.zeros((n_rows, n_cols), dtype=np.int32)
    labels[2:5, 3:7] = 1
    labels[6:9, 10:14] = 2
    labels[0:2, 0:3] = 3
    widget.add_labels(labels)
    qtbot.wait(10000)
