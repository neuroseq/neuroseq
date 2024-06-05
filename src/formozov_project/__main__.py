from __future__ import annotations

import logging
import sys

from qtpy import QtWidgets

from formozov_project.gui import Gui


def main() -> None:
    logging.basicConfig(
        format="%(levelname)s %(module)s.%(funcName)s.%(lineno)d :: %(message)s"
    )
    app = QtWidgets.QApplication(sys.argv)
    gui = Gui()
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
