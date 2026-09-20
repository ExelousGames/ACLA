"""Launch Labelme maximized without OpenCV's incompatible bundled Qt plugins."""

import importlib.util
import os
from pathlib import Path


def main() -> None:
    if importlib.util.find_spec("cv2") is not None:
        # imgviz imports OpenCV while Labelme starts. Import it first so its
        # environment changes can be cleared before PyQt5 creates the window.
        import cv2

        opencv_dir = Path(cv2.__file__).resolve().parent
        for name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
            value = os.environ.get(name)
            if value and Path(value).resolve().is_relative_to(opencv_dir):
                del os.environ[name]
    from labelme import __main__ as labelme_cli

    class MaximizedMainWindow(labelme_cli.MainWindow):
        def show(self) -> None:
            self.showMaximized()

    labelme_cli.MainWindow = MaximizedMainWindow
    labelme_cli.main()


if __name__ == "__main__":
    main()
