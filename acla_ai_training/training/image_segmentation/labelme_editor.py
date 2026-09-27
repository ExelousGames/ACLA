"""Launch Labelme maximized without OpenCV's incompatible bundled Qt plugins."""

import importlib.util
import math
import os
from pathlib import Path


def polyline_polygon_points(points: list[tuple[float, float]], width: float) -> list[tuple[float, float]]:
    """Offset each centerline point by half the width along its local normal."""
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Polygon width must be positive and finite.")
    # Repeated clicks do not define a direction or need another vertex pair.
    centers = []
    for point in points:
        if not centers or point != centers[-1]:
            centers.append(point)
    if len(centers) < 2:
        raise ValueError("A polyline needs at least two distinct points.")

    directions = []
    for (x1, y1), (x2, y2) in zip(centers, centers[1:]):
        length = math.hypot(x2 - x1, y2 - y1)
        directions.append(((x2 - x1) / length, (y2 - y1) / length))

    left, right = [], []
    for index, (x, y) in enumerate(centers):
        before = directions[max(0, index - 1)]
        after = directions[min(index, len(directions) - 1)]
        dx, dy = before[0] + after[0], before[1] + after[1]
        length = math.hypot(dx, dy)
        if length < 1e-8:  # A reversal has no bisector; use the outgoing segment.
            dx, dy = after
            length = 1.0
        ox, oy = -dy / length * width / 2, dx / length * width / 2
        left.append((x + ox, y + oy))
        right.append((x - ox, y - oy))
    return left + right[::-1]


def create_main_window(base_window):
    class PolylinePolygonMainWindow(base_window):
        _polygon_from_polyline = False
        _polygon_width = 3.0

        def _setup_actions(self):
            from PyQt5 import QtWidgets

            actions = super()._setup_actions()
            create = QtWidgets.QAction(
                actions.create_mode.icon(), "Polygon from Polyline", self,
            )
            create.setShortcut("Ctrl+Shift+L")
            create.setToolTip("Draw a centerline, then finish to create an editable polygon")
            create.setEnabled(False)
            create.triggered.connect(
                lambda: self._switch_canvas_mode(edit=False, create_mode="polyline_polygon")
            )
            width = QtWidgets.QAction("Set Polyline Polygon Width…", self)
            width.triggered.connect(self._set_polygon_width)
            actions.draw.insert(1, ("polyline_polygon", create))
            return actions._replace(
                on_load_active=(*actions.on_load_active, create),
                context_menu=(create, width, *actions.context_menu),
                edit_menu=(width, None, *actions.edit_menu),
            )

        def _set_polygon_width(self):
            from PyQt5 import QtWidgets

            width, accepted = QtWidgets.QInputDialog.getDouble(
                self, "Polyline Polygon Width", "Total width (image pixels):",
                self._polygon_width, 0.1, 100000.0, 1,
            )
            if accepted:
                self._polygon_width = width

        def _switch_canvas_mode(self, edit=True, create_mode=None):
            self._polygon_from_polyline = not edit and create_mode == "polyline_polygon"
            super()._switch_canvas_mode(
                edit=edit,
                create_mode="linestrip" if self._polygon_from_polyline else create_mode,
            )
            # The canvas uses linestrip, but the two tools remain separate modes.
            if self._polygon_from_polyline:
                for mode, action in self._actions.draw:
                    action.setEnabled(mode != "polyline_polygon")

        def _on_new_shape(self):
            if not self._polygon_from_polyline:
                return super()._on_new_shape()

            from PyQt5 import QtCore
            from labelme._shape import Shape

            canvas = self._canvas_widgets.canvas
            centerline = canvas.shapes[-1]
            polygon = Shape(shape_type="polygon")
            points = polyline_polygon_points(
                [(point.x(), point.y()) for point in centerline.points], self._polygon_width,
            )
            for x, y in points:
                polygon.add_point(QtCore.QPointF(
                    min(max(x, 0), canvas.pixmap.width()),
                    min(max(y, 0), canvas.pixmap.height()),
                ))
            polygon.close()
            canvas.shapes[-1] = polygon
            canvas.shape_backups[-1][-1] = polygon.copy()
            super()._on_new_shape()
            # Labelme reopens an unlabeled shape when the label dialog is cancelled.
            # Resume the original centerline so it can still be extended normally.
            if canvas._current is polygon:
                centerline.open()
                canvas._current = centerline
                canvas._line.points = [centerline.points[-1], centerline.points[0]]

        def show(self) -> None:
            self.showMaximized()

    return PolylinePolygonMainWindow


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

    labelme_cli.MainWindow = create_main_window(labelme_cli.MainWindow)
    labelme_cli.main()


if __name__ == "__main__":
    main()
