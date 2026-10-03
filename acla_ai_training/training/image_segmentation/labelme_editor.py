"""Launch Labelme maximized without OpenCV's incompatible bundled Qt plugins."""

import importlib.util
import math
import os
from pathlib import Path
import sys


def polyline_polygon_points(points: list[tuple[float, float]], width: float) -> list[tuple[float, float]]:
    """Stroke a locally rounded centerline with a constant-width polygon."""
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Polygon width must be positive and finite.")
    # Repeated clicks do not define a direction or need another vertex pair.
    centers = []
    for point in points:
        if not centers or point != centers[-1]:
            centers.append(point)
    if len(centers) < 2:
        raise ValueError("A polyline needs at least two distinct points.")

    import cv2
    import numpy as np
    from PyQt5 import QtCore, QtGui

    directions, lengths = [], []
    for (x1, y1), (x2, y2) in zip(centers, centers[1:]):
        length = math.hypot(x2 - x1, y2 - y1)
        lengths.append(length)
        directions.append(((x2 - x1) / length, (y2 - y1) / length))

    path = QtGui.QPainterPath(QtCore.QPointF(*centers[0]))
    for index in range(1, len(centers) - 1):
        x, y = centers[index]
        before, after = directions[index - 1], directions[index]
        # Sharper turns need longer transitions to leave room for the inner edge.
        alignment = before[0] * after[0] + before[1] * after[1]
        # Leave at least half of each segment for the neighboring corner.
        trim = min(width / max(1e-8, 1 + alignment), lengths[index - 1] / 2, lengths[index] / 2)
        path.lineTo(x - before[0] * trim, y - before[1] * trim)
        path.quadTo(x, y, x + after[0] * trim, y + after[1] * trim)
    path.lineTo(*centers[-1])

    stroker = QtGui.QPainterPathStroker()
    stroker.setWidth(width)
    stroker.setCapStyle(QtCore.Qt.FlatCap)
    stroker.setJoinStyle(QtCore.Qt.RoundJoin)
    stroker.setCurveThreshold(0.02)
    # Flatten at 4x resolution to keep both editable edges smooth to subpixels.
    # Simplifying the filled stroke removes loops at tight bends and reversals.
    scale = QtGui.QTransform.fromScale(4, 4)
    outline = scale.map(stroker.createStroke(path)).simplified().toFillPolygon()
    polygon = [(point.x() / 4, point.y() / 4) for point in outline]
    if polygon and polygon[-1] == polygon[0]:
        polygon.pop()
    # Keep editable vertices only where they change the outline visibly.
    # Scale the tolerance down for thin strokes and cap it at half a pixel.
    contour = cv2.approxPolyDP(
        np.asarray(polygon, dtype=np.float32), min(0.5, width * 0.025), closed=True,
    ).reshape(-1, 2)
    if len(contour) >= 3:
        return [(float(x), float(y)) for x, y in contour]
    # A very short, wide stroke can simplify to a line; retain its end caps.
    return polygon


def create_main_window(base_window):
    class PolylinePolygonMainWindow(base_window):
        _polygon_from_polyline = False
        _polygon_width = 15.0
        _backend_annotator = None
        _yolo26x_annotator = None

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
            self._custom_annotate_action = QtWidgets.QAction(
                actions.create_mode.icon(), "Custom Annotate", self,
            )
            self._custom_annotate_action.setToolTip(
                "Add polygons to this image using the newest trained model from the backend"
            )
            self._custom_annotate_action.setEnabled(False)
            self._custom_annotate_action.triggered.connect(self._custom_annotate)
            self._yolo26x_annotate_action = QtWidgets.QAction(
                actions.create_mode.icon(), "YOLO26x Annotate", self,
            )
            self._yolo26x_annotate_action.setToolTip(
                "Add polygon regions labeled 'other' from YOLO26x, then relabel with Edit Label"
            )
            self._yolo26x_annotate_action.setEnabled(False)
            self._yolo26x_annotate_action.triggered.connect(self._yolo26x_annotate)
            actions.draw.insert(1, ("polyline_polygon", create))
            return actions._replace(
                on_load_active=(
                    *actions.on_load_active, create, self._custom_annotate_action, self._yolo26x_annotate_action,
                ),
                context_menu=(create, width, *actions.context_menu),
                edit_menu=(self._custom_annotate_action, self._yolo26x_annotate_action, width, None, *actions.edit_menu),
            )

        def _setup_toolbars(self):
            from PyQt5 import QtCore, QtWidgets

            super()._setup_toolbars()
            toolbar = QtWidgets.QToolBar("Annotation", self)
            toolbar.setObjectName("Annotation")
            toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            toolbar.addAction(self._custom_annotate_action)
            toolbar.addAction(self._yolo26x_annotate_action)
            self.addToolBar(toolbar)

        def _on_drawing_polygon_changed(self, drawing=True):
            super()._on_drawing_polygon_changed(drawing)
            self._custom_annotate_action.setEnabled(not drawing and self._image_path is not None)
            self._yolo26x_annotate_action.setEnabled(not drawing and self._image_path is not None)

        def _custom_annotate(self):
            from training.image_segmentation.auto_annotation import BackendAutoAnnotator

            if self._backend_annotator is None:
                self._backend_annotator = BackendAutoAnnotator()
            self._auto_annotate(
                self._backend_annotator, "Custom Annotate",
                "Checking the latest backend version and local weights, then predicting polygons…",
            )

        def _yolo26x_annotate(self):
            from training.image_segmentation.auto_annotation import YOLO26xAutoAnnotator

            if self._yolo26x_annotator is None:
                self._yolo26x_annotator = YOLO26xAutoAnnotator()
            self._auto_annotate(
                self._yolo26x_annotator, "YOLO26x Annotate",
                "Checking local YOLO26x weights, downloading if missing, then predicting polygons…",
            )

        def _auto_annotate(self, annotator, title, message):
            from PyQt5 import QtCore, QtWidgets
            from PIL import Image
            from labelme import utils
            from labelme._shape import Shape

            if self._image_path is None or self._canvas_widgets.canvas.is_drawing:
                return
            # Use the displayed RGB image, including Labelme's orientation handling.
            image = Image.fromarray(utils.img_qt_to_arr(self._image)[:, :, :3].copy())
            labels = self._config["labels"]

            class PredictionThread(QtCore.QThread):
                polygons = None
                error = None

                def run(self):
                    try:
                        self.polygons = annotator.predict(image, labels)
                    except Exception as exc:
                        self.error = str(exc)

            class PredictionDialog(QtWidgets.QDialog):
                def reject(self):
                    # Keep the image fixed and the worker alive until it finishes.
                    pass

            dialog = PredictionDialog(self)
            dialog.setWindowTitle(title)
            dialog.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.addWidget(QtWidgets.QLabel(message))
            progress = QtWidgets.QProgressBar(dialog)
            progress.setRange(0, 0)
            layout.addWidget(progress)
            worker = PredictionThread(dialog)
            worker.finished.connect(dialog.accept)
            QtCore.QTimer.singleShot(0, worker.start)
            dialog.exec_()
            worker.wait()
            polygons, error = worker.polygons, worker.error
            dialog.deleteLater()
            if error is not None:
                QtWidgets.QMessageBox.warning(self, f"{title} Failed", error)
                return
            shapes = []
            for polygon in polygons:
                shape = Shape(label=polygon["label"], shape_type="polygon")
                for x, y in polygon["points"]:
                    shape.add_point(QtCore.QPointF(x, y))
                shape.close()
                shapes.append(shape)
            if shapes:
                self._switch_canvas_mode(edit=True)
                self._insert_shapes(shapes)
            self.show_status_message(
                f"Added {len(shapes)} polygons from {annotator.model_name}. Review and adjust the predictions.",
                10000,
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
