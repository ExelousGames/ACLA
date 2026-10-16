"""Read prepared frame positions and show their paths without editing annotations."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from PyQt5 import QtCore, QtGui, QtWidgets


@dataclass
class PositionFrame:
    image: Path
    session_id: str
    reference: tuple[float, float, float]
    centerline: tuple[float, float, float]
    driver: tuple[float, float, float]


@dataclass
class PositionContext:
    frames: list[PositionFrame | None]
    current_index: int
    skipped: int


def _read_position(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("session_id"), str) or not data["session_id"]:
        raise ValueError(f"Position file needs a nonempty session_id: {path.name}")
    name = data.get("image")
    if not isinstance(name, str) or Path(name).name != name or Path(name).stem != path.name.removesuffix(".position.json"):
        raise ValueError(f"Position file must name its matching image: {path.name}")
    image = path.parent / name
    if not image.is_file():
        raise FileNotFoundError(f"Image is missing: {image.name}")
    for field in ("references", "target"):
        values = data.get(field)
        if not (
            isinstance(values, list) and len(values) == 2
            and all(isinstance(row, list) and len(row) == 3 for row in values)
            and all(type(value) in (int, float) and math.isfinite(value) for row in values for value in row)
        ):
            raise ValueError(f"{field} must contain two finite XYZ triples: {path.name}")
    reference, centerline = data["references"]
    # Preparation saves target = reference - driver, in the same world axes.
    driver = tuple(value - delta for value, delta in zip(reference, data["target"][0]))
    return PositionFrame(image, data["session_id"], tuple(reference), tuple(centerline), driver)


def load_position_context(image: Path, radius: int = 100) -> PositionContext:
    """Read nearby prepared frames in this video folder, preserving gaps in the paths."""
    image = image.resolve()
    position_path = image.with_suffix(".position.json")
    current = _read_position(position_path)
    if current.image != image:
        raise ValueError(f"Position file belongs to another image: {position_path.name}")
    paths = sorted(
        image.parent.glob("*.position.json"),
        key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)],
    )
    index = paths.index(position_path)
    start = max(0, index - radius)
    frames = []
    for path in paths[start:index + radius + 1]:
        try:
            frame = current if path == position_path else _read_position(path)
            frames.append(frame if frame.session_id == current.session_id else None)
        except (OSError, ValueError):
            # Do not draw a connecting segment over a malformed or unavailable frame.
            frames.append(None)
    return PositionContext(frames, index - start, frames.count(None))


class PositionMap(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setBackgroundBrush(QtGui.QColor("#18212b"))
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setMinimumSize(280, 220)

    def fit_paths(self):
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_paths()


class PositionPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.context = None
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Frames before / after:"))
        self.radius = QtWidgets.QSpinBox()
        self.radius.setRange(0, 5000)
        self.radius.setValue(100)
        self.radius.setKeyboardTracking(False)
        controls.addWidget(self.radius)
        controls.addWidget(QtWidgets.QLabel("Axes:"))
        self.axes = QtWidgets.QComboBox()
        for label, axes in (("X / Z", (0, 2)), ("X / Y", (0, 1)), ("Y / Z", (1, 2))):
            self.axes.addItem(label, axes)
        self.axes.setToolTip("Choose the horizontal plane for the saved world coordinates")
        controls.addWidget(self.axes)
        layout.addLayout(controls)
        self.map = PositionMap()
        layout.addWidget(self.map, 1)
        legend = QtWidgets.QLabel(
            '<span style="color:#8899aa">━━ Centerline</span> &nbsp; '
            '<span style="color:#39b87f">━━ Reference lap</span> &nbsp; '
            '<span style="color:#e89632">━━ Driver trajectory</span>'
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.status = QtWidgets.QLabel("Open an image with a matching .position.json file.")
        self.status.setTextFormat(QtCore.Qt.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.radius.valueChanged.connect(self._refresh)
        self.axes.currentIndexChanged.connect(self._draw)

    def set_image(self, image):
        self.image = Path(image) if image is not None else None
        self._refresh()

    def _refresh(self):
        self.context = None
        self.map.scene().clear()
        if self.image is None:
            self.status.setText("Open an image with a matching .position.json file.")
            return
        if not self.image.with_suffix(".position.json").is_file():
            self.status.setText(f"No position data for {self.image.name}.")
            return
        try:
            self.context = load_position_context(self.image, self.radius.value())
        except (OSError, ValueError) as exc:
            self.status.setText(f"Cannot read position data: {exc}")
            return
        count = len(self.context.frames) - self.context.skipped
        message = f"{self.image.name} · {count} frames · dots mark the current frame"
        if self.context.skipped:
            message += f"\nSkipped {self.context.skipped} unavailable, invalid or other-session frames."
        self.status.setText(message)
        self._draw()

    def _draw(self):
        scene = self.map.scene()
        scene.clear()
        if self.context is None:
            return
        horizontal, vertical = self.axes.currentData()
        current = self.context.frames[self.context.current_index]
        origin = current.driver

        def project(xyz):
            return QtCore.QPointF(xyz[horizontal] - origin[horizontal], origin[vertical] - xyz[vertical])

        points = []
        for field, color in (("centerline", "#8899aa"), ("reference", "#39b87f"), ("driver", "#e89632")):
            pen = QtGui.QPen(QtGui.QColor(color), 2)
            pen.setCosmetic(True)
            if field == "centerline":
                pen.setStyle(QtCore.Qt.DashLine)
            path = QtGui.QPainterPath()
            connected = False
            for frame in self.context.frames:
                if frame is None:
                    connected = False
                    continue
                point = project(getattr(frame, field))
                points.append(point)
                if connected:
                    path.lineTo(point)
                else:
                    path.moveTo(point)
                connected = True
            scene.addPath(path, pen)
            marker = scene.addEllipse(-4, -4, 8, 8, QtGui.QPen(QtCore.Qt.white), QtGui.QBrush(QtGui.QColor(color)))
            marker.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            marker.setPos(project(getattr(current, field)))
            marker.setZValue(1)
        bounds = QtGui.QPolygonF(points).boundingRect()
        margin = max(bounds.width(), bounds.height(), 1) * 0.08
        scene.setSceneRect(bounds.adjusted(-margin, -margin, margin, margin))
        self.map.fit_paths()
