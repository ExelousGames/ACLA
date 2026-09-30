from __future__ import annotations

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.image_segmentation import PACKAGE_DIR
from training.image_segmentation.labelme_editor import create_main_window, polyline_polygon_points


@pytest.mark.parametrize("points", [
    [(10, 20), (30, 20)],
    [(20, 10), (20, 30)],
    [(10, 10), (30, 30), (50, 10)],
    [(10, 10), (30, 10), (30, 50)],
    [(10, 10), (30, 10), (10, 10)],
])
def test_polygon_has_one_vertex_on_each_side_of_each_centerline_point(points):
    polygon = polyline_polygon_points(points, 6)

    assert len(polygon) == 2 * len(points)
    for center, left, right in zip(points, polygon[:len(points)], reversed(polygon[len(points):])):
        assert ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2) == pytest.approx(center)
        assert math.dist(left, right) == pytest.approx(6)


def test_straight_polygon_follows_both_sides_without_crossing_the_centerline():
    assert polyline_polygon_points([(10, 20), (20, 20), (30, 20)], 8) == [
        (10, 24), (20, 24), (30, 24), (30, 16), (20, 16), (10, 16),
    ]


def test_bend_uses_the_average_direction():
    polygon = polyline_polygon_points([(10, 10), (30, 10), (30, 50)], 8)
    offset = 4 / math.sqrt(2)
    assert polygon[1] == pytest.approx((30 - offset, 10 + offset))
    assert polygon[-2] == pytest.approx((30 + offset, 10 - offset))


def test_repeated_clicks_do_not_add_degenerate_edges():
    assert polyline_polygon_points([(10, 20), (10, 20), (30, 20)], 8) == [
        (10, 24), (30, 24), (30, 16), (10, 16),
    ]


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_invalid_width_is_rejected(width):
    with pytest.raises(ValueError, match="width"):
        polyline_polygon_points([(10, 20), (30, 20)], width)


@pytest.mark.parametrize("points", [[], [(10, 20)], [(10, 20), (10, 20)]])
def test_centerline_needs_two_distinct_points(points):
    with pytest.raises(ValueError, match="distinct points"):
        polyline_polygon_points(points, 8)


@pytest.fixture
def editor(tmp_path, monkeypatch):
    app_module = pytest.importorskip("labelme.app")
    from PyQt5 import QtCore, QtGui, QtWidgets

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("QT_QPA_PLATFORM_PLUGIN_PATH", raising=False)
    monkeypatch.delenv("QT_QPA_FONTDIR", raising=False)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Keep window preferences isolated from the user's annotation session.
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    monkeypatch.setattr(app_module.QtCore, "QSettings", lambda *args: settings)
    image = QtGui.QImage(100, 100, QtGui.QImage.Format_RGB32)
    image.fill(QtCore.Qt.white)
    image_path = tmp_path / "frame.png"
    image.save(str(image_path))
    window = create_main_window(app_module.MainWindow)(
        config_file=PACKAGE_DIR / "labelme.yaml", file_or_dir=str(image_path),
        config_overrides={"labels": ["track"]},
    )
    monkeypatch.setattr(window._label_dialog, "popup", lambda *args, **kwargs: ("track", {}, None, ""))
    yield window, image_path.with_suffix(".json")
    window.close()
    window.deleteLater()
    app.processEvents()


def finish_polyline(window, points):
    from PyQt5 import QtCore
    from labelme._shape import Shape

    canvas = window._canvas_widgets.canvas
    shape = Shape(shape_type="linestrip")
    for x, y in points:
        shape.add_point(QtCore.QPointF(x, y))
    canvas._current = shape
    canvas._finalize()
    return canvas


def test_polygon_tool_saves_polygon_and_keeps_undo_history(editor):
    window, annotation = editor
    action = dict(window._actions.draw)["polyline_polygon"]
    assert action.isEnabled()
    assert action in window._menus.edit.actions()
    action.trigger()
    assert not action.isEnabled()
    assert window._actions.create_line_strip_mode.isEnabled()
    canvas = finish_polyline(window, [(10, 20), (30, 20), (60, 40)])

    saved = json.loads(annotation.read_text())["shapes"]
    assert len(saved) == 1
    assert saved[0]["shape_type"] == "polygon"
    assert saved[0]["label"] == "track"
    assert len(saved[0]["points"]) == 6
    assert canvas.shape_backups[-1][-1].shape_type == "polygon"

    finish_polyline(window, [(10, 60), (30, 60)])
    assert len(canvas.shapes) == 2
    window.undo_shape_edit()
    assert len(canvas.shapes) == 1

    # An edit followed by undo restores the generated polygon, not its centerline.
    original = list(canvas.shapes[0].points)
    canvas.shapes[0].points[0] = original[0] + type(original[0])(2, 1)
    canvas.backup_shapes()
    window.undo_shape_edit()
    assert canvas.shapes[0].shape_type == "polygon"
    assert canvas.shapes[0].points == original


def test_width_setting_controls_new_polygon_and_clips_to_image(editor, monkeypatch):
    from PyQt5 import QtWidgets

    window, annotation = editor
    monkeypatch.setattr(QtWidgets.QInputDialog, "getDouble", lambda *args: (12.0, True))
    window._set_polygon_width()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getDouble", lambda *args: (99.0, False))
    window._set_polygon_width()
    window._switch_canvas_mode(edit=False, create_mode="polyline_polygon")
    finish_polyline(window, [(10, 3), (90, 3)])
    assert json.loads(annotation.read_text())["shapes"][0]["points"] == [
        [10, 9], [90, 9], [90, 0], [10, 0],
    ]


def test_cancelled_label_resumes_centerline_and_can_finish_again(editor, monkeypatch):
    window, annotation = editor
    window._switch_canvas_mode(edit=False, create_mode="polyline_polygon")
    monkeypatch.setattr(window._label_dialog, "popup", lambda *args: (None, {}, None, ""))
    canvas = finish_polyline(window, [(10, 20), (30, 20)])
    assert not annotation.exists()
    assert canvas.shapes == []
    assert canvas._current.shape_type == "linestrip"
    assert [(p.x(), p.y()) for p in canvas._current.points] == [(10, 20), (30, 20)]
    assert canvas._line.points == [canvas._current.points[-1], canvas._current.points[0]]

    monkeypatch.setattr(window._label_dialog, "popup", lambda *args: ("track", {}, None, ""))
    canvas._finalize()
    assert json.loads(annotation.read_text())["shapes"][0]["shape_type"] == "polygon"


def test_switching_back_to_normal_polyline_preserves_linestrip(editor):
    window, annotation = editor
    window._switch_canvas_mode(edit=False, create_mode="polyline_polygon")
    window._actions.create_line_strip_mode.trigger()
    finish_polyline(window, [(10, 20), (30, 20)])
    assert json.loads(annotation.read_text())["shapes"][0]["shape_type"] == "linestrip"
    window.close_file()
    assert not dict(window._actions.draw)["polyline_polygon"].isEnabled()


@pytest.mark.parametrize("provider,title", [("custom", "Custom Annotate"), ("yolo26x", "YOLO26x Annotate")])
def test_auto_annotation_button_appends_saves_and_undoes_predictions(editor, provider, title):
    from PyQt5 import QtCore, QtWidgets

    window, annotation = editor
    window._switch_canvas_mode(edit=False, create_mode="polyline_polygon")
    canvas = finish_polyline(window, [(10, 20), (30, 20)])
    original = json.loads(annotation.read_text())["shapes"]
    predicted_label = "person" if provider == "yolo26x" else "track"

    def predict(image, labels):
        assert QtCore.QThread.currentThread() != window.thread()
        assert image.mode == "RGB" and image.size == (100, 100)
        assert labels == ["track"]
        return [{"label": predicted_label, "points": [[10, 10], [90, 10], [50, 90]]}]

    annotator_attribute = "_backend_annotator" if provider == "custom" else "_yolo26x_annotator"
    setattr(window, annotator_attribute, SimpleNamespace(predict=predict, model_name="test-model"))
    action = getattr(window, f"_{provider}_annotate_action")
    assert action.text() == title
    assert action.isEnabled()
    assert action in window._menus.edit.actions()
    assert any(action in toolbar.actions() for toolbar in window.findChildren(QtWidgets.QToolBar))
    action.trigger()

    saved = json.loads(annotation.read_text())["shapes"]
    assert saved[:1] == original
    assert saved[1]["label"] == predicted_label
    assert saved[1]["shape_type"] == "polygon"
    assert saved[1]["points"] == [[10, 10], [90, 10], [50, 90]]
    assert len(canvas.shapes) == 2
    assert window._actions.undo.isEnabled()
    window.undo_shape_edit()
    assert len(canvas.shapes) == 1
    assert canvas.shapes[0].points == canvas.shape_backups[-1][0].points
    window.close_file()
    assert not action.isEnabled()


def test_yolo26x_predictions_with_foreign_labels_can_be_manually_relabeled(editor):
    window, annotation = editor
    predictions = [
        {"label": "person", "points": [[10, 10], [40, 10], [25, 40]]},
        {"label": "car", "points": [[50, 50], [90, 50], [70, 90]]},
    ]
    window._yolo26x_annotator = SimpleNamespace(
        predict=MagicMock(return_value=predictions), model_name="YOLO26x",
    )

    window._yolo26x_annotate_action.trigger()

    saved = json.loads(annotation.read_text())["shapes"]
    assert [shape["label"] for shape in saved] == ["person", "car"]
    assert window._config["labels"] == ["track"]
    assert window._docks.unique_label_list.find_label_item("person") is not None
    assert window._docks.unique_label_list.find_label_item("car") is not None

    canvas = window._canvas_widgets.canvas
    canvas.select_shapes([canvas.shapes[0]])
    window._edit_label()

    relabeled = json.loads(annotation.read_text())["shapes"]
    assert [shape["label"] for shape in relabeled] == ["track", "car"]
    assert [shape["points"] for shape in relabeled] == [shape["points"] for shape in saved]


@pytest.mark.parametrize("failure", [None, RuntimeError("Backend unavailable")])
@pytest.mark.parametrize("provider,title", [("custom", "Custom Annotate"), ("yolo26x", "YOLO26x Annotate")])
def test_empty_or_failed_prediction_preserves_annotations(editor, monkeypatch, failure, provider, title):
    from PyQt5 import QtWidgets

    window, annotation = editor
    window._switch_canvas_mode(edit=False, create_mode="polyline_polygon")
    canvas = finish_polyline(window, [(10, 20), (30, 20)])
    saved = annotation.read_bytes()
    backups = len(canvas.shape_backups)
    warning = MagicMock()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", warning)
    annotator_attribute = "_backend_annotator" if provider == "custom" else "_yolo26x_annotator"
    setattr(window, annotator_attribute, SimpleNamespace(
        predict=MagicMock(return_value=[], side_effect=failure), model_name="test-model",
    ))
    getattr(window, f"_{provider}_annotate_action").trigger()

    assert annotation.read_bytes() == saved
    assert len(canvas.shape_backups) == backups
    assert len(canvas.shapes) == 1
    if failure:
        warning.assert_called_once_with(window, f"{title} Failed", "Backend unavailable")
    else:
        warning.assert_not_called()


@pytest.mark.parametrize("provider", ["custom", "yolo26x"])
def test_auto_annotation_is_disabled_while_drawing(editor, provider):
    window, _ = editor
    action = getattr(window, f"_{provider}_annotate_action")
    window._on_drawing_polygon_changed(True)
    assert not action.isEnabled()
    window._on_drawing_polygon_changed(False)
    assert action.isEnabled()
    window.close_file()
    window._on_drawing_polygon_changed(False)
    assert not action.isEnabled()
