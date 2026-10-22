import json
import pytest

from training.image_segmentation.labelme_position import load_position_context


def position_frame(folder, number, *, session="session-a", references=None):
    image = folder / f"frame_{number}.jpg"
    image.touch()
    references = references or [[number, 20, 30], [number, 40, 50]]
    image.with_suffix(".position.json").write_text(json.dumps({
        "session_id": session,
        "image": image.name,
        "references": references,
        "target": [[1, 2, 3], [1, 22, 23]],
        "driver_position": [value - delta for value, delta in zip(references[0], [1, 2, 3])],
    }))
    return image


def test_context_reads_100_frames_either_side_in_numeric_order(tmp_path):
    images = {i: position_frame(tmp_path, i) for i in range(230)}
    context = load_position_context(images[115])

    assert len(context.frames) == 201
    assert context.current_index == 100
    assert context.frames[0].image == images[15]
    assert context.frames[-1].image == images[215]
    current = context.frames[context.current_index]
    assert current.reference == (115, 20, 30)
    assert current.centerline == (115, 40, 50)
    assert current.driver == (114, 18, 27)
    assert context.skipped == 0


def test_context_stops_at_folder_edges_and_preserves_coordinate_precision(tmp_path):
    first = position_frame(tmp_path, 1, references=[[12345678.125, 20, 30], [12345678.125, 40, 50]])
    position_frame(tmp_path, 2)
    context = load_position_context(first, radius=1)
    assert len(context.frames) == 2
    assert context.current_index == 0
    assert context.frames[0].driver[0] == 12345677.125
    assert load_position_context(tmp_path / "frame_2.jpg").current_index == 1


def test_local_targets_are_not_subtracted_from_world_reference(tmp_path):
    image = position_frame(tmp_path, 1)
    path = image.with_suffix(".position.json")
    data = json.loads(path.read_text())
    data["target"] = [[-2, 1, 3], [-22, 1, 23]]
    path.write_text(json.dumps(data))
    assert load_position_context(image).frames[0].driver == (0, 18, 27)


def test_context_keeps_gaps_for_other_sessions_bad_files_and_missing_images(tmp_path):
    position_frame(tmp_path, 1, session="session-b")
    position_frame(tmp_path, 2).with_suffix(".position.json").write_text("{broken")
    current = position_frame(tmp_path, 3)
    position_frame(tmp_path, 4).unlink()
    last = position_frame(tmp_path, 5)
    other_video = tmp_path / "video_002"
    other_video.mkdir()
    position_frame(other_video, 6)

    context = load_position_context(current)
    assert context.frames[:2] == [None, None]
    assert context.frames[3] is None
    assert context.frames[4].image == last
    assert context.current_index == 2
    assert context.skipped == 3
    assert len(context.frames) == 5


@pytest.mark.parametrize("change", [
    {"references": [[1, 2, 3]]},
    {"driver_position": [1, 2, float("nan")]},
    {"references": [[1, True, 3], [4, 5, 6]]},
    {"session_id": ""},
    {"image": "another.jpg"},
])
def test_invalid_current_position_file_reports_error(tmp_path, change):
    image = position_frame(tmp_path, 1)
    path = image.with_suffix(".position.json")
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, **change}))
    with pytest.raises(ValueError):
        load_position_context(image)


def test_missing_current_position_file_is_not_replaced_with_neighbor(tmp_path):
    position_frame(tmp_path, 1)
    with pytest.raises(FileNotFoundError):
        load_position_context(tmp_path / "frame_2.jpg")
