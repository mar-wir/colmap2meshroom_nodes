"""
Tests of ColmapSparse2MeshroomSfmConvertions.py (COLMAP sparse model to Meshroom SfMData).

The COLMAP reader and the conversion of cameras, poses and views are tested with the shared base, in
test_ColmapSfmBase.py; these tests cover the sparse conversion end to end.

Run from the colmap folder (no Meshroom needed): python -m pytest tests -v
Shared fixtures (mock models, writers, fake Meshroom API) are in conftest.py.
"""

import copy
import logging
import os

import pytest

import ColmapSparse2MeshroomSfmConvertions as sparse
from ColmapSfmBase import PoseConverter, SfmViewsConverter


# Integration tests: model files to SfMData.

class TestColmap2Meshroom:

    @pytest.fixture
    def convert(self, sparse_model, caplog):
        """Run colmap2meshroom on the sparse model, capturing the logs in caplog."""
        caplog.set_level(logging.INFO)
        return lambda **kwargs: sparse.colmap2meshroom(sparse_model, **kwargs)

    def test_views_from_image_folder(self, convert, caplog, tmp_path):
        images = str(tmp_path / "images")
        sfm_data = convert(image_folder=images)
        assert sfm_data["version"] == SfmViewsConverter.DEFAULT_SFM_VERSION
        assert [v["viewId"] for v in sfm_data["views"]] == ["1", "2", "3"]
        assert sfm_data["views"][2] == {
            "viewId": "3", "frameId": "3", "path": os.path.join(images, "b/IMG_3.jpg"), "metadata": {},
            "poseId": "3", "intrinsicId": "2", "width": "800", "height": "600"}
        assert "Image b/IMG_4.jpg uses unknown camera 99 (skipping)" in caplog.text
        assert "Image b/IMG_5.jpg has no valid pose (skipping)" in caplog.text
        assert "COLMAP model: 3 cameras, 5 images, 4 points" in caplog.text

    def test_only_used_cameras_become_intrinsics(self, convert):
        assert [i["intrinsicId"] for i in convert(image_folder="images")["intrinsics"]] == ["1", "2"]

    def test_poses(self, convert, sparse_scene):
        poses = convert(image_folder="images")["poses"]
        assert [p["poseId"] for p in poses] == ["1", "2", "3"]
        assert poses[1] == PoseConverter.pose("2", sparse_scene.images[1])

    def test_landmarks(self, convert):
        structure = convert(image_folder="images")["structure"]
        assert [lm["landmarkId"] for lm in structure] == ["1", "3"]
        assert structure[0] == {
            "landmarkId": "1", "descType": "unknown", "color": ["10", "20", "30"], "X": ["0.1", "0.2", "0.3"],
            "observations": [
                {"observationId": "1", "featureId": "0", "x": ["10.5", "20.25"], "scale": "0"},
                {"observationId": "2", "featureId": "0", "x": ["11.0", "21.0"], "scale": "0"},
            ],
        }
        # duplicate observation in image 1: the first one is kept
        assert [(o["observationId"], o["featureId"], o["x"]) for o in structure[1]["observations"]] == \
            [("1", "1", ["30.0", "40.0"]), ("3", "0", ["70.0", "80.0"])]

    def test_without_landmarks(self, convert):
        assert "structure" not in convert(image_folder="images", include_landmarks=False)

    def test_text_model_gives_the_same_sfm(self, convert, make_model, sparse_scene):
        text_model = make_model("text", sparse_scene.cameras, sparse_scene.images, sparse_scene.points3D,
                                ext=".txt")
        assert convert(image_folder="images") == sparse.colmap2meshroom(text_model, image_folder="images")

    def test_source_sfm_views(self, convert, caplog, sparse_source_sfm):
        sfm_data = convert(source_sfm_data=sparse_source_sfm())
        assert sfm_data["version"] == ["1", "2", "10"]
        assert [v["viewId"] for v in sfm_data["views"]] == ["101", "102", "103"]
        view = sfm_data["views"][0]
        assert (view["path"], view["poseId"], view["intrinsicId"]) == ("/photos/a/IMG_1.jpg", "101", "1")
        assert view["metadata"] == {"Exif:FocalLength": "4.5", "Make": "TestCam"}
        assert not {"rigId", "subPoseId", "isPoseIndependant", "resectionId"} & set(view)
        # the COLMAP size wins over CameraInit's, with a warning
        assert (sfm_data["views"][2]["width"], sfm_data["views"][2]["height"]) == ("800", "600")
        assert "Image b/IMG_3.jpg is 4000x3000 in the input SfMData but 800x600 in COLMAP" in caplog.text
        assert "IMG_1.jpg is" not in caplog.text

    def test_source_sfm_is_not_modified(self, convert, sparse_source_sfm):
        source = sparse_source_sfm()
        expected = copy.deepcopy(source)
        convert(source_sfm_data=source)
        assert source == expected

    def test_source_sfm_intrinsics(self, convert, sparse_source_sfm):
        intrinsics = convert(source_sfm_data=sparse_source_sfm())["intrinsics"]
        assert intrinsics[0]["serialNumber"] == "SN7"
        # version 1.2.10: focal along x; fx = 500 for a 640 px wide image on a 6.4 mm sensor
        assert float(intrinsics[0]["focalLength"]) == pytest.approx(5.0)
        assert float(intrinsics[1]["focalLength"]) == pytest.approx(800.0 * 8.0 / 800.0)

    def test_source_sfm_landmarks_use_view_ids(self, convert, sparse_source_sfm):
        observations = convert(source_sfm_data=sparse_source_sfm())["structure"][0]["observations"]
        assert [o["observationId"] for o in observations] == ["101", "102"]

    def test_image_missing_from_source_sfm_is_skipped(self, convert, caplog, sparse_source_sfm):
        source = sparse_source_sfm()
        del source["views"][1]
        sfm_data = convert(source_sfm_data=source)
        assert [v["viewId"] for v in sfm_data["views"]] == ["101", "103"]
        assert "Image a/IMG_2.jpg not found in the input SfMData (skipping)" in caplog.text
        # point 1 is then seen by one exported view only
        assert [lm["landmarkId"] for lm in sfm_data["structure"]] == ["3"]

    def test_too_old_version_raises(self, convert, sparse_source_sfm):
        with pytest.raises(RuntimeError, match=r"1\.2\.7 is too old"):
            convert(source_sfm_data=sparse_source_sfm(version=("1", "2", "7")))

    def test_custom_logger(self, sparse_model, caplog):
        caplog.set_level(logging.INFO)
        sparse.colmap2meshroom(sparse_model, image_folder="images", logger=logging.getLogger("colmap_tests.custom"))
        assert {r.name for r in caplog.records} == {"colmap_tests.custom"}


class TestConvert:

    def test_requires_image_folder_or_sfm(self, sparse_model, tmp_path):
        with pytest.raises(RuntimeError, match="image folder or an SfMData"):
            sparse.convert(sparse_model, str(tmp_path / "out.sfm"))

    def test_writes_json(self, sparse_model, tmp_path, read_json):
        output = str(tmp_path / "out.sfm")
        sparse.convert(str(tmp_path), output, image_folder="images")  # parent folder: finds sparse/0
        assert read_json(output) == sparse.colmap2meshroom(sparse_model, "images")

    def test_reads_input_sfm_file(self, sparse_model, tmp_path, write_json, read_json, sparse_source_sfm):
        output = str(tmp_path / "out.sfm")
        sparse.convert(sparse_model, output, input_sfm=write_json("cameraInit.sfm", sparse_source_sfm()),
                       include_landmarks=False)
        sfm_data = read_json(output)
        assert [v["viewId"] for v in sfm_data["views"]] == ["101", "102", "103"]
        assert "structure" not in sfm_data


class TestSparseSfmConverter:

    def test_nothing_is_read_before_conversion(self, tmp_path):
        converter = sparse.SparseSfmConverter(str(tmp_path / "missing"), image_folder="images")
        with pytest.raises(RuntimeError, match="No COLMAP model"):
            converter.to_sfm_data()

    def test_from_files(self, sparse_model, write_json, sparse_source_sfm):
        converter = sparse.SparseSfmConverter.from_files(
            sparse_model, input_sfm=write_json("cameraInit.sfm", sparse_source_sfm()), include_landmarks=False)
        assert converter.source_sfm_data == sparse_source_sfm()
        assert not converter.include_landmarks

    def test_from_files_requires_image_folder_or_sfm(self, sparse_model):
        with pytest.raises(RuntimeError, match="image folder or an SfMData"):
            sparse.SparseSfmConverter.from_files(sparse_model)

    def test_to_sfm_data_and_write_agree(self, sparse_model, tmp_path, read_json):
        converter = sparse.SparseSfmConverter(sparse_model, image_folder="images")
        converter.write(str(tmp_path / "out.sfm"))
        assert read_json(str(tmp_path / "out.sfm")) == converter.to_sfm_data()


class TestCommandLine:

    def test_image_folder(self, sparse_model, tmp_path, run_script, read_json):
        result = run_script("sparse", tmp_path, "out.sfm", "--imageFolder", "images")
        assert result.returncode == 0, result.stderr
        assert "SfMData: 3 views, 2 intrinsics, 2 landmarks" in result.stderr
        assert len(read_json(str(tmp_path / "out.sfm"))["structure"]) == 2

    def test_input_sfm(self, sparse_model, tmp_path, run_script, read_json, write_json, sparse_source_sfm):
        result = run_script("sparse", sparse_model, "out.sfm", "--inputSfm",
                            write_json("cameraInit.sfm", sparse_source_sfm()))
        assert result.returncode == 0, result.stderr
        assert [v["viewId"] for v in read_json(str(tmp_path / "out.sfm"))["views"]] == ["101", "102", "103"]

    def test_no_landmarks_text_model(self, make_model, sparse_scene, tmp_path, run_script, read_json):
        model = make_model("model", sparse_scene.cameras, sparse_scene.images, sparse_scene.points3D, ext=".txt")
        result = run_script("sparse", model, "out.sfm", "--imageFolder", "images", "--noLandmarks")
        assert result.returncode == 0, result.stderr
        assert "structure" not in read_json(str(tmp_path / "out.sfm"))

    def test_missing_model_fails(self, tmp_path, run_script):
        result = run_script("sparse", tmp_path, "out.sfm", "--imageFolder", "images")
        assert result.returncode != 0
        assert "No COLMAP model" in result.stderr


# Integration tests: Meshroom node, with a fake Meshroom API.

@pytest.fixture(scope="module")
def node_class(load_node):
    """The ColmapSparse2MeshroomSfmConvertion node class, loaded with a fake Meshroom API."""
    return load_node("sparse", "ColmapSparse2MeshroomSfmConvertion")


class TestNode:

    def test_description(self, node_class):
        assert node_class.category == "Colmap"
        assert [a.name for a in node_class.inputs] == ["input", "inputSfm", "imageFolder", "includeLandmarks",
                                                       "verboseLevel"]
        assert [a.name for a in node_class.outputs] == ["outputSfm"]
        assert node_class.outputs[0].value == "{nodeCacheFolder}/sfmdata.sfm"

    def test_process_chunk(self, node_class, make_chunk, sparse_model, tmp_path, write_json, read_json,
                           sparse_source_sfm, caplog):
        caplog.set_level(logging.INFO)
        output = str(tmp_path / "out.sfm")
        chunk = make_chunk(node_class, input=sparse_model, inputSfm=write_json("cameraInit.sfm", sparse_source_sfm()),
                           outputSfm=output, verboseLevel="debug")
        node_class().processChunk(chunk)
        chunk.logManager.start.assert_called_once_with("debug")
        chunk.logManager.end.assert_called_once_with()
        assert any(r.name == chunk.logger.name for r in caplog.records)
        assert len(read_json(output)["views"]) == 3

    def test_process_chunk_ends_log_on_error(self, node_class, make_chunk, tmp_path):
        chunk = make_chunk(node_class, input=str(tmp_path), imageFolder="images", outputSfm=str(tmp_path / "o.sfm"))
        with pytest.raises(RuntimeError, match="No COLMAP model"):
            node_class().processChunk(chunk)
        chunk.logManager.end.assert_called_once_with()
