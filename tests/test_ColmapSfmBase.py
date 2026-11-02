"""
Tests of ColmapSfmBase.py (shared base of the COLMAP conversion nodes).

Run from the colmap folder (no Meshroom needed): python -m pytest tests -v
Shared fixtures (mock models, writers, fake Meshroom API) are in conftest.py.
"""

import copy
import io
import json
import logging
import math
import os
import struct
from types import SimpleNamespace
from unittest import mock

import pytest

from ColmapSfmBase import (Camera, CameraConverter, ColmapConversionNode, ColmapModelReader, Image,
                           LandmarkFactory, PoseConverter, SfmJson, SfmViewsConverter, SourceViews)

SQRT_HALF = math.sqrt(0.5)


# ColmapModelReader

class TestReadNextBytes:

    def test_little_endian(self):
        fid = io.BytesIO(b"\x01\x00\x00\x00\x02\x00")
        assert ColmapModelReader.read_next_bytes(fid, 4, "I") == (1,)
        assert ColmapModelReader.read_next_bytes(fid, 2, "H") == (2,)

    def test_big_endian(self):
        assert ColmapModelReader.read_next_bytes(io.BytesIO(b"\x00\x01"), 2, "H", endian_character=">") == (1,)


class TestBinaryReaders:

    def test_cameras(self, tmp_path, colmap_files, sparse_scene):
        colmap_files.write_cameras_binary(tmp_path / "cameras.bin", sparse_scene.cameras)
        assert ColmapModelReader.read_cameras_binary(tmp_path / "cameras.bin") == \
            {c.id: c for c in sparse_scene.cameras}

    def test_every_known_camera_model(self, tmp_path, colmap_files):
        cameras = [Camera(model_id, name, 10, 20, [float(i) for i in range(num_params)])
                   for model_id, (name, num_params) in ColmapModelReader.CAMERA_MODELS.items()]
        colmap_files.write_cameras_binary(tmp_path / "cameras.bin", cameras)
        assert list(ColmapModelReader.read_cameras_binary(tmp_path / "cameras.bin").values()) == cameras

    def test_unknown_camera_model_raises(self, tmp_path):
        (tmp_path / "cameras.bin").write_bytes(struct.pack("<QIiQQ", 1, 1, 42, 10, 10))
        with pytest.raises(RuntimeError, match="Unknown COLMAP camera model id: 42"):
            ColmapModelReader.read_cameras_binary(tmp_path / "cameras.bin")

    def test_images(self, tmp_path, colmap_files, sparse_scene):
        colmap_files.write_images_binary(tmp_path / "images.bin", sparse_scene.images)
        assert ColmapModelReader.read_images_binary(tmp_path / "images.bin") == \
            {i.id: i for i in sparse_scene.images}

    def test_images_keep_file_order(self, tmp_path, colmap_files, sparse_scene):
        colmap_files.write_images_binary(tmp_path / "images.bin", sparse_scene.images[::-1])
        assert list(ColmapModelReader.read_images_binary(tmp_path / "images.bin")) == [5, 4, 3, 2, 1]

    def test_image_name_with_spaces_and_unicode(self, tmp_path, colmap_files):
        image = Image(1, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1, "my folder/été 1.jpg", [], [])
        colmap_files.write_images_binary(tmp_path / "images.bin", [image])
        assert ColmapModelReader.read_images_binary(tmp_path / "images.bin")[1].name == "my folder/été 1.jpg"

    def test_points3D(self, tmp_path, colmap_files, sparse_scene):
        colmap_files.write_points3D_binary(tmp_path / "points3D.bin", sparse_scene.points3D)
        assert ColmapModelReader.read_points3D_binary(tmp_path / "points3D.bin") == \
            {p.id: p for p in sparse_scene.points3D}


class TestTextReaders:

    def test_cameras(self, tmp_path, colmap_files, sparse_scene):
        colmap_files.write_cameras_text(tmp_path / "cameras.txt", sparse_scene.cameras)
        assert ColmapModelReader.read_cameras_text(tmp_path / "cameras.txt") == \
            {c.id: c for c in sparse_scene.cameras}

    def test_images(self, tmp_path, colmap_files, sparse_scene):
        colmap_files.write_images_text(tmp_path / "images.txt", sparse_scene.images)
        assert ColmapModelReader.read_images_text(tmp_path / "images.txt") == \
            {i.id: i for i in sparse_scene.images}

    def test_image_without_points_has_empty_second_line(self, tmp_path, colmap_files, sparse_scene):
        # image 5 has no 2D point: its second line is empty and must not be read as the next image
        colmap_files.write_images_text(tmp_path / "images.txt", [sparse_scene.images[4], sparse_scene.images[0]])
        images = ColmapModelReader.read_images_text(tmp_path / "images.txt")
        assert list(images) == [5, 1]
        assert images[5].xys == []
        assert images[1].xys == sparse_scene.images[0].xys

    def test_points3D(self, tmp_path, colmap_files, sparse_scene):
        colmap_files.write_points3D_text(tmp_path / "points3D.txt", sparse_scene.points3D)
        assert ColmapModelReader.read_points3D_text(tmp_path / "points3D.txt") == \
            {p.id: p for p in sparse_scene.points3D}

    def test_comments_and_blank_lines_are_skipped(self, tmp_path):
        (tmp_path / "cameras.txt").write_text("# comment\n\n1 PINHOLE 4 3 1 2 3 4\n\n# another\n")
        assert ColmapModelReader.read_cameras_text(tmp_path / "cameras.txt") == {
            1: Camera(1, "PINHOLE", 4, 3, [1.0, 2.0, 3.0, 4.0])}


class TestFindModelFolder:

    @pytest.mark.parametrize("sub", ["", "0", os.path.join("sparse", "0"), "sparse"])
    def test_layouts(self, tmp_path, make_model, sparse_scene, sub):
        make_model(os.path.join("project", sub), sparse_scene.cameras, sparse_scene.images)
        root = str(tmp_path / "project")
        assert ColmapModelReader.find_model_folder(root) == (os.path.join(root, sub) if sub else root, ".bin")

    def test_text_model(self, make_model, sparse_scene):
        folder = make_model("model", sparse_scene.cameras, sparse_scene.images, ext=".txt")
        assert ColmapModelReader.find_model_folder(folder) == (folder, ".txt")

    def test_binary_preferred_over_text(self, make_model, sparse_scene):
        make_model("model", sparse_scene.cameras, sparse_scene.images, ext=".txt")
        folder = make_model("model", sparse_scene.cameras, sparse_scene.images, ext=".bin")
        assert ColmapModelReader.find_model_folder(folder)[1] == ".bin"

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="No COLMAP model"):
            ColmapModelReader.find_model_folder(str(tmp_path))


class TestReadModel:

    def test_binary_and_text_give_the_same_model(self, make_model, sparse_scene):
        scene = (sparse_scene.cameras, sparse_scene.images, sparse_scene.points3D)
        assert ColmapModelReader.read(make_model("bin", *scene, ext=".bin")) == \
            ColmapModelReader.read(make_model("txt", *scene, ext=".txt"))

    def test_read_points_false(self, sparse_model):
        cameras, images, points3D = ColmapModelReader.read(sparse_model, read_points=False)
        assert (len(cameras), len(images), points3D) == (3, 5, {})

    def test_missing_points_file(self, make_model, sparse_scene):
        assert ColmapModelReader.read(make_model("model", sparse_scene.cameras, sparse_scene.images))[2] == {}


# SfmJson

class TestSfmJson:

    SFM_DATA = {"version": ["1", "2", "12"], "views": [{"viewId": "1"}], "intrinsics": [], "poses": []}

    @pytest.mark.parametrize("value, expected", [(True, "true"), (False, "false"), (12, "12"), (1.5, "1.5"),
                                                 ("x", "x")])
    def test_value(self, value, expected):
        assert SfmJson.value(value) == expected

    def test_load(self, write_json):
        assert SfmJson.load(write_json("in.sfm", self.SFM_DATA)) == self.SFM_DATA

    def test_write(self, tmp_path, read_json):
        path = str(tmp_path / "out.sfm")
        SfmJson.write(path, self.SFM_DATA)
        assert read_json(path) == self.SFM_DATA
        with open(path) as fid:
            assert fid.read() == json.dumps(self.SFM_DATA, indent=4)

    def test_write_streamed(self, tmp_path, read_json):
        landmarks = [{"landmarkId": str(i), "X": ["0", "0", str(i)], "observations": []} for i in range(3)]
        path = str(tmp_path / "out.sfm")
        assert SfmJson.write_streamed(path, self.SFM_DATA, iter(landmarks)) == 3
        assert read_json(path) == dict(self.SFM_DATA, structure=landmarks)
        # one landmark per line
        with open(path) as fid:
            assert sum(1 for line in fid if '"landmarkId"' in line) == 3

    def test_write_streamed_without_landmarks(self, tmp_path, read_json):
        path = str(tmp_path / "out.sfm")
        assert SfmJson.write_streamed(path, self.SFM_DATA, iter([])) == 0
        assert read_json(path) == dict(self.SFM_DATA, structure=[])

    def test_write_streamed_does_not_modify_sfm_data(self, tmp_path):
        sfm_data = copy.deepcopy(self.SFM_DATA)
        SfmJson.write_streamed(str(tmp_path / "out.sfm"), sfm_data, iter([]))
        assert sfm_data == self.SFM_DATA


# CameraConverter

class TestCameraConverterToPinhole:

    @pytest.mark.parametrize("model, params, expected", [
        ("SIMPLE_PINHOLE", [1, 2, 3], (1, 1, 2, 3, "none", [])),
        ("PINHOLE", [1, 2, 3, 4], (1, 2, 3, 4, "none", [])),
        ("SIMPLE_RADIAL", [1, 2, 3, 4], (1, 1, 2, 3, "radialk1", [4])),
        ("RADIAL", [1, 2, 3, 4, 5], (1, 1, 2, 3, "radialk3", [4, 5, 0.0])),
        ("OPENCV", [1, 2, 3, 4, 5, 6, 7, 8], (1, 2, 3, 4, "brown", [5, 6, 0.0, 7, 8])),
        ("FULL_OPENCV", [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0], (1, 2, 3, 4, "brown", [5, 6, 9, 7, 8])),
        ("OPENCV_FISHEYE", [1, 2, 3, 4, 5, 6, 7, 8], (1, 2, 3, 4, "fisheye", [5, 6, 7, 8])),
        ("SIMPLE_RADIAL_FISHEYE", [1, 2, 3, 4], (1, 1, 2, 3, "fisheye", [4, 0.0, 0.0, 0.0])),
        ("RADIAL_FISHEYE", [1, 2, 3, 4, 5], (1, 1, 2, 3, "fisheye", [4, 5, 0.0, 0.0])),
        ("SIMPLE_FISHEYE", [1, 2, 3], (1, 1, 2, 3, "fisheye", [0.0, 0.0, 0.0, 0.0])),
        ("FISHEYE", [1, 2, 3, 4], (1, 2, 3, 4, "fisheye", [0.0, 0.0, 0.0, 0.0])),
        ("FOV", [1, 2, 3, 4, 5], (1, 2, 3, 4, "fisheye1", [5])),
    ])
    def test_supported_models(self, model, params, expected):
        assert CameraConverter.to_pinhole(Camera(1, model, 10, 10, params)) == expected

    def test_full_opencv_with_rational_distortion_raises(self):
        with pytest.raises(RuntimeError, match="rational distortion"):
            CameraConverter.to_pinhole(Camera(1, "FULL_OPENCV", 10, 10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0.1, 0, 0]))

    @pytest.mark.parametrize("model", ["THIN_PRISM_FISHEYE", "RAD_TAN_THIN_PRISM_FISHEYE", "SIMPLE_DIVISION",
                                       "DIVISION", "EUCM", "EQUIRECTANGULAR"])
    def test_unsupported_models_raise(self, model):
        with pytest.raises(RuntimeError, match="is not supported"):
            CameraConverter.to_pinhole(Camera(1, model, 10, 10, [0.0] * 16))


class TestCameraConverterIntrinsic:

    def test_default_sensor(self, sparse_scene):
        intrinsic = CameraConverter.intrinsic(1, sparse_scene.cameras[0], (1, 2, 12))
        assert intrinsic["intrinsicId"] == "1"
        assert (intrinsic["width"], intrinsic["height"]) == ("640", "480")
        assert intrinsic["type"] == "pinhole"
        assert float(intrinsic["sensorWidth"]) == pytest.approx(36.0)
        assert float(intrinsic["sensorHeight"]) == pytest.approx(27.0)
        assert float(intrinsic["focalLength"]) == pytest.approx(500.0 * 36.0 / 640.0)
        assert float(intrinsic["pixelRatio"]) == pytest.approx(1.0)
        assert [float(v) for v in intrinsic["principalPoint"]] == [0.0, 0.0]
        assert intrinsic["distortionType"] == "radialk1"
        assert [float(v) for v in intrinsic["distortionParams"]] == [0.01]
        assert intrinsic["undistortionType"] == "none"
        assert (intrinsic["serialNumber"], intrinsic["initialFocalLength"]) == ("-1", "-1")
        # every value is a string or a list, like AliceVision's json
        assert all(isinstance(value, (str, list)) for value in intrinsic.values())

    @pytest.mark.parametrize("version, focal_px", [((1, 2, 11), 900.0), ((1, 2, 12), 900.0), ((1, 2, 10), 800.0)])
    def test_focal_along_y_from_version_1_2_11(self, sparse_scene, version, focal_px):
        intrinsic = CameraConverter.intrinsic(2, sparse_scene.cameras[1], version)  # fx = 800, fy = 900, width 800
        assert float(intrinsic["focalLength"]) == pytest.approx(focal_px * 36.0 / 800.0)

    def test_pixel_ratio_and_principal_point(self, sparse_scene):
        intrinsic = CameraConverter.intrinsic(2, sparse_scene.cameras[1], (1, 2, 12))
        assert float(intrinsic["pixelRatio"]) == pytest.approx(900.0 / 800.0)
        assert [float(v) for v in intrinsic["principalPoint"]] == [10.0, -10.0]

    def test_portrait_image_uses_largest_dimension(self):
        intrinsic = CameraConverter.intrinsic(1, Camera(1, "PINHOLE", 480, 640, [500.0, 500.0, 240.0, 320.0]),
                                              (1, 2, 12))
        assert float(intrinsic["focalLength"]) == pytest.approx(500.0 * 36.0 / 640.0)
        assert float(intrinsic["sensorHeight"]) == pytest.approx(27.0)

    def test_template_sensor_infos(self, sparse_scene, sparse_source_sfm):
        intrinsic = CameraConverter.intrinsic(1, sparse_scene.cameras[0], (1, 2, 12),
                                              sparse_source_sfm()["intrinsics"][0])
        assert float(intrinsic["sensorWidth"]) == pytest.approx(6.4)
        assert float(intrinsic["sensorHeight"]) == pytest.approx(4.8)
        assert float(intrinsic["focalLength"]) == pytest.approx(5.0)
        assert (intrinsic["serialNumber"], intrinsic["initialFocalLength"]) == ("SN7", "4.5")
        assert intrinsic["pixelRatioLocked"] == "false"

    def test_template_without_sensor_height(self, sparse_scene):
        intrinsic = CameraConverter.intrinsic(1, sparse_scene.cameras[0], (1, 2, 12), {"sensorWidth": "8"})
        assert float(intrinsic["sensorHeight"]) == pytest.approx(6.0)


# PoseConverter

class TestPoseConverter:

    @pytest.mark.parametrize("qvec, expected", [
        ([1, 0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ([2, 0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # normalized
        ([SQRT_HALF, 0, 0, SQRT_HALF], [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90 degrees around z
        ([0, 0, 1, 0], [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 180 degrees around y
    ])
    def test_rotation_matrix(self, qvec, expected):
        assert PoseConverter.rotation_matrix(qvec) == [pytest.approx(row) for row in expected]

    def test_rotation_is_orthonormal(self):
        r = PoseConverter.rotation_matrix([0.3, -0.5, 0.7, 0.1])
        for i in range(3):
            for j in range(3):
                assert sum(r[k][i] * r[k][j] for k in range(3)) == pytest.approx(1.0 if i == j else 0.0)

    def test_pose(self, sparse_scene):
        pose = PoseConverter.pose("2", sparse_scene.images[1])  # R = 90 degrees around z, t = (1, 2, 3)
        assert pose["poseId"] == "2"
        transform = pose["pose"]["transform"]
        # column-major R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        assert [float(v) for v in transform["rotation"]] == pytest.approx([0, 1, 0, -1, 0, 0, 0, 0, 1])
        # C = -R^T t
        assert [float(v) for v in transform["center"]] == pytest.approx([-2.0, 1.0, -3.0])
        assert (pose["pose"]["locked"], pose["pose"]["rotationOnly"], pose["pose"]["removable"]) == \
            ("false", "false", "false")

    def test_pose_center_maps_to_camera_origin(self):
        image = Image(1, [0.3, -0.5, 0.7, 0.1], [1.5, -2.0, 0.25], 1, "x.jpg", [], [])
        r = PoseConverter.rotation_matrix(image.qvec)
        center = [float(c) for c in PoseConverter.pose("1", image)["pose"]["transform"]["center"]]
        assert [sum(r[row][k] * center[k] for k in range(3)) + image.tvec[row] for row in range(3)] == \
            pytest.approx([0.0, 0.0, 0.0])

    def test_is_valid(self, sparse_scene):
        image = sparse_scene.images[0]
        assert PoseConverter.is_valid(image)
        assert not PoseConverter.is_valid(sparse_scene.images[4])  # null quaternion
        assert not PoseConverter.is_valid(image._replace(tvec=[0.0, float("nan"), 0.0]))
        assert not PoseConverter.is_valid(image._replace(qvec=[float("inf"), 0.0, 0.0, 0.0]))


# SourceViews

class TestSourceViews:

    @pytest.fixture
    def views(self, camera_init):
        return SourceViews(camera_init.sfm([
            camera_init.view("1", "/data/a/IMG_1.jpg"),
            camera_init.view("2", "/data/x/IMG_2.jpg"),
            camera_init.view("3", "/data/y/IMG_2.jpg"),
            camera_init.view("4", "C:\\data\\win\\IMG_4.jpg"),
        ]))

    @pytest.mark.parametrize("name, view_id", [
        ("IMG_1.jpg", "1"),
        ("a/IMG_1.jpg", "1"),
        ("project/data/a/IMG_1.jpg", "1"),  # longer than the view path
        ("x/IMG_2.jpg", "2"),
        ("y/IMG_2.jpg", "3"),
        ("win/IMG_4.jpg", "4"),  # Windows separators in the view path
        ("b/IMG_1.jpg", "1"),  # unknown folder: falls back to the unique file name
    ])
    def test_found(self, views, name, view_id):
        assert views.find(name)["viewId"] == view_id

    @pytest.mark.parametrize("name", ["IMG_2.jpg", "IMG_9.jpg", "z/IMG_2.jpg"])
    def test_ambiguous_or_not_found(self, views, name):
        assert views.find(name) is None

    def test_empty_sfm(self):
        assert SourceViews({}).find("IMG_1.jpg") is None


# SfmViewsConverter

class TestSfmViewsConverter:

    @pytest.fixture
    def model(self, sparse_scene):
        return {c.id: c for c in sparse_scene.cameras}, {i.id: i for i in sparse_scene.images}

    def test_parse_version(self):
        assert SfmViewsConverter.parse_version(["1", "2", "12"]) == (1, 2, 12)
        assert SfmViewsConverter.parse_version(["1", "2", "9"]) < SfmViewsConverter.parse_version(["1", "2", "10"])

    def test_image_folder(self, model, caplog):
        caplog.set_level(logging.WARNING)
        converter = SfmViewsConverter(*model, image_folder="imgs")
        sfm_data = converter.convert()
        assert sfm_data["version"] == SfmViewsConverter.DEFAULT_SFM_VERSION
        assert set(sfm_data) == {"version", "views", "intrinsics", "poses"}
        assert converter.view_ids == {1: "1", 2: "2", 3: "3"}
        assert [v["path"] for v in sfm_data["views"]] == [os.path.join("imgs", n) for n in
                                                          ("a/IMG_1.jpg", "a/IMG_2.jpg", "b/IMG_3.jpg")]
        assert [i["intrinsicId"] for i in sfm_data["intrinsics"]] == ["1", "2"]
        assert sfm_data["poses"][1] == PoseConverter.pose("2", model[1][2])
        assert "Image b/IMG_4.jpg uses unknown camera 99 (skipping)" in caplog.text
        assert "Image b/IMG_5.jpg has no valid pose (skipping)" in caplog.text

    def test_source_sfm(self, model, sparse_source_sfm, caplog):
        caplog.set_level(logging.WARNING)
        converter = SfmViewsConverter(*model, source_sfm_data=sparse_source_sfm())
        sfm_data = converter.convert()
        assert converter.view_ids == {1: "101", 2: "102", 3: "103"}
        view = sfm_data["views"][0]
        assert (view["viewId"], view["poseId"], view["intrinsicId"]) == ("101", "101", "1")
        assert not set(SfmViewsConverter.RIG_KEYS) & set(view)
        assert sfm_data["intrinsics"][0]["serialNumber"] == "SN7"
        assert "Image b/IMG_3.jpg is 4000x3000 in the input SfMData but 800x600 in COLMAP" in caplog.text

    def test_source_sfm_is_not_modified(self, model, sparse_source_sfm):
        source = sparse_source_sfm()
        expected = copy.deepcopy(source)
        SfmViewsConverter(*model, source_sfm_data=source).convert()
        assert source == expected

    def test_too_old_version_raises(self, model, sparse_source_sfm):
        with pytest.raises(RuntimeError, match=r"1\.2\.7 is too old"):
            SfmViewsConverter(*model, source_sfm_data=sparse_source_sfm(version=("1", "2", "7"))).convert()

    def test_convert_again_resets_view_ids(self, model):
        converter = SfmViewsConverter(*model)
        converter.convert()
        converter.images = {1: model[1][1]}
        converter.convert()
        assert converter.view_ids == {1: "1"}

    def test_logger(self, model, caplog):
        caplog.set_level(logging.WARNING)
        SfmViewsConverter(*model, logger=logging.getLogger("colmap_tests.views")).convert()
        assert {r.name for r in caplog.records} == {"colmap_tests.views"}


# LandmarkFactory

class TestLandmarkFactory:

    def test_observation(self):
        assert LandmarkFactory().observation("12", 3, 10.5, 20.25) == {
            "observationId": "12", "featureId": "3", "x": ["10.5", "20.25"], "scale": "0"}

    def test_landmark(self):
        observations = [{"observationId": "1"}, {"observationId": "2"}]
        assert LandmarkFactory().landmark(7, (1, 2, 3), (0.5, -1.0, 2.0), observations) == {
            "landmarkId": "7", "descType": "unknown", "color": ["1", "2", "3"], "X": ["0.5", "-1.0", "2.0"],
            "observations": observations}

    @pytest.mark.parametrize("count", [0, 1])
    def test_landmark_seen_by_too_few_views(self, count):
        assert LandmarkFactory().landmark(7, (1, 2, 3), (0.0, 0.0, 0.0), [{"observationId": "1"}] * count) is None

    def test_number_formats(self):
        factory = LandmarkFactory(format_pixel="{:.3f}".format, format_coordinate="{:.9g}".format)
        assert factory.observation("1", 0, 1.23456, 2.0)["x"] == ["1.235", "2.000"]
        landmark = factory.landmark(0, (0, 0, 0), (0.1, 1e-12, 123456789.5), [{}, {}])
        assert landmark["X"] == ["0.1", "1e-12", "123456790"]
        # colors and ids are not affected
        assert landmark["color"] == ["0", "0", "0"]


# ColmapConversionNode

class TestColmapConversionNode:

    class Node(ColmapConversionNode):
        def __init__(self, error=None):
            self.calls = []
            self.error = error

        def run_conversion(self, node, logger):
            self.calls.append((node, logger))
            if self.error:
                raise self.error

    @pytest.fixture
    def chunk(self):
        return SimpleNamespace(node=SimpleNamespace(verboseLevel=SimpleNamespace(value="debug")),
                               logManager=mock.Mock(), logger=logging.getLogger("colmap_tests.mixin"))

    def test_category(self):
        assert ColmapConversionNode.category == "Colmap"

    def test_process_chunk(self, chunk):
        node = self.Node()
        node.processChunk(chunk)
        assert node.calls == [(chunk.node, chunk.logger)]
        chunk.logManager.start.assert_called_once_with("debug")
        chunk.logManager.end.assert_called_once_with()

    def test_process_chunk_ends_log_on_error(self, chunk):
        with pytest.raises(RuntimeError, match="boom"):
            self.Node(RuntimeError("boom")).processChunk(chunk)
        chunk.logManager.end.assert_called_once_with()

    def test_run_conversion_must_be_implemented(self, chunk):
        with pytest.raises(NotImplementedError):
            ColmapConversionNode().processChunk(chunk)

    @pytest.mark.parametrize("name", ["sparse", "dense"])
    def test_shared_attributes(self, load_node, name):
        node_class = load_node(name, {"sparse": "ColmapSparse2MeshroomSfmConvertion",
                                      "dense": "ColmapDense2MeshroomSfmConvertion"}[name])
        assert node_class.category == "Colmap"
        verbose_level = node_class.inputs[-1]
        assert (verbose_level.name, verbose_level.value) == ("verboseLevel", "info")
        assert verbose_level.values == ["fatal", "error", "warning", "info", "debug", "trace"]
        assert [(o.name, o.value) for o in node_class.outputs] == [("outputSfm", "{nodeCacheFolder}/sfmdata.sfm")]
        # the mixin comes first (its processChunk and category win) and is not itself a desc.Node, which
        # Meshroom would register as a node of the base module
        mixin, node_base = node_class.__mro__[1], node_class.__mro__[2]
        assert (mixin.__name__, node_base.__name__) == ("ColmapConversionNode", "_FakeNode")
        assert not issubclass(mixin, node_base)
