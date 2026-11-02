"""
Tests of ColmapDense2MeshroomSfmConvertions.py (COLMAP fused point cloud to Meshroom SfMData).

Run from the colmap folder (no Meshroom needed): python -m pytest tests -v
Shared fixtures (mock dense workspace, writers, fake Meshroom API) are in conftest.py.
"""

import io
import logging
import os
import shutil
import struct

import pytest

import ColmapDense2MeshroomSfmConvertions as dense
from ColmapSfmBase import Camera, Image

POINT_COUNT = 10


@pytest.fixture
def points(dense_scene):
    """The fused points of the default workspace, seen by images 1 and 2."""
    return dense_scene.points(POINT_COUNT)


@pytest.fixture
def workspace(make_dense_workspace, dense_scene, points):
    """Dense workspace whose fused.ply.vis uses the image id order (newer COLMAP versions)."""
    return make_dense_workspace(points, [dense_scene.vis([1, 2])] * len(points))


@pytest.fixture
def output(tmp_path):
    return str(tmp_path / "dense.sfm")


@pytest.fixture
def convert(output, read_json, caplog):
    """Run the conversion into the output file, capturing the logs in caplog.

    Returns:
        convert(folder, **kwargs) -> written SfMData.
    """
    caplog.set_level(logging.INFO)

    def _convert(folder, **kwargs):
        dense.convert(folder, output, **kwargs)
        return read_json(output)
    return _convert


def landmark_ids(sfm_data):
    return [lm["landmarkId"] for lm in sfm_data["structure"]]


# Unit tests: PLY reading.

class TestReadPlyHeader:

    def test_binary_header(self, tmp_path, colmap_files, points):
        colmap_files.write_ply(tmp_path / "p.ply", points[:3])
        with open(tmp_path / "p.ply", "rb") as fid:
            ply_format, elements = dense.PlyPointReader.read_header(fid)
            data = fid.read()
        assert ply_format == "binary_little_endian"
        assert elements == [("vertex", 3, [("float", "x"), ("float", "y"), ("float", "z"), ("float", "nx"),
                                           ("float", "ny"), ("float", "nz"), ("uchar", "red"),
                                           ("uchar", "green"), ("uchar", "blue")])]
        # positioned at the start of the data: 3 rows of 6 floats and 3 uchars
        assert len(data) == 3 * 27

    def test_several_elements_and_list_property(self):
        fid = io.BytesIO(b"ply\nformat ascii 1.0\ncomment made by test\nelement vertex 2\nproperty float x\n"
                         b"element face 1\nproperty list uchar int vertex_indices\nend_header\n")
        assert dense.PlyPointReader.read_header(fid) == ("ascii", [("vertex", 2, [("float", "x")]),
                                                        ("face", 1, [("list uchar int", "vertex_indices")])])

    def test_not_a_ply_file(self):
        fid = io.BytesIO(b"obj\n")
        fid.name = "mesh.obj"
        with pytest.raises(RuntimeError, match="Not a PLY file: mesh.obj"):
            dense.PlyPointReader.read_header(fid)

    def test_truncated_header(self):
        fid = io.BytesIO(b"ply\nformat ascii 1.0\nelement vertex 2\n")
        fid.name = "cut.ply"
        with pytest.raises(RuntimeError, match="Truncated PLY header: cut.ply"):
            dense.PlyPointReader.read_header(fid)


class TestVertexElement:

    def test_vertex_element(self):
        elements = [("vertex", 4, [("float", "x"), ("double", "y"), ("float32", "z"), ("uchar", "red")])]
        assert dense.PlyPointReader.vertex_element("p.ply", elements) == (4, ["x", "y", "z", "red"], "fdfB")

    @pytest.mark.parametrize("elements, message", [
        ([], "first element is not 'vertex'"),
        ([("face", 1, []), ("vertex", 1, [("float", "x"), ("float", "y"), ("float", "z")])],
         "first element is not 'vertex'"),
        ([("vertex", 1, [("float", "x"), ("float", "y"), ("float", "z"), ("list uchar int", "idx")])],
         "Unsupported PLY vertex property 'list uchar int idx'"),
        ([("vertex", 1, [("float", "x"), ("float", "y")])], "without x, y, z coordinates"),
    ])
    def test_invalid(self, elements, message):
        with pytest.raises(RuntimeError, match=message):
            dense.PlyPointReader.vertex_element("p.ply", elements)

    def test_ply_vertex_count(self, tmp_path, colmap_files, points):
        colmap_files.write_ply(tmp_path / "p.ply", points)
        assert dense.PlyPointReader.vertex_count(tmp_path / "p.ply") == POINT_COUNT


class TestReadPlyPoints:

    @pytest.mark.parametrize("ply_format", ["ascii", "binary_little_endian", "binary_big_endian"])
    def test_formats(self, tmp_path, colmap_files, points, ply_format):
        colmap_files.write_ply(tmp_path / "p.ply", points, ply_format=ply_format)
        assert list(dense.PlyPointReader.read_points(tmp_path / "p.ply")) == [pytest.approx(p) for p in points]

    def test_colors_are_ints(self, tmp_path, colmap_files, points):
        colmap_files.write_ply(tmp_path / "p.ply", points, ply_format="ascii")
        assert all(isinstance(c, int) for p in dense.PlyPointReader.read_points(tmp_path / "p.ply") for c in p[3:])

    def test_without_colors_and_normals(self, tmp_path, colmap_files, points):
        colmap_files.write_ply(tmp_path / "p.ply", points, normals=False, colors=False)
        assert list(dense.PlyPointReader.read_points(tmp_path / "p.ply")) == \
            [pytest.approx(tuple(p[:3]) + (255, 255, 255)) for p in points]

    def test_double_coordinates_in_any_order(self, tmp_path):
        header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty uchar blue\n"
                  b"property double z\nproperty double y\nproperty double x\nproperty uchar green\n"
                  b"property uchar red\nend_header\n")
        (tmp_path / "p.ply").write_bytes(header + struct.pack("<BdddBB", 3, 0.3, 0.2, 0.1, 2, 1))
        assert list(dense.PlyPointReader.read_points(tmp_path / "p.ply")) == [(0.1, 0.2, 0.3, 1, 2, 3)]

    def test_empty_cloud(self, tmp_path, colmap_files):
        colmap_files.write_ply(tmp_path / "p.ply", [])
        assert list(dense.PlyPointReader.read_points(tmp_path / "p.ply")) == []

    @pytest.mark.parametrize("ply_format", ["ascii", "binary_little_endian"])
    def test_truncated_data(self, tmp_path, colmap_files, points, ply_format):
        path = str(tmp_path / "p.ply")
        colmap_files.write_ply(path, points[:3], ply_format=ply_format, declared_count=5)
        with pytest.raises(RuntimeError, match="Truncated PLY file"):
            list(dense.PlyPointReader.read_points(path))

    def test_unsupported_format(self, tmp_path):
        (tmp_path / "p.ply").write_bytes(b"ply\nformat binary_middle_endian 1.0\nelement vertex 0\n"
                                         b"property float x\nproperty float y\nproperty float z\nend_header\n")
        with pytest.raises(RuntimeError, match="Unsupported PLY format 'binary_middle_endian'"):
            list(dense.PlyPointReader.read_points(tmp_path / "p.ply"))


class TestReadPlyVisibility:

    def test_visibility(self, tmp_path, colmap_files):
        visibility = [(0, 1), (2,), (), (5, 3, 1, 0)]
        colmap_files.write_visibility(tmp_path / "p.ply.vis", visibility)
        assert dense.FusedPointCloud.visibility_count(tmp_path / "p.ply.vis") == 4
        assert list(dense.FusedPointCloud.read_visibility(tmp_path / "p.ply.vis")) == visibility

    def test_empty(self, tmp_path, colmap_files):
        colmap_files.write_visibility(tmp_path / "p.ply.vis", [])
        assert dense.FusedPointCloud.visibility_count(tmp_path / "p.ply.vis") == 0
        assert list(dense.FusedPointCloud.read_visibility(tmp_path / "p.ply.vis")) == []


class TestFusedPointCloud:

    def test_default_visibility_path(self):
        assert dense.FusedPointCloud(os.path.join("dense", "fused.ply")).vis_path == \
            os.path.join("dense", "fused.ply.vis")

    def test_check_and_iterate(self, workspace, dense_scene, points):
        cloud = dense.FusedPointCloud(os.path.join(workspace, "fused.ply"))
        assert cloud.check() == POINT_COUNT
        assert [(tuple(p), v) for p, v in cloud] == [(tuple(p), dense_scene.vis([1, 2])) for p in points]

    @pytest.mark.parametrize("missing", ["fused.ply", "fused.ply.vis"])
    def test_check_missing_file(self, workspace, missing):
        os.remove(os.path.join(workspace, missing))
        with pytest.raises(RuntimeError, match="File not found: .*" + missing.replace(".", r"\.") + "$"):
            dense.FusedPointCloud(os.path.join(workspace, "fused.ply")).check()

    def test_check_count_mismatch(self, workspace, colmap_files):
        colmap_files.write_visibility(os.path.join(workspace, "fused.ply.vis"), [(0, 1)] * (POINT_COUNT - 1))
        with pytest.raises(RuntimeError, match="has 10 points but .* has 9"):
            dense.FusedPointCloud(os.path.join(workspace, "fused.ply")).check()


# Unit tests: projection.

class TestCameraProjection:

    def test_pinhole(self, dense_scene):
        projection = dense.CameraProjection(dense_scene.camera, dense_scene.images[2])
        assert projection.rotation == [pytest.approx(row) for row in [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]]
        assert (projection.translation, projection.fx, projection.fy, projection.cx, projection.cy,
                projection.width, projection.height) == ([0.0, 0.0, 5.0], 500.0, 500.0, 320.0, 240.0, 640, 480)

    @pytest.mark.parametrize("camera", [
        Camera(1, "SIMPLE_PINHOLE", 640, 480, [500.0, 320.0, 240.0]),
        Camera(1, "SIMPLE_RADIAL", 640, 480, [500.0, 320.0, 240.0, 0.0]),
        Camera(1, "OPENCV", 640, 480, [500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0]),
    ])
    def test_models_without_distortion(self, dense_scene, camera):
        projection = dense.CameraProjection(camera, dense_scene.images[1])
        assert (projection.fx, projection.fy, projection.cx, projection.cy) == (500.0, 500.0, 320.0, 240.0)

    @pytest.mark.parametrize("camera", [
        Camera(7, "SIMPLE_RADIAL", 640, 480, [500.0, 320.0, 240.0, 0.1]),
        Camera(7, "OPENCV", 640, 480, [500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.001, 0.0]),
    ])
    def test_distortion_raises(self, dense_scene, camera):
        with pytest.raises(RuntimeError, match="COLMAP camera 7 .* has distortion"):
            dense.CameraProjection(camera, dense_scene.images[1])


class TestProject:

    @pytest.fixture
    def projection(self, dense_scene):
        """Image 1: identity rotation, camera 5 units before the origin along z."""
        return dense.CameraProjection(dense_scene.camera, dense_scene.images[1])

    def test_principal_point(self, projection):
        assert projection.project((0.0, 0.0, 0.0)) == pytest.approx((320.0, 240.0))

    def test_matches_hand_computation(self, projection, dense_scene, points):
        for point in points:
            assert projection.project(point) == pytest.approx(dense_scene.project(1, point))

    def test_rotated_camera(self, dense_scene, points):
        projection = dense.CameraProjection(dense_scene.camera, dense_scene.images[2])
        for point in points:
            assert projection.project(point) == pytest.approx(dense_scene.project(2, point))

    def test_extra_values_are_ignored(self, projection):
        assert projection.project((0.0, 0.0, 0.0, 255, 0, 0)) == pytest.approx((320.0, 240.0))

    @pytest.mark.parametrize("point", [(0.0, 0.0, -5.0), (0.0, 0.0, -6.0)])
    def test_behind_or_on_camera_plane(self, projection, point):
        assert projection.project(point) is None

    # u = 100 x + 320, v = 100 y + 240 at z = 0
    @pytest.mark.parametrize("x, y, visible", [
        (-3.2, 0.0, True), (3.2, 0.0, True),  # on the image border
        (-3.205, 0.0, True), (3.205, 0.0, True), (0.0, 2.405, True),  # inside the 1 px margin
        (-3.215, 0.0, False), (3.215, 0.0, False), (0.0, -2.415, False), (0.0, 2.415, False),
    ])
    def test_image_bounds_and_margin(self, projection, x, y, visible):
        assert (projection.project((x, y, 0.0)) is not None) == visible


# Unit tests: fused.ply.vis image order.

class TestDetectImageOrder:

    @pytest.fixture
    def model(self, dense_scene):
        images = {image.id: image for image in dense_scene.images}  # file order 3, 1, 2
        projections = {i: dense.CameraProjection(dense_scene.camera, image) for i, image in images.items()}
        return images, projections

    def detect(self, model, workspace, points, logger):
        images, projections = model
        cloud = dense.FusedPointCloud(os.path.join(workspace, "fused.ply"))
        return dense.DenseSfmConverter.detect_image_order(images, projections, cloud, len(points), logger)

    def test_image_id_order(self, model, workspace, points, caplog):
        caplog.set_level(logging.INFO)
        assert self.detect(model, workspace, points, logging.getLogger("t")) == [1, 2, 3]
        # in file order, indices (0, 1) are images 3 (behind) and 1 (valid)
        assert "image id order (20 valid sample observations, 10 with the other order)" in caplog.text

    def test_file_order(self, model, make_dense_workspace, dense_scene, points, caplog):
        caplog.set_level(logging.INFO)
        workspace = make_dense_workspace(points, [dense_scene.vis([1, 2], dense_scene.file_order)] * len(points))
        assert self.detect(model, workspace, points, logging.getLogger("t")) == [3, 1, 2]
        assert "file order (20 valid sample observations, 10 with the other order)" in caplog.text

    def test_sorted_model_does_not_read_files(self, dense_scene):
        images = {1: dense_scene.images[1], 2: dense_scene.images[2]}
        assert dense.DenseSfmConverter.detect_image_order(images, {}, dense.FusedPointCloud("missing.ply"), 10,
                                                          logging.getLogger("t")) == [1, 2]

    def test_no_points(self, model):
        assert dense.DenseSfmConverter.detect_image_order(model[0], model[1], dense.FusedPointCloud("missing.ply"),
                                                          0, logging.getLogger("t")) == [3, 1, 2]

    def test_out_of_range_indices_score_nothing(self, model, make_dense_workspace, points):
        workspace = make_dense_workspace(points, [(7, 8)] * len(points))
        assert self.detect(model, workspace, points, logging.getLogger("t")) == [3, 1, 2]


# Integration tests: dense workspace to SfMData.

class TestConvert:

    def test_views_from_dense_images(self, convert, workspace):
        sfm_data = convert(workspace)
        assert [v["viewId"] for v in sfm_data["views"]] == ["1", "2", "3"]
        assert sfm_data["views"][0]["path"] == os.path.join(workspace, "images", "a.jpg")
        assert [i["intrinsicId"] for i in sfm_data["intrinsics"]] == ["1"]
        assert sfm_data["intrinsics"][0]["distortionType"] == "none"
        assert len(sfm_data["poses"]) == 3

    def test_landmarks(self, convert, workspace, dense_scene, points, caplog):
        structure = convert(workspace)["structure"]
        assert landmark_ids({"structure": structure}) == [str(i) for i in range(POINT_COUNT)]
        for i, (landmark, point) in enumerate(zip(structure, points)):
            assert landmark["descType"] == "unknown"
            assert landmark["color"] == [str(c) for c in point[3:]]
            assert [float(c) for c in landmark["X"]] == pytest.approx(point[:3])
            assert [o["observationId"] for o in landmark["observations"]] == ["1", "2"]
            for observation, image_id in zip(landmark["observations"], (1, 2)):
                assert observation["featureId"] == str(i)
                assert observation["scale"] == "0"
                assert [float(v) for v in observation["x"]] == pytest.approx(dense_scene.project(image_id, point),
                                                                             abs=1e-3)
        assert "SfMData: 3 views, 1 intrinsics, 10 landmarks (0 points seen by less than 2 views dropped)" in \
            caplog.text

    def test_file_order_visibility_gives_the_same_landmarks(self, convert, workspace, make_dense_workspace,
                                                            dense_scene, points):
        expected = convert(workspace)["structure"]
        other = make_dense_workspace(points, [dense_scene.vis([1, 2], dense_scene.file_order)] * len(points),
                                     folder="dense_file_order")
        assert convert(other)["structure"] == expected

    @pytest.mark.parametrize("ply_format", ["ascii", "binary_big_endian"])
    def test_ply_formats(self, convert, workspace, make_dense_workspace, dense_scene, points, ply_format):
        expected = convert(workspace)["structure"]
        other = make_dense_workspace(points, [dense_scene.vis([1, 2])] * len(points), ply_format=ply_format,
                                     folder="dense_" + ply_format)
        assert convert(other)["structure"] == expected

    @pytest.mark.parametrize("max_points, expected_ids", [
        (3, ["0", "4", "8"]),  # keeps 1 point out of ceil(10 / 3) = 4
        (5, ["0", "2", "4", "6", "8"]),
        (0, [str(i) for i in range(POINT_COUNT)]),
        (POINT_COUNT, [str(i) for i in range(POINT_COUNT)]),
        (1000, [str(i) for i in range(POINT_COUNT)]),
    ])
    def test_max_points(self, convert, workspace, max_points, expected_ids):
        assert landmark_ids(convert(workspace, max_points=max_points)) == expected_ids

    def test_points_seen_by_one_view_are_dropped(self, convert, make_dense_workspace, dense_scene, points,
                                                 caplog):
        visibility = [dense_scene.vis([1, 2])] * len(points)
        visibility[3] = dense_scene.vis([1])
        visibility[5] = ()
        assert landmark_ids(convert(make_dense_workspace(points, visibility))) == \
            ["0", "1", "2", "4", "6", "7", "8", "9"]
        assert "8 landmarks (2 points seen by less than 2 views dropped)" in caplog.text

    def test_observations_behind_the_camera_are_invalid(self, convert, make_dense_workspace, dense_scene, points,
                                                        caplog):
        visibility = [dense_scene.vis([1, 2])] * len(points)
        visibility[0] = dense_scene.vis([1, 2, 3])  # behind image 3
        structure = convert(make_dense_workspace(points, visibility))["structure"]
        assert [o["observationId"] for o in structure[0]["observations"]] == ["1", "2"]
        assert "Observations: 21 in total, 0 of images not exported, 1 invalid" in caplog.text
        assert "probably does not match" not in caplog.text

    def test_many_invalid_observations_warn(self, convert, make_dense_workspace, dense_scene, points, caplog):
        convert(make_dense_workspace(points, [dense_scene.vis([1, 2, 3])] * len(points)))
        assert "33% of the observations are invalid" in caplog.text
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_out_of_range_image_index_is_invalid(self, convert, make_dense_workspace, dense_scene, points,
                                                 caplog):
        visibility = [dense_scene.vis([1, 2])] * len(points)
        visibility[0] = (0, 1, 99)
        assert len(convert(make_dense_workspace(points, visibility))["structure"][0]["observations"]) == 2
        assert "1 invalid" in caplog.text

    def test_input_sfm(self, convert, make_dense_workspace, dense_scene, points, camera_init, write_json, caplog):
        workspace = make_dense_workspace(points, [dense_scene.vis([1, 2, 3])] * len(points))
        images = os.path.join(workspace, "images")
        # c.jpg (image 3) was not imported in CameraInit
        source = camera_init.sfm([camera_init.view("1001", os.path.join(images, "a.jpg"), intrinsic_id="9"),
                                  camera_init.view("1002", os.path.join(images, "b.jpg"), intrinsic_id="9")],
                                 [{"intrinsicId": "9", "sensorWidth": "6.4", "serialNumber": "SN9"}])
        sfm_data = convert(workspace, input_sfm=write_json("cameraInit.sfm", source))
        assert [v["viewId"] for v in sfm_data["views"]] == ["1001", "1002"]
        assert sfm_data["intrinsics"][0]["serialNumber"] == "SN9"
        assert {o["observationId"] for lm in sfm_data["structure"] for o in lm["observations"]} == \
            {"1001", "1002"}
        assert "Image c.jpg not found in the input SfMData (skipping)" in caplog.text
        assert "Observations: 30 in total, 10 of images not exported, 0 invalid" in caplog.text

    def test_image_folder(self, convert, workspace):
        views = convert(workspace, image_folder="/elsewhere/undistorted")["views"]
        assert views[0]["path"] == os.path.join("/elsewhere/undistorted", "a.jpg")

    def test_custom_fused_ply(self, convert, workspace, tmp_path):
        os.makedirs(str(tmp_path / "other"))
        for name in ("fused.ply", "fused.ply.vis"):
            shutil.move(os.path.join(workspace, name), str(tmp_path / "other" / ("cloud" + name[5:])))
        sfm_data = convert(workspace, fused_ply=str(tmp_path / "other" / "cloud.ply"))
        assert len(sfm_data["structure"]) == POINT_COUNT

    def test_logs_through_given_logger(self, workspace, output, caplog):
        caplog.set_level(logging.INFO)
        dense.convert(workspace, output, logger=logging.getLogger("colmap_tests.dense"))
        assert {r.name for r in caplog.records} == {"colmap_tests.dense"}

    @pytest.mark.parametrize("missing", ["fused.ply", "fused.ply.vis"])
    def test_missing_fused_files(self, workspace, output, missing):
        os.remove(os.path.join(workspace, missing))
        with pytest.raises(RuntimeError, match="File not found: .*" + missing.replace(".", r"\.") + "$"):
            dense.convert(workspace, output)

    def test_visibility_count_mismatch(self, make_dense_workspace, dense_scene, points, output, colmap_files):
        workspace = make_dense_workspace(points, [dense_scene.vis([1, 2])] * len(points))
        colmap_files.write_visibility(os.path.join(workspace, "fused.ply.vis"),
                                      [dense_scene.vis([1, 2])] * (len(points) - 1))
        with pytest.raises(RuntimeError, match="has 10 points but .* has 9"):
            dense.convert(workspace, output)

    def test_distorted_model_raises(self, make_dense_workspace, dense_scene, points, output):
        workspace = make_dense_workspace(points, [dense_scene.vis([1, 2])] * len(points),
                                         cameras=[Camera(1, "SIMPLE_RADIAL", 640, 480, [500.0, 320.0, 240.0, 0.05])])
        with pytest.raises(RuntimeError, match="has distortion"):
            dense.convert(workspace, output)

    def test_no_exported_image_raises(self, workspace, output, camera_init, write_json):
        source = camera_init.sfm([camera_init.view("1", "/other/z.jpg")])
        with pytest.raises(RuntimeError, match="No image of the COLMAP model could be exported"):
            dense.convert(workspace, output, input_sfm=write_json("cameraInit.sfm", source))

    def test_images_without_valid_pose_are_skipped(self, convert, make_dense_workspace, dense_scene, points,
                                                   caplog):
        images = dense_scene.images + [Image(4, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1, "d.jpg", [], [])]
        visibility = [dense_scene.vis([1, 2]) + (3,)] * len(points)  # index 3 = image 4 in both orders
        sfm_data = convert(make_dense_workspace(points, visibility, images=images))
        assert [v["viewId"] for v in sfm_data["views"]] == ["1", "2", "3"]
        assert "10 of images not exported" in caplog.text
        assert len(sfm_data["structure"]) == POINT_COUNT


class TestDenseSfmConverter:

    def test_write_returns_count_and_fills_stats(self, make_dense_workspace, dense_scene, points, output):
        visibility = [dense_scene.vis([1, 2])] * len(points)
        visibility[0] = dense_scene.vis([1, 2, 3])  # behind image 3
        visibility[1] = dense_scene.vis([1])  # one view only
        converter = dense.DenseSfmConverter(make_dense_workspace(points, visibility))
        assert converter.write(output) == POINT_COUNT - 1
        assert converter.stats == {"observations": 20, "notExported": 0, "invalid": 1, "dropped": 1}

    def test_settings(self, tmp_path):
        folder = str(tmp_path / "dense")
        converter = dense.DenseSfmConverter(folder)
        assert (converter.cloud.ply_path, converter.cloud.vis_path) == \
            (os.path.join(folder, "fused.ply"), os.path.join(folder, "fused.ply.vis"))
        assert converter.image_folder == os.path.join(folder, "images")
        assert converter.max_points == dense.DenseSfmConverter.DEFAULT_MAX_POINTS
        # with a CameraInit SfMData, the views give the image paths
        assert dense.DenseSfmConverter(folder, input_sfm="cameraInit.sfm").image_folder == ""

    def test_nothing_is_read_before_write(self, tmp_path):
        dense.DenseSfmConverter(str(tmp_path / "missing"))


class TestCommandLine:

    def test_defaults(self, workspace, tmp_path, run_script, read_json):
        result = run_script("dense", workspace, "out.sfm")
        assert result.returncode == 0, result.stderr
        assert "10 landmarks" in result.stderr
        assert len(read_json(str(tmp_path / "out.sfm"))["structure"]) == POINT_COUNT

    def test_options(self, workspace, tmp_path, run_script, read_json, camera_init, write_json):
        images = os.path.join(workspace, "images")
        source = camera_init.sfm([camera_init.view(str(100 + i), os.path.join(images, name))
                                  for i, name in enumerate(["a.jpg", "b.jpg", "c.jpg"])])
        result = run_script("dense", workspace, "out.sfm", "--maxPoints", "3", "--inputSfm",
                            write_json("cameraInit.sfm", source), "--fusedPly", os.path.join(workspace, "fused.ply"))
        assert result.returncode == 0, result.stderr
        sfm_data = read_json(str(tmp_path / "out.sfm"))
        assert landmark_ids(sfm_data) == ["0", "4", "8"]
        assert [v["viewId"] for v in sfm_data["views"]] == ["100", "101", "102"]

    def test_missing_fused_ply_fails(self, tmp_path, run_script):
        result = run_script("dense", tmp_path, "out.sfm")
        assert result.returncode != 0
        assert "File not found" in result.stderr


# Integration tests: Meshroom node, with a fake Meshroom API.

@pytest.fixture(scope="module")
def node_class(load_node):
    """The ColmapDense2MeshroomSfmConvertion node class, loaded with a fake Meshroom API."""
    return load_node("dense", "ColmapDense2MeshroomSfmConvertion")


class TestNode:

    def test_description(self, node_class):
        assert node_class.category == "Colmap"
        assert [a.name for a in node_class.inputs] == ["input", "fusedPly", "inputSfm", "imageFolder", "maxPoints",
                                                       "verboseLevel"]
        assert [a.name for a in node_class.outputs] == ["outputSfm"]
        assert node_class.inputs[4].value == dense.DenseSfmConverter.DEFAULT_MAX_POINTS

    def test_process_chunk(self, node_class, make_chunk, workspace, output, read_json, caplog):
        caplog.set_level(logging.INFO)
        chunk = make_chunk(node_class, input=workspace, outputSfm=output, maxPoints=5)
        node_class().processChunk(chunk)
        chunk.logManager.start.assert_called_once_with("info")
        chunk.logManager.end.assert_called_once_with()
        assert landmark_ids(read_json(output)) == ["0", "2", "4", "6", "8"]
        assert any(r.name == chunk.logger.name for r in caplog.records)

    def test_process_chunk_ends_log_on_error(self, node_class, make_chunk, tmp_path, output):
        chunk = make_chunk(node_class, input=str(tmp_path), outputSfm=output)
        with pytest.raises(RuntimeError, match="File not found"):
            node_class().processChunk(chunk)
        chunk.logManager.end.assert_called_once_with()
