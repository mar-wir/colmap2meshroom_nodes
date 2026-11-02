"""
Shared pytest fixtures for the tests of the COLMAP conversion nodes and their shared base.

Fixtures:
    colmap_files        writers for COLMAP models, fused point clouds, visibility and json files
    sparse_scene        mock sparse model: cameras, images and points3D with all the edge cases
    make_model          writes a COLMAP model (binary or text) under tmp_path
    sparse_model        sparse_scene written as a binary model in tmp_path/sparse/0
    dense_scene         mock undistorted dense scene: camera, images, points and their exact projections
    make_dense_workspace writes a COLMAP dense workspace (sparse/, images/, fused.ply, fused.ply.vis)
    camera_init         builders of CameraInit-like SfMData (views, intrinsics)
    sparse_source_sfm   CameraInit SfMData matching sparse_scene
    write_json          writes a json file under tmp_path
    run_script          runs a conversion script from the command line
    load_node           imports a node script (and the base) with a fake Meshroom API, returns its node class
    make_chunk          fake Meshroom chunk for a node class

The tests folder has no __init__.py on purpose: Meshroom discovers node modules with pkgutil.iter_modules,
which skips folders that are not packages, so these files are never loaded as nodes.

Run from the colmap folder (no Meshroom needed):
    python -m pytest tests -v
"""

import importlib.util
import json
import logging
import math
import os
import struct
import subprocess
import sys
import types
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from unittest import mock

import pytest

COLMAP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODULE = "ColmapSfmBase"
SCRIPTS = {
    "sparse": os.path.join(COLMAP_DIR, "ColmapSparse2MeshroomSfmConvertions.py"),
    "dense": os.path.join(COLMAP_DIR, "ColmapDense2MeshroomSfmConvertions.py"),
}

# the modules under test are imported as standalone modules, like the command line scripts
if COLMAP_DIR not in sys.path:
    sys.path.insert(0, COLMAP_DIR)

from ColmapSfmBase import Camera, ColmapModelReader, Image, Point3D  # noqa: E402

MODEL_IDS = {name: model_id for model_id, (name, _) in ColmapModelReader.CAMERA_MODELS.items()}
SQRT_HALF = math.sqrt(0.5)


# COLMAP sparse model writers, the inverse of the readers (see colmap/scripts/python/read_write_model.py).

def write_cameras_binary(path: str, cameras: Iterable[Camera]) -> None:
    """Write cameras.bin.

    Args:
        path: Output file path.
        cameras: Cameras to write, in file order.
    """
    cameras = list(cameras)
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            fid.write(struct.pack("<IiQQ", camera.id, MODEL_IDS[camera.model], camera.width, camera.height))
            fid.write(struct.pack("<{}d".format(len(camera.params)), *camera.params))


def write_images_binary(path: str, images: Iterable[Image]) -> None:
    """Write images.bin.

    Args:
        path: Output file path.
        images: Images to write, in file order.
    """
    images = list(images)
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images:
            fid.write(struct.pack("<IdddddddI", image.id, *image.qvec, *image.tvec, image.camera_id))
            fid.write(image.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image.xys)))
            for (x, y), point3D_id in zip(image.xys, image.point3D_ids):
                fid.write(struct.pack("<ddq", x, y, point3D_id))


def write_points3D_binary(path: str, points3D: Iterable[Point3D]) -> None:
    """Write points3D.bin.

    Args:
        path: Output file path.
        points3D: Points to write, in file order.
    """
    points3D = list(points3D)
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(points3D)))
        for point in points3D:
            fid.write(struct.pack("<QdddBBBd", point.id, *point.xyz, *point.rgb, point.error))
            fid.write(struct.pack("<Q", len(point.image_ids)))
            for image_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
                fid.write(struct.pack("<II", image_id, point2D_idx))


def write_cameras_text(path: str, cameras: Iterable[Camera]) -> None:
    """Write cameras.txt, with COLMAP's comment header.

    Args:
        path: Output file path.
        cameras: Cameras to write, in file order.
    """
    with open(path, "w") as fid:
        fid.write("# Camera list with one line of data per camera:\n")
        fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera in cameras:
            fid.write(" ".join([str(camera.id), camera.model, str(camera.width), str(camera.height)]
                               + [repr(p) for p in camera.params]) + "\n")


def write_images_text(path: str, images: Iterable[Image]) -> None:
    """Write images.txt: two lines per image, the second one empty for images without 2D points.

    Args:
        path: Output file path.
        images: Images to write, in file order.
    """
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("# Image list with two lines of data per image:\n")
        fid.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fid.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image in images:
            fid.write(" ".join([str(image.id)] + [repr(v) for v in list(image.qvec) + list(image.tvec)]
                               + [str(image.camera_id), image.name]) + "\n")
            fid.write(" ".join("{!r} {!r} {}".format(x, y, point3D_id)
                               for (x, y), point3D_id in zip(image.xys, image.point3D_ids)) + "\n")


def write_points3D_text(path: str, points3D: Iterable[Point3D]) -> None:
    """Write points3D.txt.

    Args:
        path: Output file path.
        points3D: Points to write, in file order.
    """
    with open(path, "w") as fid:
        fid.write("# 3D point list with one line of data per point:\n")
        for point in points3D:
            track = []
            for image_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
                track += [str(image_id), str(point2D_idx)]
            fid.write(" ".join([str(point.id)] + [repr(c) for c in point.xyz] + [str(c) for c in point.rgb]
                               + [repr(point.error)] + track) + "\n")


def write_model(folder: str, cameras: Iterable[Camera], images: Iterable[Image],
                points3D: Optional[Iterable[Point3D]] = None, ext: str = ".bin") -> str:
    """Write a COLMAP sparse model.

    Args:
        folder: Model folder, created if needed.
        cameras: Cameras of the model.
        images: Images of the model, in file order.
        points3D: Points of the model; no points3D file is written if None.
        ext: ".bin" or ".txt".

    Returns:
        The model folder.
    """
    os.makedirs(folder, exist_ok=True)
    binary = ext == ".bin"
    (write_cameras_binary if binary else write_cameras_text)(os.path.join(folder, "cameras" + ext), cameras)
    (write_images_binary if binary else write_images_text)(os.path.join(folder, "images" + ext), images)
    if points3D is not None:
        (write_points3D_binary if binary else write_points3D_text)(os.path.join(folder, "points3D" + ext),
                                                                   points3D)
    return folder


# Fused point cloud writers, as written by colmap stereo_fusion (colmap/util/ply.cc, colmap/mvs/fusion.cc).

def write_ply(path: str, points: Sequence[Sequence[float]], ply_format: str = "binary_little_endian",
              normals: bool = True, colors: bool = True, declared_count: Optional[int] = None) -> None:
    """Write a PLY point cloud with COLMAP's fused.ply layout (float coordinates and normals, uchar colors).

    Args:
        path: Output file path.
        points: (x, y, z, red, green, blue) points; colors are ignored if colors is False.
        ply_format: "ascii", "binary_little_endian" or "binary_big_endian".
        normals: Write nx, ny, nz properties (set to 0, 0, 1) like COLMAP.
        colors: Write red, green, blue properties.
        declared_count: Vertex count written in the header, len(points) if None (to mock truncated files).
    """
    properties = [("float", "x"), ("float", "y"), ("float", "z")]
    if normals:
        properties += [("float", "nx"), ("float", "ny"), ("float", "nz")]
    if colors:
        properties += [("uchar", "red"), ("uchar", "green"), ("uchar", "blue")]
    header = "ply\nformat {} 1.0\nelement vertex {}\n".format(
        ply_format, len(points) if declared_count is None else declared_count)
    header += "".join("property {} {}\n".format(t, n) for t, n in properties) + "end_header\n"

    rows = [list(p[:3]) + ([0.0, 0.0, 1.0] if normals else []) + (list(p[3:6]) if colors else []) for p in points]
    with open(path, "wb") as fid:
        fid.write(header.encode("ascii"))
        if ply_format == "ascii":
            for row in rows:
                fid.write((" ".join(str(v) for v in row) + "\n").encode("ascii"))
        else:
            endian = "<" if ply_format == "binary_little_endian" else ">"
            record = struct.Struct(endian + "".join("f" if t == "float" else "B" for t, _ in properties))
            for row in rows:
                fid.write(record.pack(*row))


def write_visibility(path: str, visibility: Sequence[Sequence[int]], declared_count: Optional[int] = None) -> None:
    """Write a COLMAP fused.ply.vis file.

    Args:
        path: Output file path.
        visibility: For every point, the indices of the images it was fused from.
        declared_count: Point count written in the file, len(visibility) if None.
    """
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(visibility) if declared_count is None else declared_count))
        for image_indices in visibility:
            fid.write(struct.pack("<I{}I".format(len(image_indices)), len(image_indices), *image_indices))


def float32(value: float) -> float:
    """Round a value to float32, as stored in fused.ply."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


@pytest.fixture(scope="session")
def colmap_files() -> SimpleNamespace:
    """Writers for the files read by the conversions.

    Returns:
        A namespace of the write_* functions of this module and float32().
    """
    return SimpleNamespace(
        write_cameras_binary=write_cameras_binary, write_images_binary=write_images_binary,
        write_points3D_binary=write_points3D_binary, write_cameras_text=write_cameras_text,
        write_images_text=write_images_text, write_points3D_text=write_points3D_text,
        write_model=write_model, write_ply=write_ply, write_visibility=write_visibility, float32=float32)


# Sparse scene.

@pytest.fixture
def sparse_scene() -> SimpleNamespace:
    """Mock sparse model covering the edge cases of the conversion.

    - 3 cameras: SIMPLE_RADIAL, PINHOLE with fx != fy and an off-center principal point, and an unused one.
    - 5 images in two folders: 1 and 3 with identity rotation, 2 rotated by 90 degrees around z, 4 using an
      unknown camera, 5 without a valid pose (null quaternion) and without 2D points.
    - 4 points: 1 seen by images 1 and 2; 2 seen by image 1 only; 3 seen twice by image 1 and once by
      image 3; 4 seen by images 4 (not exported) and 1.

    Returns:
        A namespace with cameras, images and points3D lists.
    """
    return SimpleNamespace(
        cameras=[
            Camera(1, "SIMPLE_RADIAL", 640, 480, [500.0, 320.0, 240.0, 0.01]),
            Camera(2, "PINHOLE", 800, 600, [800.0, 900.0, 410.0, 290.0]),
            Camera(3, "PINHOLE", 100, 100, [50.0, 50.0, 50.0, 50.0]),
        ],
        images=[
            Image(1, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0], 1, "a/IMG_1.jpg",
                  [(10.5, 20.25), (30.0, 40.0), (50.0, 60.0)], [1, 3, 3]),
            Image(2, [SQRT_HALF, 0.0, 0.0, SQRT_HALF], [1.0, 2.0, 3.0], 1, "a/IMG_2.jpg", [(11.0, 21.0)], [1]),
            Image(3, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 2, "b/IMG_3.jpg", [(70.0, 80.0)], [3]),
            Image(4, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 99, "b/IMG_4.jpg", [(1.0, 2.0)], [4]),
            Image(5, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1, "b/IMG_5.jpg", [], []),
        ],
        points3D=[
            Point3D(1, [0.1, 0.2, 0.3], [10, 20, 30], 0.5, [1, 2], [0, 0]),
            Point3D(2, [1.0, 1.0, 1.0], [40, 50, 60], 0.5, [1], [1]),
            Point3D(3, [-1.0, 0.5, 2.0], [70, 80, 90], 0.25, [1, 1, 3], [1, 2, 0]),
            Point3D(4, [2.0, 2.0, 2.0], [1, 2, 3], 1.0, [4, 1], [0, 2]),
        ],
    )


@pytest.fixture
def make_model(tmp_path) -> Callable[..., str]:
    """Factory writing a COLMAP model under tmp_path.

    Returns:
        make_model(relative_folder, cameras, images, points3D=None, ext=".bin") -> model folder.
    """
    def _make_model(relative_folder: str, cameras: Iterable[Camera], images: Iterable[Image],
                    points3D: Optional[Iterable[Point3D]] = None, ext: str = ".bin") -> str:
        return write_model(str(tmp_path / relative_folder), cameras, images, points3D, ext)
    return _make_model


@pytest.fixture
def sparse_model(make_model, sparse_scene) -> str:
    """The sparse scene, written as a binary model in tmp_path/sparse/0.

    Returns:
        The model folder.
    """
    return make_model(os.path.join("sparse", "0"), sparse_scene.cameras, sparse_scene.images,
                      sparse_scene.points3D)


# Dense scene.

@pytest.fixture
def dense_scene() -> SimpleNamespace:
    """Mock undistorted model of a dense workspace.

    One PINHOLE camera (640x480, f = 500). Images 1 and 2 face each other across the origin, image 3 faces
    away from it, so points near the origin are seen by images 1 and 2 and are behind image 3. images.bin
    stores them in the order 3, 1, 2: the file order and the image id order of fused.ply.vis differ, and
    only one of them makes the points project into their images.

    Returns:
        A namespace with:
            camera, images: the model, images in file order.
            id_order, file_order: image id -> fused.ply.vis index, for both orders.
            points(count=10): points near the origin, rounded to float32 like in fused.ply.
            vis(image_ids, order): fused.ply.vis indices of images in an order.
            project(image_id, point): (u, v) projection into image 1 or 2, computed by hand.
    """
    def points(count: int = 10) -> List[Tuple[float, float, float, int, int, int]]:
        result = []
        for i in range(count):
            t = i / float(max(count - 1, 1))
            result.append((float32(-0.5 + t), float32(0.25 - 0.5 * t), float32(-0.25 + 0.5 * t),
                           i % 256, (2 * i) % 256, 255 - i % 256))
        return result

    def project(image_id: int, point: Sequence[float]) -> Tuple[float, float]:
        x, y, z = point[:3]
        if image_id == 1:
            return 500.0 * x / (z + 5.0) + 320.0, 500.0 * y / (z + 5.0) + 240.0
        if image_id == 2:
            return 500.0 * -x / (5.0 - z) + 320.0, 500.0 * y / (5.0 - z) + 240.0
        raise ValueError("Image {} sees no point".format(image_id))

    id_order = {1: 0, 2: 1, 3: 2}
    return SimpleNamespace(
        camera=Camera(1, "PINHOLE", 640, 480, [500.0, 500.0, 320.0, 240.0]),
        images=[
            Image(3, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -5.0], 1, "c.jpg", [], []),
            Image(1, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0], 1, "a.jpg", [], []),
            # 180 degrees around y: R = diag(-1, 1, -1)
            Image(2, [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 5.0], 1, "b.jpg", [], []),
        ],
        id_order=id_order,
        file_order={3: 0, 1: 1, 2: 2},
        points=points,
        vis=lambda image_ids, order=id_order: tuple(order[i] for i in image_ids),
        project=project,
    )


@pytest.fixture
def make_dense_workspace(tmp_path, dense_scene) -> Callable[..., str]:
    """Factory writing a COLMAP dense workspace (sparse/, images/, fused.ply, fused.ply.vis) under tmp_path.

    Returns:
        make_dense_workspace(points, visibility, cameras=None, images=None, ply_format="binary_little_endian",
        folder="dense") -> workspace folder. cameras and images default to the dense scene.
    """
    def _make(points: Sequence[Sequence[float]], visibility: Sequence[Sequence[int]],
              cameras: Optional[List[Camera]] = None, images: Optional[List[Image]] = None,
              ply_format: str = "binary_little_endian", folder: str = "dense") -> str:
        root = str(tmp_path / folder)
        images = dense_scene.images if images is None else images
        write_model(os.path.join(root, "sparse"), cameras or [dense_scene.camera], images)
        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        for image in images:
            open(os.path.join(root, "images", image.name), "wb").close()
        write_ply(os.path.join(root, "fused.ply"), points, ply_format=ply_format)
        write_visibility(os.path.join(root, "fused.ply.vis"), visibility)
        return root
    return _make


# CameraInit SfMData and json files.

@pytest.fixture(scope="session")
def camera_init() -> SimpleNamespace:
    """Builders of CameraInit-like SfMData.

    Returns:
        A namespace with:
            view(view_id, path, intrinsic_id="1", width=640, height=480): a view as written by CameraInit,
                with rig infos and metadata.
            sfm(views, intrinsics=(), version=("1", "2", "12")): an SfMData with views and intrinsics only.
    """
    def view(view_id: str, path: str, intrinsic_id: str = "1", width: int = 640,
             height: int = 480) -> Dict[str, Any]:
        return {
            "viewId": view_id, "poseId": view_id, "frameId": "0", "intrinsicId": intrinsic_id,
            "rigId": "-1", "subPoseId": "-1", "isPoseIndependant": "true", "resectionId": "0",
            "path": path, "width": str(width), "height": str(height),
            "metadata": {"Exif:FocalLength": "4.5", "Make": "TestCam"},
        }

    def sfm(views: Sequence[Dict[str, Any]], intrinsics: Sequence[Dict[str, Any]] = (),
            version: Sequence[str] = ("1", "2", "12")) -> Dict[str, Any]:
        return {"version": list(version), "featuresFolders": [], "matchesFolders": [],
                "views": list(views), "intrinsics": list(intrinsics)}

    return SimpleNamespace(view=view, sfm=sfm)


@pytest.fixture
def sparse_source_sfm(camera_init) -> Callable[..., Dict[str, Any]]:
    """CameraInit SfMData matching the sparse scene, with other view ids and paths, and image 3 at full size.

    Returns:
        sparse_source_sfm(version=("1", "2", "10")) -> SfMData. Version 1.2.10 stores the focal along x.
    """
    def _source_sfm(version: Sequence[str] = ("1", "2", "10")) -> Dict[str, Any]:
        return camera_init.sfm(
            [
                camera_init.view("101", "/photos/a/IMG_1.jpg", intrinsic_id="7"),
                camera_init.view("102", "/photos/a/IMG_2.jpg", intrinsic_id="7"),
                camera_init.view("103", "/photos/b/IMG_3.jpg", intrinsic_id="8", width=4000, height=3000),
            ],
            [
                {"intrinsicId": "7", "sensorWidth": "6.4", "sensorHeight": "4.8", "serialNumber": "SN7",
                 "initialFocalLength": "4.5", "pixelRatioLocked": "false"},
                {"intrinsicId": "8", "sensorWidth": "8", "serialNumber": "SN8"},
            ],
            version=version)
    return _source_sfm


@pytest.fixture
def write_json(tmp_path) -> Callable[[str, Any], str]:
    """Factory writing a json file under tmp_path.

    Returns:
        write_json(name, data) -> file path.
    """
    def _write_json(name: str, data: Any) -> str:
        path = str(tmp_path / name)
        with open(path, "w") as fid:
            json.dump(data, fid)
        return path
    return _write_json


@pytest.fixture(scope="session")
def read_json() -> Callable[[str], Any]:
    """Returns read_json(path) -> loaded json data."""
    def _read_json(path: str) -> Any:
        with open(path) as fid:
            return json.load(fid)
    return _read_json


# Command line.

@pytest.fixture
def run_script(tmp_path) -> Callable[..., subprocess.CompletedProcess]:
    """Factory running a conversion script in a new Python process, from tmp_path.

    Returns:
        run_script(name, *args) -> completed process, name being "sparse" or "dense".
    """
    def _run(name: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, SCRIPTS[name]] + [str(a) for a in args], capture_output=True,
                              text=True, cwd=str(tmp_path))
    return _run


# Fake Meshroom API, to load the node classes outside of Meshroom.

class _FakeParam:
    """Node attribute description: keeps the keyword arguments as attributes."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeNode:
    """Base class of node descriptions."""


def _fake_meshroom_modules() -> Dict[str, types.ModuleType]:
    """Build fake meshroom, meshroom.core, meshroom.core.desc and meshroom.core.utils modules."""
    desc = types.ModuleType("meshroom.core.desc")
    desc.Node = _FakeNode
    for name in ("File", "IntParam", "FloatParam", "BoolParam", "ChoiceParam", "StringParam"):
        setattr(desc, name, type(name, (_FakeParam,), {}))
    utils = types.ModuleType("meshroom.core.utils")
    utils.VERBOSE_LEVEL = ["fatal", "error", "warning", "info", "debug", "trace"]
    core = types.ModuleType("meshroom.core")
    core.desc, core.utils = desc, utils
    meshroom = types.ModuleType("meshroom")
    meshroom.core = core
    return {"meshroom": meshroom, "meshroom.core": core, "meshroom.core.desc": desc, "meshroom.core.utils": utils}


@pytest.fixture(scope="session")
def load_node() -> Callable[[str, str], type]:
    """Factory importing a node script with a fake meshroom.core, so that its node class is defined.

    Returns:
        load_node(name, class_name) -> node class, name being "sparse" or "dense".
    """
    def _load(name: str, class_name: str) -> type:
        # sys.modules is restored on exit: the base module imported with the fake API does not leak to the
        # other tests, which use the standalone (desc = None) one
        with mock.patch.dict(sys.modules, _fake_meshroom_modules()):
            sys.modules.pop(BASE_MODULE, None)
            spec =importlib.util.spec_from_file_location("colmap_{}_node_under_test".format(name), SCRIPTS[name])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return getattr(module, class_name)
    return _load


@pytest.fixture(scope="session")
def make_chunk() -> Callable[..., SimpleNamespace]:
    """Factory of fake Meshroom node chunks.

    Only the attributes declared by the node are available on chunk.node, so a node reading an undeclared
    attribute fails like in Meshroom.

    Returns:
        make_chunk(node_class, **values) -> chunk with node (attribute values, overriding the declared
        defaults), logManager (a mock) and logger.
    """
    def _make(node_class: type, **values: Any) -> SimpleNamespace:
        attributes = {attr.name: SimpleNamespace(value=values.get(attr.name, attr.value))
                      for attr in node_class.inputs + node_class.outputs}
        return SimpleNamespace(node=SimpleNamespace(**attributes), logManager=mock.Mock(),
                               logger=logging.getLogger("colmap_tests.chunk"))
    return _make
