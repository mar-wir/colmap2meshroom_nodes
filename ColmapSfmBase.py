"""
Shared base of the COLMAP to Meshroom conversion nodes.

Both nodes turn a COLMAP reconstruction into an AliceVision SfMData file: the sparse node
(ColmapSparse2MeshroomSfmConvertions.py) from a sparse model, the dense node
(ColmapDense2MeshroomSfmConvertions.py) from a fused point cloud. Their cameras are converted in the same
way, so everything they have in common lives here, and both import this file, which must be in the same
folder. Only the standard library is used, so the nodes also work as plain Python, outside of Meshroom.

    ColmapModelReader       reads COLMAP sparse models (cameras, images, points3D; .bin or .txt)
    SfmJson                 AliceVision SfMData json conventions, loading and writing (also streamed)
    CameraConverter         COLMAP camera -> AliceVision pinhole intrinsic
    PoseConverter           COLMAP image pose -> AliceVision pose
    SourceViews             matches COLMAP image names against the views of a CameraInit SfMData
    SfmViewsConverter       COLMAP cameras and images -> SfMData views, intrinsics and poses
    LandmarkFactory         builds SfMData landmarks and observations
    ColmapConversionNode    shared behavior of the Meshroom nodes (mixin, combined with desc.Node)

How they fit together (both nodes):

    ColmapModelReader.read() -> cameras, images ----> SfmViewsConverter --> views, intrinsics, poses
                                                       |  uses CameraConverter, PoseConverter, and
                                                       |  SourceViews when a CameraInit SfMData is given
                                                       +-> view_ids (COLMAP image id -> view id)
    points (COLMAP 3D points or fused points) + view_ids -> LandmarkFactory --> structure
    SfMData --> SfmJson.write() / SfmJson.write_streamed()

See README.md for the usage of the nodes. This file defines no Meshroom node: Meshroom logs
"No class defined in plugin" for it at startup, which is expected.
"""

__version__ = "1.0"

import json
import logging
import math
import os
import struct
from collections import namedtuple
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

# Meshroom's node API; None outside of Meshroom, which allows running the conversions as standalone scripts.
desc = None  # type: Any
VERBOSE_LEVEL = None  # type: Any
try:
    from meshroom.core import desc
    from meshroom.core.utils import VERBOSE_LEVEL
except ImportError:
    desc = None

# COLMAP model elements, as read by ColmapModelReader
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

# AliceVision SfMData (json) and its parts: views, intrinsics, poses, landmarks
SfmData = Dict[str, Any]
# COLMAP model: cameras, images and points3D, keyed by COLMAP id
Model = Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]


class ColmapModelReader:
    """Read COLMAP sparse models (cameras, images, points3D), in binary (.bin) or text (.txt) format.

    Used for: loading the reconstruction both nodes start from. The sparse node reads the sparse model
    (e.g. sparse/0), the dense node the undistorted model of the dense workspace (dense/sparse).

    Why it is needed: COLMAP's own Python reader (scripts/python/read_write_model.py) depends on numpy, which
    Meshroom's bundled Python does not provide; this is a standard-library port of it. Grouping the readers
    in a class keeps the file formats (camera model ids, binary record layouts, text conventions) in one
    place.

    Since COLMAP 3.12 a model also holds rigs.{bin,txt} and frames.{bin,txt}, but images.{bin,txt} still store
    the full cam_from_world pose of every image, so they are all that is needed here.

    Usage:
        cameras, images, points3D = ColmapModelReader.read("project/sparse/0")
        cameras, images, _ = ColmapModelReader.read("project/dense", read_points=False)  # finds dense/sparse

    Attributes:
        CAMERA_MODELS: COLMAP camera model id -> (model name, number of parameters), see
            colmap/sensor/models.h.
    """

    CAMERA_MODELS = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
        6: ("FULL_OPENCV", 12),
        7: ("FOV", 5),
        8: ("SIMPLE_RADIAL_FISHEYE", 4),
        9: ("RADIAL_FISHEYE", 5),
        10: ("THIN_PRISM_FISHEYE", 12),
        11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
        12: ("SIMPLE_DIVISION", 4),
        13: ("DIVISION", 5),
        14: ("SIMPLE_FISHEYE", 3),
        15: ("FISHEYE", 4),
        16: ("EUCM", 6),
        17: ("EQUIRECTANGULAR", 2),
    }

    @staticmethod
    def read_next_bytes(fid: IO[bytes], num_bytes: int, format_char_sequence: str,
                        endian_character: str = "<") -> Tuple[Any, ...]:
        """Read and unpack the next bytes from a binary file.

        Args:
            fid: Binary file.
            num_bytes: Number of bytes to read.
            format_char_sequence: struct format characters of the values, without endianness.
            endian_character: struct endianness character, little endian by default (as COLMAP writes).

        Returns:
            The unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    @classmethod
    def read_cameras_binary(cls, path: str) -> Dict[int, Camera]:
        """Read a COLMAP cameras.bin file.

        Args:
            path: cameras.bin path.

        Returns:
            The cameras, keyed by camera id, in file order.

        Raises:
            RuntimeError: If a camera model id is unknown.
        """
        cameras = {}
        with open(path, "rb") as fid:
            num_cameras = cls.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_cameras):
                camera_id, model_id, width, height = cls.read_next_bytes(fid, 24, "IiQQ")
                if model_id not in cls.CAMERA_MODELS:
                    raise RuntimeError("Unknown COLMAP camera model id: {}".format(model_id))
                model_name, num_params = cls.CAMERA_MODELS[model_id]
                params = cls.read_next_bytes(fid, 8 * num_params, "d" * num_params)
                cameras[camera_id] = Camera(camera_id, model_name, width, height, list(params))
        return cameras

    @classmethod
    def read_images_binary(cls, path: str) -> Dict[int, Image]:
        """Read a COLMAP images.bin file.

        Args:
            path: images.bin path.

        Returns:
            The images, keyed by image id, in file order.
        """
        images = {}
        with open(path, "rb") as fid:
            num_images = cls.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_images):
                props = cls.read_next_bytes(fid, 64, "IdddddddI")
                image_id = props[0]
                qvec = list(props[1:5])
                tvec = list(props[5:8])
                camera_id = props[8]
                name = b""
                current_char = fid.read(1)
                while current_char != b"\x00":  # null-terminated image name
                    name += current_char
                    current_char = fid.read(1)
                num_points2D = cls.read_next_bytes(fid, 8, "Q")[0]
                xys = []
                point3D_ids = []
                for x, y, point3D_id in struct.iter_unpack("<ddq", fid.read(24 * num_points2D)):
                    xys.append((x, y))
                    point3D_ids.append(point3D_id)
                images[image_id] = Image(image_id, qvec, tvec, camera_id, name.decode("utf-8"), xys, point3D_ids)
        return images

    @classmethod
    def read_points3D_binary(cls, path: str) -> Dict[int, Point3D]:
        """Read a COLMAP points3D.bin file.

        Args:
            path: points3D.bin path.

        Returns:
            The points, keyed by point id, in file order.
        """
        points3D = {}
        with open(path, "rb") as fid:
            num_points = cls.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_points):
                props = cls.read_next_bytes(fid, 43, "QdddBBBd")
                track_length = cls.read_next_bytes(fid, 8, "Q")[0]
                track = cls.read_next_bytes(fid, 8 * track_length, "II" * track_length)
                points3D[props[0]] = Point3D(props[0], list(props[1:4]), list(props[4:7]), props[7],
                                             list(track[0::2]), list(track[1::2]))
        return points3D

    @staticmethod
    def _data_lines(path: str) -> Iterator[str]:
        """Read the data lines of a COLMAP text file.

        Args:
            path: Text file path.

        Yields:
            The stripped lines, skipping comments. Empty lines are kept: in images.txt, an empty second line
            means an image without 2D points.
        """
        with open(path, "r") as fid:
            for line in fid:
                if not line.startswith("#"):
                    yield line.strip()

    @classmethod
    def read_cameras_text(cls, path: str) -> Dict[int, Camera]:
        """Read a COLMAP cameras.txt file.

        Args:
            path: cameras.txt path.

        Returns:
            The cameras, keyed by camera id, in file order.
        """
        cameras = {}
        for line in cls._data_lines(path):
            if not line:
                continue
            elems = line.split()
            camera_id = int(elems[0])
            cameras[camera_id] = Camera(camera_id, elems[1], int(elems[2]), int(elems[3]),
                                        [float(x) for x in elems[4:]])
        return cameras

    @classmethod
    def read_images_text(cls, path: str) -> Dict[int, Image]:
        """Read a COLMAP images.txt file (two lines per image: pose and name, then 2D points).

        Args:
            path: images.txt path.

        Returns:
            The images, keyed by image id, in file order.
        """
        images = {}
        lines = cls._data_lines(path)
        for line in lines:
            if not line:
                continue
            elems = line.split(maxsplit=9)
            image_id = int(elems[0])
            points = next(lines, "").split()
            xys = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 3)]
            point3D_ids = [int(points[i]) for i in range(2, len(points), 3)]
            images[image_id] = Image(image_id, [float(x) for x in elems[1:5]], [float(x) for x in elems[5:8]],
                                     int(elems[8]), elems[9], xys, point3D_ids)
        return images

    @classmethod
    def read_points3D_text(cls, path: str) -> Dict[int, Point3D]:
        """Read a COLMAP points3D.txt file.

        Args:
            path: points3D.txt path.

        Returns:
            The points, keyed by point id, in file order.
        """
        points3D = {}
        for line in cls._data_lines(path):
            if not line:
                continue
            elems = line.split()
            point3D_id = int(elems[0])
            track = [int(x) for x in elems[8:]]
            points3D[point3D_id] = Point3D(point3D_id, [float(x) for x in elems[1:4]],
                                           [int(x) for x in elems[4:7]], float(elems[7]), track[0::2], track[1::2])
        return points3D

    @staticmethod
    def find_model_folder(path: str) -> Tuple[str, str]:
        """Find the folder holding a COLMAP model, descending into "sparse/0" style subfolders if needed.

        The folders tried are, in order: path, path/0, path/sparse/0 and path/sparse. In each folder, the
        binary format is preferred over the text format.

        Args:
            path: Model folder, or a parent folder of it.

        Returns:
            The model folder and the extension of its files (".bin" or ".txt").

        Raises:
            RuntimeError: If no cameras.bin or cameras.txt is found.
        """
        for folder in (path, os.path.join(path, "0"), os.path.join(path, "sparse", "0"),
                       os.path.join(path, "sparse")):
            for ext in (".bin", ".txt"):
                if os.path.isfile(os.path.join(folder, "cameras" + ext)):
                    return folder, ext
        raise RuntimeError("No COLMAP model (cameras.bin or cameras.txt) found in: " + path)

    @classmethod
    def read(cls, path: str, read_points: bool = True) -> Model:
        """Read a COLMAP sparse model, in binary or text format.

        Args:
            path: Model folder, or a parent folder of it (see find_model_folder).
            read_points: Read points3D too; if False, or if the model has no points3D file, no point is
                returned.

        Returns:
            The cameras, images and points3D, each keyed by COLMAP id, in file order.
        """
        folder, ext = cls.find_model_folder(path)
        if ext == ".bin":
            readers = (cls.read_cameras_binary, cls.read_images_binary, cls.read_points3D_binary)
        else:
            readers = (cls.read_cameras_text, cls.read_images_text, cls.read_points3D_text)
        cameras = readers[0](os.path.join(folder, "cameras" + ext))
        images = readers[1](os.path.join(folder, "images" + ext))
        points3D = {}
        points_path = os.path.join(folder, "points3D" + ext)
        if read_points and os.path.isfile(points_path):
            points3D = readers[2](points_path)
        return cameras, images, points3D


class SfmJson:
    """AliceVision SfMData json conventions and files (see aliceVision/sfmDataIO/jsonIO.cpp).

    Used for: reading the SfMData of CameraInit and writing the converted SfMData, in both nodes.

    Why it is needed: AliceVision writes every json value as a string ("1.5", "true"); value() follows that
    convention, so the files match those written by Meshroom itself. And the dense node can export millions
    of points: json.dump would hold the whole structure in memory, so write_streamed() writes the landmarks
    one by one as they are produced.

    Usage:
        source_sfm_data = SfmJson.load("MeshroomCache/CameraInit/<uid>/cameraInit.sfm")
        SfmJson.write("out.sfm", sfm_data)
        count = SfmJson.write_streamed("out.sfm", sfm_data_without_structure, landmark_generator)
    """

    @staticmethod
    def value(value: Any) -> str:
        """Convert a value to an SfMData json string.

        AliceVision writes every json value as a string; mimicking it produces files identical to native ones.

        Args:
            value: Value to convert.

        Returns:
            "true" or "false" for booleans, str(value) otherwise.
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def load(path: str) -> SfmData:
        """Load an SfMData json file (e.g. the output of CameraInit).

        Args:
            path: .sfm file path.

        Returns:
            The SfMData.
        """
        with open(path, "r") as json_file:
            return json.load(json_file)

    @staticmethod
    def write(path: str, sfm_data: SfmData) -> None:
        """Write an SfMData json file, indented like AliceVision's.

        Args:
            path: Output .sfm file path.
            sfm_data: SfMData to write.
        """
        with open(path, "w") as json_file:
            json.dump(sfm_data, json_file, indent=4)

    @staticmethod
    def write_streamed(path: str, sfm_data: SfmData, landmarks: Iterator[Dict[str, Any]]) -> int:
        """Write an SfMData file, streaming the landmarks.

        Landmarks are written one per line as they are produced, so that large point clouds are never held in
        memory as json objects.

        Args:
            path: Output .sfm file path.
            sfm_data: SfMData without "structure" (views, intrinsics, poses, ...).
            landmarks: Landmarks, written as the "structure" of the SfMData.

        Returns:
            The number of landmarks written.
        """
        header = json.dumps(sfm_data, indent=4)
        count = 0
        with open(path, "w") as json_file:
            json_file.write(header[:header.rindex("}")].rstrip())
            json_file.write(',\n    "structure": [')
            for landmark in landmarks:
                json_file.write((",\n" if count else "\n") + "        " + json.dumps(landmark, separators=(",", ":")))
                count += 1
            json_file.write("\n    ]\n}\n")
        return count


class CameraConverter:
    """Convert COLMAP cameras into AliceVision pinhole intrinsics.

    Used for: the intrinsics of the SfMData (through SfmViewsConverter), and in the dense node, the pinhole
    parameters used to project the fused points into the images.

    Why it is needed: the two libraries describe cameras differently. COLMAP stores the focal length in
    pixels and a parameter list whose meaning depends on the camera model; AliceVision stores the focal length
    in mm for a sensor width, the principal point as an offset from the image center, named distortion
    models with their own coefficient order, and changed the axis of "focalLength" in SfMData 1.2.11.

    Usage:
        fx, fy, cx, cy, distortion_type, distortion_params = CameraConverter.to_pinhole(camera)
        intrinsic = CameraConverter.intrinsic(camera.id, camera, (1, 2, 12), cameraInit_intrinsic)

    Attributes:
        DEFAULT_SENSOR_WIDTH: Sensor width in mm used without CameraInit sensor infos.
        FOCAL_Y_SFM_VERSION: Up to SfMData 1.2.10, "focalLength" is the focal along x; from 1.2.11 it is the
            focal along y.
    """

    DEFAULT_SENSOR_WIDTH = 36.0
    FOCAL_Y_SFM_VERSION = (1, 2, 11)

    @staticmethod
    def to_pinhole(camera: Camera) -> Tuple[float, float, float, float, str, List[float]]:
        """Map a COLMAP camera to AliceVision's pinhole intrinsic.

        Distortion is expressed on normalized camera coordinates in both libraries, so parameters map
        directly.

        Args:
            camera: COLMAP camera.

        Returns:
            (fx, fy, cx, cy, distortionType, distortionParams), focal lengths and principal point in pixels.

        Raises:
            RuntimeError: If the camera model is not supported, or is FULL_OPENCV with rational distortion.
        """
        model, p = camera.model, camera.params
        if model == "SIMPLE_PINHOLE":
            return p[0], p[0], p[1], p[2], "none", []
        if model == "PINHOLE":
            return p[0], p[1], p[2], p[3], "none", []
        if model == "SIMPLE_RADIAL":
            return p[0], p[0], p[1], p[2], "radialk1", [p[3]]
        if model == "RADIAL":
            return p[0], p[0], p[1], p[2], "radialk3", [p[3], p[4], 0.0]
        if model == "OPENCV":
            # AliceVision brown: [k1, k2, k3, t1, t2] with t1, t2 = OpenCV p1, p2
            return p[0], p[1], p[2], p[3], "brown", [p[4], p[5], 0.0, p[6], p[7]]
        if model == "FULL_OPENCV":
            # Only representable without the rational part (k4, k5, k6)
            if any(p[9:12]):
                raise RuntimeError("FULL_OPENCV camera {} has rational distortion (k4, k5, k6), "
                                   "which AliceVision does not support".format(camera.id))
            return p[0], p[1], p[2], p[3], "brown", [p[4], p[5], p[8], p[6], p[7]]
        if model == "OPENCV_FISHEYE":
            return p[0], p[1], p[2], p[3], "fisheye", [p[4], p[5], p[6], p[7]]
        if model == "SIMPLE_RADIAL_FISHEYE":
            return p[0], p[0], p[1], p[2], "fisheye", [p[3], 0.0, 0.0, 0.0]
        if model == "RADIAL_FISHEYE":
            return p[0], p[0], p[1], p[2], "fisheye", [p[3], p[4], 0.0, 0.0]
        if model == "SIMPLE_FISHEYE":
            return p[0], p[0], p[1], p[2], "fisheye", [0.0, 0.0, 0.0, 0.0]
        if model == "FISHEYE":
            return p[0], p[1], p[2], p[3], "fisheye", [0.0, 0.0, 0.0, 0.0]
        if model == "FOV":
            return p[0], p[1], p[2], p[3], "fisheye1", [p[4]]
        raise RuntimeError("COLMAP camera model {} (camera {}) is not supported. Supported models: "
                           "SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV (without k4-k6), "
                           "OPENCV_FISHEYE, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, SIMPLE_FISHEYE, FISHEYE, FOV"
                           .format(model, camera.id))

    @classmethod
    def intrinsic(cls, intrinsic_id: int, camera: Camera, version: Tuple[int, ...],
                  template: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build an AliceVision intrinsic from a COLMAP camera.

        Args:
            intrinsic_id: Intrinsic id (the COLMAP camera id).
            camera: COLMAP camera.
            version: SfMData version, which decides whether "focalLength" is along x or y.
            template: CameraInit intrinsic whose sensor infos (sensor size, serial number, initial focal
                length, pixel ratio lock) are reused, if given.

        Returns:
            The intrinsic, with every value as a string like AliceVision writes them.

        Raises:
            RuntimeError: If the camera model is not supported (see to_pinhole).
        """
        template = template or {}
        fx, fy, cx, cy, distortion_type, distortion_params = cls.to_pinhole(camera)
        max_size = float(max(camera.width, camera.height))
        # AliceVision's sensorWidth is matched against the largest image dimension
        sensor_width = float(template.get("sensorWidth", cls.DEFAULT_SENSOR_WIDTH))
        sensor_height = float(template.get("sensorHeight",
                                           sensor_width * min(camera.width, camera.height) / max_size))
        focal = fy if version >= cls.FOCAL_Y_SFM_VERSION else fx
        v = SfmJson.value

        return {
            "intrinsicId": v(intrinsic_id),
            "width": v(camera.width),
            "height": v(camera.height),
            "sensorWidth": v(sensor_width),
            "sensorHeight": v(sensor_height),
            "serialNumber": template.get("serialNumber", "-1"),
            "type": "pinhole",
            "initializationMode": "calibrated",
            "initialFocalLength": template.get("initialFocalLength", "-1"),
            "focalLength": v(focal * sensor_width / max_size),
            "pixelRatio": v(fy / fx),
            "pixelRatioLocked": template.get("pixelRatioLocked", "true"),
            "offsetLocked": "false",
            "scaleLocked": "false",
            # principal point is stored as an offset from the image center
            "principalPoint": [v(cx - camera.width / 2.0), v(cy - camera.height / 2.0)],
            "distortionLocked": "false",
            "distortionInitializationMode": "calibrated",
            "distortionParams": [v(x) for x in distortion_params],
            "undistortionOffset": ["0", "0"],
            "undistortionParams": [],
            "distortionType": distortion_type,
            "undistortionType": "none",
            "locked": "false",
        }


class PoseConverter:
    """Convert COLMAP image poses into AliceVision poses.

    Used for: the poses of the SfMData (through SfmViewsConverter), and in the dense node, the rotation used
    to project the fused points into the images.

    Why it is needed: COLMAP stores each image's world-to-camera transform as a quaternion and a translation
    (R, t); AliceVision stores the same rotation as a column-major matrix together with the camera center
    C = -R^T t. Images that COLMAP could not register can also hold an invalid pose, which is_valid() detects.

    Usage:
        if PoseConverter.is_valid(image):
            pose = PoseConverter.pose(view_id, image)
        rotation = PoseConverter.rotation_matrix(image.qvec)
    """

    @staticmethod
    def rotation_matrix(qvec: Sequence[float]) -> List[List[float]]:
        """Convert a quaternion to a rotation matrix.

        Args:
            qvec: Quaternion (w, x, y, z), as stored by COLMAP; it does not need to be normalized.

        Returns:
            The 3x3 rotation matrix, as a list of rows.
        """
        w, x, y, z = qvec
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]

    @classmethod
    def pose(cls, pose_id: str, image: Image) -> Dict[str, Any]:
        """Build an AliceVision pose from the pose of a COLMAP image.

        COLMAP stores cam_from_world (R, t). AliceVision stores the same world-to-camera rotation R and the
        camera center C = -R^T t; matrices are serialized in column-major order (Eigen's default).

        Args:
            pose_id: Pose id (the view id).
            image: COLMAP image.

        Returns:
            The pose, unlocked.
        """
        rotation = cls.rotation_matrix(image.qvec)
        center = [-sum(rotation[row][col] * image.tvec[row] for row in range(3)) for col in range(3)]
        return {
            "poseId": SfmJson.value(pose_id),
            "pose": {
                "transform": {
                    "rotation": [SfmJson.value(rotation[row][col]) for col in range(3) for row in range(3)],
                    "center": [SfmJson.value(x) for x in center],
                },
                "locked": "false",
                "rotationOnly": "false",
                "removable": "false",
            },
        }

    @staticmethod
    def is_valid(image: Image) -> bool:
        """Check that a COLMAP image has a usable pose.

        Args:
            image: COLMAP image.

        Returns:
            False if its quaternion is null or a pose value is not finite.
        """
        norm = sum(q * q for q in image.qvec)
        return norm > 1e-12 and all(math.isfinite(v) for v in list(image.qvec) + list(image.tvec))


class SourceViews:
    """Match COLMAP image names against the views of a CameraInit sfm.

    Used for: reusing CameraInit's views (view ids, image paths, metadata, sensor infos) when the nodes get an
    "Input SfM", instead of creating new views from the COLMAP image names.

    Why it is needed: COLMAP names images by their path relative to its image folder ("a/IMG_1.jpg"), while
    CameraInit stores absolute paths. Matching the longest common path suffix handles both, and an ambiguous
    name (the same file name in two folders) is refused rather than guessed.

    Usage:
        views = SourceViews(SfmJson.load("cameraInit.sfm"))
        view = views.find("a/IMG_1.jpg")  # the CameraInit view, or None

    Attributes:
        by_suffix: Every trailing part of each view path ("c.jpg", "b/c.jpg", "a/b/c.jpg") -> the views
            whose path ends with it.
    """

    def __init__(self, source_sfm_data: SfmData) -> None:
        """Index the view paths of an SfMData.

        Args:
            source_sfm_data: SfMData from CameraInit.
        """
        self.by_suffix = {}  # type: Dict[str, List[Dict[str, Any]]]
        for view in source_sfm_data.get("views", []):
            parts = self._split(view["path"])
            for i in range(len(parts)):
                self.by_suffix.setdefault("/".join(parts[i:]), []).append(view)

    @staticmethod
    def _split(path: str) -> List[str]:
        """Split a path into its parts, accepting both / and \\ separators."""
        return [p for p in os.path.normpath(path).replace("\\", "/").split("/") if p]

    def find(self, image_name: str) -> Optional[Dict[str, Any]]:
        """Find the view of a COLMAP image.

        COLMAP names are relative to its image folder: the longest suffix of the name matching some views
        is used, falling back to shorter suffixes (down to the file name) when none matches.

        Args:
            image_name: COLMAP image name, e.g. "a/IMG_1.jpg".

        Returns:
            The matching view, or None if no view matches or the longest matching suffix is ambiguous.
        """
        parts = self._split(image_name)
        for i in range(len(parts)):
            candidates = self.by_suffix.get("/".join(parts[i:]), [])
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                return None
        return None


class SfmViewsConverter:
    """Convert the cameras and posed images of a COLMAP model into SfMData views, intrinsics and poses.

    Images using an unknown camera, without a valid pose or (with a source SfMData) not found in its views
    are skipped with a warning. Only the cameras of the exported images become intrinsics.

    Used for: the camera part of the SfMData, which is the same in both nodes. After convert(), view_ids
    tells which COLMAP image became which view, so the landmarks can reference the views.

    Why it is needed: views, intrinsics and poses depend on each other (pose id = view id, intrinsics only for
    the exported images, the SfMData version decides the intrinsic format) and on the same skip rules. Doing
    it in one place guarantees that both nodes write the same cameras, and that the dense node attaches the
    fused points to exactly the views that were exported.

    Usage:
        cameras, images, _ = ColmapModelReader.read("project/sparse/0", read_points=False)
        converter = SfmViewsConverter(cameras, images, source_sfm_data=SfmJson.load("cameraInit.sfm"))
        sfm_data = converter.convert()  # version, views, intrinsics, poses
        view_id = converter.view_ids.get(colmap_image_id)  # None if the image was skipped

    Attributes:
        DEFAULT_SFM_VERSION: SfMData version written by Meshroom 2025.1 (AliceVision 3.3), used without a
            source SfMData. AliceVision refuses to load files with a version newer than its own, so a source
            SfMData's version is used instead when given.
        MIN_SFM_VERSION: Oldest supported version: separate "distortionType" / "undistortionType" fields
            appeared in 1.2.8.
        RIG_KEYS: CameraInit view fields dropped, as each COLMAP image gets its own pose.
        view_ids: After convert(), COLMAP image id -> view id of every exported image, by image id.
    """

    DEFAULT_SFM_VERSION = ["1", "2", "12"]
    MIN_SFM_VERSION = (1, 2, 8)
    RIG_KEYS = ("rigId", "subPoseId", "isPoseIndependant", "resectionId")

    def __init__(self, cameras: Dict[int, Camera], images: Dict[int, Image], image_folder: str = "",
                 source_sfm_data: Optional[SfmData] = None, logger: Optional[logging.Logger] = None) -> None:
        """Set up the conversion.

        Args:
            cameras: COLMAP cameras, keyed by camera id.
            images: COLMAP images, keyed by image id.
            image_folder: Folder of the images used by COLMAP; view paths are <image_folder>/<image name>.
                Only used without source_sfm_data.
            source_sfm_data: SfMData from CameraInit. Its version, view ids, paths and metadata and its
                sensor infos are kept; views are matched to COLMAP images by file name (see SourceViews).
            logger: Logger, the module logger if None.
        """
        self.cameras = cameras
        self.images = images
        self.image_folder = image_folder
        self.source_sfm_data = source_sfm_data
        self.logger = logger or logging.getLogger(__name__)
        self.view_ids = {}  # type: Dict[int, str]

    @staticmethod
    def parse_version(version: Sequence[str]) -> Tuple[int, ...]:
        """Convert an SfMData version (e.g. ["1", "2", "12"]) to a comparable tuple of ints."""
        return tuple(int(v) for v in version)

    def convert(self) -> SfmData:
        """Convert the model, filling view_ids.

        Returns:
            The SfMData: version, views, intrinsics and poses (no structure).

        Raises:
            RuntimeError: If the source SfMData version is older than MIN_SFM_VERSION or a camera model is not
                supported.
        """
        source_sfm_data = self.source_sfm_data
        version = list(source_sfm_data["version"]) if source_sfm_data else list(self.DEFAULT_SFM_VERSION)
        if self.parse_version(version) < self.MIN_SFM_VERSION:
            raise RuntimeError("SfMData version {} is too old (Meshroom 2023.3 or newer is required)"
                               .format(".".join(version)))
        source_views = SourceViews(source_sfm_data) if source_sfm_data else None
        source_intrinsics = {i["intrinsicId"]: i for i in (source_sfm_data or {}).get("intrinsics", [])}

        views = []
        poses = []
        intrinsic_templates = {}
        self.view_ids = {}
        for image_id, image in sorted(self.images.items()):
            view = self._view(image_id, image, source_views, source_intrinsics, intrinsic_templates)
            if view is None:
                continue
            views.append(view)
            poses.append(PoseConverter.pose(view["viewId"], image))
            self.view_ids[image_id] = view["viewId"]

        used_cameras = sorted({self.images[i].camera_id for i in self.view_ids})
        intrinsics = [CameraConverter.intrinsic(camera_id, self.cameras[camera_id], self.parse_version(version),
                                                intrinsic_templates.get(camera_id))
                      for camera_id in used_cameras]
        return {"version": version, "views": views, "intrinsics": intrinsics, "poses": poses}

    def _view(self, image_id: int, image: Image, source_views: Optional[SourceViews],
              source_intrinsics: Dict[str, Dict[str, Any]],
              intrinsic_templates: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build the view of an image, or return None (with a warning) if the image is skipped.

        Args:
            image_id: COLMAP image id.
            image: COLMAP image.
            source_views: CameraInit views, None without source SfMData.
            source_intrinsics: CameraInit intrinsics, keyed by intrinsic id.
            intrinsic_templates: COLMAP camera id -> CameraInit intrinsic, filled with the first CameraInit
                intrinsic found for each camera.

        Returns:
            The view, or None.
        """
        if image.camera_id not in self.cameras:
            self.logger.warning("Image {} uses unknown camera {} (skipping)".format(image.name, image.camera_id))
            return None
        if not PoseConverter.is_valid(image):
            self.logger.warning("Image {} has no valid pose (skipping)".format(image.name))
            return None
        camera = self.cameras[image.camera_id]

        if source_views:
            source_view = source_views.find(image.name)
            if source_view is None:
                self.logger.warning("Image {} not found in the input SfMData (skipping)".format(image.name))
                return None
            view = {k: v for k, v in source_view.items() if k not in self.RIG_KEYS}
            source_size = (int(view.get("width", 0)), int(view.get("height", 0)))
            if source_size[0] > 0 and source_size != (camera.width, camera.height):
                self.logger.warning("Image {} is {}x{} in the input SfMData but {}x{} in COLMAP: COLMAP "
                                    "intrinsics are kept as is".format(image.name, view.get("width"),
                                                                       view.get("height"), camera.width,
                                                                       camera.height))
            if view.get("intrinsicId") in source_intrinsics:
                intrinsic_templates.setdefault(image.camera_id, source_intrinsics[view["intrinsicId"]])
        else:
            view = {
                "viewId": SfmJson.value(image_id),
                "frameId": SfmJson.value(image_id),
                "path": os.path.join(self.image_folder, image.name),
                "metadata": {},
            }

        view.update({
            "poseId": view["viewId"],
            "intrinsicId": SfmJson.value(image.camera_id),
            "width": SfmJson.value(camera.width),
            "height": SfmJson.value(camera.height),
        })
        return view


class LandmarkFactory:
    """Build SfMData landmarks and their observations.

    Landmarks have descType "unknown": no AliceVision features exist for COLMAP points, so nodes that need
    feature or match files cannot use them.

    Used for: the "structure" of the SfMData. The sparse node builds it from the COLMAP 3D points, observed
    at their keypoints; the dense node from the fused points, observed at their projections.

    Why it is needed: both nodes must write the same landmark layout and apply the same rule (a point needs at
    least 2 views to be exported). The number formatting is configurable because the dense node writes
    millions of points, where fewer digits keep the file noticeably smaller.

    Usage:
        factory = LandmarkFactory()  # or LandmarkFactory("{:.3f}".format, "{:.9g}".format)
        observations = [factory.observation(view_id, feature_id, x, y) for ...]
        landmark = factory.landmark(point_id, (r, g, b), (x, y, z), observations)  # None if < 2 views

    Attributes:
        MIN_OBSERVATIONS: Points seen by less views are not exported.
    """

    MIN_OBSERVATIONS = 2

    def __init__(self, format_pixel: Callable[[float], str] = SfmJson.value,
                 format_coordinate: Callable[[float], str] = SfmJson.value) -> None:
        """Set the number formatting.

        Args:
            format_pixel: Formats observation pixel coordinates.
            format_coordinate: Formats landmark 3D coordinates.
        """
        self.format_pixel = format_pixel
        self.format_coordinate = format_coordinate

    def observation(self, view_id: str, feature_id: Any, x: float, y: float) -> Dict[str, Any]:
        """Build an observation of a landmark.

        Args:
            view_id: Id of the view the landmark is observed in.
            feature_id: Feature id (the COLMAP keypoint index, or the landmark id without keypoints).
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            The observation.
        """
        return {
            "observationId": view_id,
            "featureId": SfmJson.value(feature_id),
            "x": [self.format_pixel(x), self.format_pixel(y)],
            "scale": "0",
        }

    def landmark(self, landmark_id: Any, color: Sequence[int], xyz: Sequence[float],
                 observations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build a landmark.

        Args:
            landmark_id: Landmark id.
            color: (red, green, blue) color.
            xyz: World coordinates.
            observations: Observations, at most one per view.

        Returns:
            The landmark, or None if it has less than MIN_OBSERVATIONS observations.
        """
        if len(observations) < self.MIN_OBSERVATIONS:
            return None
        return {
            "landmarkId": SfmJson.value(landmark_id),
            "descType": "unknown",
            "color": [SfmJson.value(c) for c in color],
            "X": [self.format_coordinate(c) for c in xyz],
            "observations": observations,
        }


class ColmapConversionNode:
    """Shared behavior of the COLMAP conversion nodes: category, attributes and chunk processing.

    Used for: the Meshroom node classes of both scripts.

    Why it is needed: both nodes appear in the same "Colmap" category, have the same "Verbose Level" input and
    "Output SfM" output, and must open and close the node log around the conversion, even when it fails.

    This is a mixin, not a desc.Node: Meshroom registers every desc.Node subclass of a module as a node, so a
    desc.Node base class here would show up in Meshroom as an extra, empty node. A node combines it with
    desc.Node (mixin first, so that its processChunk and category are used) and implements run_conversion:

        class MyNode(ColmapConversionNode, desc.Node):
            inputs = [..., ColmapConversionNode.verbose_level_attribute()]
            outputs = [ColmapConversionNode.output_sfm_attribute()]

            def run_conversion(self, node, logger):
                convert(node.input.value, node.outputSfm.value, logger=logger)
    """

    category = "Colmap"

    @staticmethod
    def verbose_level_attribute() -> Any:
        """Returns the "verboseLevel" input attribute description."""
        return desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        )

    @staticmethod
    def output_sfm_attribute() -> Any:
        """Returns the "outputSfm" output attribute description."""
        return desc.File(
            name="outputSfm",
            label="Output SfM",
            description="Path to the output SfMData file.",
            value="{nodeCacheFolder}/sfmdata.sfm",
        )

    def processChunk(self, chunk: Any) -> None:
        """Run the conversion, with the node log open at the node verbose level.

        Args:
            chunk: Meshroom node chunk being computed, giving the node attributes and logger.
        """
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            self.run_conversion(chunk.node, chunk.logger)
        finally:
            chunk.logManager.end()

    def run_conversion(self, node: Any, logger: logging.Logger) -> None:
        """Run the conversion with the node attribute values.

        Args:
            node: Meshroom node, giving the attribute values (e.g. node.input.value).
            logger: Node logger.
        """
        raise NotImplementedError
