r"""
COLMAP dense reconstruction to Meshroom SfM conversion.

Converts the fused point cloud of a COLMAP dense reconstruction (fused.ply and its visibility file
fused.ply.vis, written by "colmap stereo_fusion") into an AliceVision SfMData file (.sfm, json). Cameras and
poses come from the undistorted sparse model of the dense workspace, and every fused point becomes a
landmark observed by the images it was fused from. Meshroom's Meshing node can then build the mesh from
these points directly, so the DepthMap and DepthMapFilter nodes (the slowest part of the Meshroom
pipeline) are not needed.

    PlyPointReader                        reads the vertices of PLY point clouds
    FusedPointCloud                       fused.ply and fused.ply.vis, read together
    CameraProjection                      projects world points into an undistorted image
    DenseSfmConverter                     the conversion
    ColmapDense2MeshroomSfmConvertion     the Meshroom node
    convert()                             shortcut to DenseSfmConverter, for scripts

The COLMAP reader and the conversion of the cameras are in ColmapSfmBase.py, shared with the sparse node
(ColmapSparse2MeshroomSfmConvertions.py); it must be in the same folder. Only the standard library is used, so
the conversion also runs as a plain script:

    python3 ColmapDense2MeshroomSfmConvertions.py <colmap_dense_folder> <output.sfm>
        [--fusedPly PATH] [--inputSfm cameraInit.sfm] [--imageFolder PATH] [--maxPoints N]

See README.md for the dense workspace layout, the Meshroom graph (including the PrepareDenseScene node that
Texturing needs), the node attributes, the conversion details and the limitations.
"""

__version__ = "1.0"

import argparse
import itertools
import logging
import math
import os
import struct
from typing import IO, Any, Dict, Iterator, List, Optional, Sequence, Tuple

if __package__:
    # Meshroom imports node files as modules of their package folder
    from .ColmapSfmBase import (Camera, CameraConverter, ColmapConversionNode, ColmapModelReader, Image,
                                LandmarkFactory, PoseConverter, SfmData, SfmJson, SfmViewsConverter, desc)
else:
    # standalone: the base module is found in the folder of this script
    from ColmapSfmBase import (Camera, CameraConverter, ColmapConversionNode, ColmapModelReader, Image,
                               LandmarkFactory, PoseConverter, SfmData, SfmJson, SfmViewsConverter, desc)

# PLY element: (name, count, [(property type, property name)])
PlyElement = Tuple[str, int, List[Tuple[str, str]]]
# fused point: (x, y, z, red, green, blue)
Point = Tuple[float, float, float, int, int, int]


class PlyPointReader:
    """Read the vertices of PLY point clouds (ascii, binary little and big endian).

    Used for: reading fused.ply, through FusedPointCloud.

    Why it is needed: the conversion must not depend on packages that Meshroom's bundled Python lacks (plyfile,
    numpy, open3d), and fused point clouds can hold millions of points, so the vertices are read as a stream,
    chunk by chunk, without loading the whole file. Only the subset of PLY used for point clouds is
    supported: a first "vertex" element with scalar properties, as written by COLMAP (colmap/util/ply.cc).

    Usage:
        count = PlyPointReader.vertex_count("fused.ply")
        for x, y, z, red, green, blue in PlyPointReader.read_points("fused.ply"):
            ...

    Attributes:
        TYPES: PLY property type -> struct format character.
    """

    TYPES = {
        "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
        "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
        "int": "i", "int32": "i", "uint": "I", "uint32": "I",
        "float": "f", "float32": "f", "double": "d", "float64": "d",
    }

    @staticmethod
    def read_header(fid: IO[bytes]) -> Tuple[Optional[str], List[PlyElement]]:
        """Read a PLY header, leaving the file positioned at the start of the data.

        Args:
            fid: PLY file opened in binary mode, positioned at its start.

        Returns:
            The format ("ascii", "binary_little_endian" or "binary_big_endian", None if missing) and the
            elements, as a list of (name, count, [(property type, property name)]).

        Raises:
            RuntimeError: If the file is not a PLY file or its header is truncated.
        """
        if fid.readline().strip() != b"ply":
            raise RuntimeError("Not a PLY file: " + fid.name)
        ply_format = None
        elements = []
        for line in fid:
            words = line.decode("ascii", "replace").split()
            if not words:
                continue
            if words[0] == "format":
                ply_format = words[1]
            elif words[0] == "element":
                elements.append((words[1], int(words[2]), []))
            elif words[0] == "property" and elements:
                elements[-1][2].append((" ".join(words[1:-1]), words[-1]))
            elif words[0] == "end_header":
                return ply_format, elements
        raise RuntimeError("Truncated PLY header: " + fid.name)

    @classmethod
    def vertex_element(cls, path: str, elements: List[PlyElement]) -> Tuple[int, List[str], str]:
        """Describe the vertex element of a PLY file, which must be its first element.

        Args:
            path: PLY file path, for error messages.
            elements: Elements of the PLY header, as returned by read_header.

        Returns:
            The vertex count, the property names and the struct format characters of the properties.

        Raises:
            RuntimeError: If the first element is not "vertex", a property type is not supported (e.g. a list)
                or the x, y, z coordinates are missing.
        """
        if not elements or elements[0][0] != "vertex":
            raise RuntimeError("PLY file whose first element is not 'vertex': " + path)
        _, count, properties = elements[0]
        for prop_type, name in properties:
            if prop_type not in cls.TYPES:
                raise RuntimeError("Unsupported PLY vertex property '{} {}' in: {}".format(prop_type, name, path))
        names = [name for _, name in properties]
        if not {"x", "y", "z"} <= set(names):
            raise RuntimeError("PLY vertices without x, y, z coordinates: " + path)
        return count, names, "".join(cls.TYPES[prop_type] for prop_type, _ in properties)

    @classmethod
    def vertex_count(cls, path: str) -> int:
        """Read the number of vertices of a PLY file from its header.

        Args:
            path: PLY file path.

        Returns:
            The vertex count.
        """
        with open(path, "rb") as fid:
            return cls.vertex_element(path, cls.read_header(fid)[1])[0]

    @staticmethod
    def _binary_rows(fid: IO[bytes], count: int, record: struct.Struct,
                     chunk_size: int = 65536) -> Iterator[Tuple[Any, ...]]:
        """Read binary PLY rows chunk by chunk.

        Args:
            fid: PLY file, positioned at the first row.
            count: Number of rows to read.
            record: Struct of one row, endianness included.
            chunk_size: Number of rows read at once.

        Yields:
            The unpacked values of each row.

        Raises:
            RuntimeError: If the file holds less than count rows.
        """
        remaining = count
        while remaining > 0:
            num_rows = min(remaining, chunk_size)
            data = fid.read(num_rows * record.size)
            if len(data) != num_rows * record.size:
                raise RuntimeError("Truncated PLY file: " + fid.name)
            for row in record.iter_unpack(data):
                yield row
            remaining -= num_rows

    @classmethod
    def read_points(cls, path: str) -> Iterator[Point]:
        """Read the vertices of a PLY point cloud.

        Args:
            path: PLY file path.

        Yields:
            (x, y, z, red, green, blue) for every vertex, white if the file has no colors.

        Raises:
            RuntimeError: If the format is not supported or the file is truncated.
        """
        with open(path, "rb") as fid:
            ply_format, elements = cls.read_header(fid)
            count, names, formats = cls.vertex_element(path, elements)
            if ply_format == "ascii":
                rows = ([float(v) for v in line.split()] for line in fid if line.strip())
            elif ply_format in ("binary_little_endian", "binary_big_endian"):
                endian = "<" if ply_format == "binary_little_endian" else ">"
                rows = cls._binary_rows(fid, count, struct.Struct(endian + formats))
            else:
                raise RuntimeError("Unsupported PLY format '{}': {}".format(ply_format, path))

            xyz = [names.index(n) for n in ("x", "y", "z")]
            rgb = [names.index(n) for n in ("red", "green", "blue")] if {"red", "green", "blue"} <= set(names) \
                else None
            num_rows = 0
            for row in itertools.islice(rows, count):
                num_rows += 1
                color = (int(row[rgb[0]]), int(row[rgb[1]]), int(row[rgb[2]])) if rgb else (255, 255, 255)
                yield (row[xyz[0]], row[xyz[1]], row[xyz[2]]) + color
            if num_rows != count:
                raise RuntimeError("Truncated PLY file: " + path)


class FusedPointCloud:
    """A COLMAP fused point cloud: fused.ply and its visibility file fused.ply.vis.

    Used for: reading the input of the dense conversion, point by point together with the images each point
    was fused from.

    Why it is needed: the points (fused.ply, a standard PLY file) and their visibility (fused.ply.vis, a COLMAP
    specific file, see WritePointsVisibility in colmap/mvs/fusion.cc) are two files that only make sense
    together: the n-th visibility entry belongs to the n-th point. This class checks that both exist and
    match, and reads them in lockstep, as streams.

    Usage:
        cloud = FusedPointCloud("dense/fused.ply")  # visibility: dense/fused.ply.vis
        num_points = cloud.check()
        for point, image_indices in cloud:
            ...

    Attributes:
        ply_path: fused.ply path.
        vis_path: fused.ply.vis path.
    """

    def __init__(self, ply_path: str, vis_path: Optional[str] = None) -> None:
        """Set the file paths; nothing is read yet.

        Args:
            ply_path: fused.ply path.
            vis_path: Visibility file path, <ply_path>.vis if None.
        """
        self.ply_path = ply_path
        self.vis_path = vis_path if vis_path is not None else ply_path + ".vis"

    def check(self) -> int:
        """Check that both files exist and hold the same number of points.

        Returns:
            The number of points.

        Raises:
            RuntimeError: If a file is missing or the point counts differ.
        """
        for path in (self.ply_path, self.vis_path):
            if not os.path.isfile(path):
                raise RuntimeError("File not found: " + path)
        num_points = PlyPointReader.vertex_count(self.ply_path)
        num_visibilities = self.visibility_count(self.vis_path)
        if num_visibilities != num_points:
            raise RuntimeError("{} has {} points but {} has {}".format(
                self.ply_path, num_points, self.vis_path, num_visibilities))
        return num_points

    def points(self) -> Iterator[Point]:
        """Yields the (x, y, z, red, green, blue) points of fused.ply."""
        return PlyPointReader.read_points(self.ply_path)

    def visibility(self) -> Iterator[Tuple[int, ...]]:
        """Yields, for every point, the fused.ply.vis indices of the images it was fused from."""
        return self.read_visibility(self.vis_path)

    def __iter__(self) -> Iterator[Tuple[Point, Tuple[int, ...]]]:
        """Yields (point, image indices) for every point."""
        return zip(self.points(), self.visibility())

    @staticmethod
    def visibility_count(path: str) -> int:
        """Read the number of points of a visibility file.

        Args:
            path: Visibility file path.

        Returns:
            The point count.
        """
        with open(path, "rb") as fid:
            return ColmapModelReader.read_next_bytes(fid, 8, "Q")[0]

    @staticmethod
    def read_visibility(path: str) -> Iterator[Tuple[int, ...]]:
        """Read a visibility file.

        The file holds a uint64 point count, then for each point a uint32 image count and uint32 image indices.

        Args:
            path: Visibility file path.

        Yields:
            For every point of fused.ply, the indices of the images it was fused from.
        """
        with open(path, "rb") as fid:
            num_points = ColmapModelReader.read_next_bytes(fid, 8, "Q")[0]
            data = fid.read()
        offset = 0
        for _ in range(num_points):
            num_images = struct.unpack_from("<I", data, offset)[0]
            yield struct.unpack_from("<{}I".format(num_images), data, offset + 4)
            offset += 4 + 4 * num_images


class CameraProjection:
    """World -> pixel projection of an undistorted COLMAP image.

    Used for: computing where each fused point appears in the images it was fused from (the observation
    coordinates of the landmarks), validating these observations, and finding the image order of
    fused.ply.vis.

    Why it is needed: fused.ply.vis only says which images a point comes from, not where it is in them, while
    an SfMData observation needs pixel coordinates. The fused points refer to the undistorted images of the
    dense workspace, so the camera must have no distortion: a distorted camera means the wrong model (the
    original sparse/0 instead of dense/sparse) was given, which is refused rather than silently projected
    wrongly.

    Usage:
        projection = CameraProjection(cameras[image.camera_id], image)
        uv = projection.project((x, y, z))  # (u, v) in pixels, or None if not visible

    Attributes:
        MARGIN: Projections further than this outside the image (in pixels) are invalid.
        rotation, translation: World-to-camera transform.
        fx, fy, cx, cy: Pinhole parameters in pixels.
        width, height: Image size.
    """

    MARGIN = 1.0

    def __init__(self, camera: Camera, image: Image) -> None:
        """Build the projection of an image.

        Args:
            camera: COLMAP camera of the image, which must have no distortion.
            image: COLMAP image, holding the pose.

        Raises:
            RuntimeError: If the camera has distortion, i.e. the model is not the undistorted one.
        """
        self.fx, self.fy, self.cx, self.cy, _, distortion_params = CameraConverter.to_pinhole(camera)
        if any(distortion_params):
            raise RuntimeError("COLMAP camera {} ({}) has distortion: use the undistorted model of the dense "
                               "workspace (dense/sparse, written by colmap image_undistorter), as the fused points "
                               "refer to undistorted images".format(camera.id, camera.model))
        self.rotation = PoseConverter.rotation_matrix(image.qvec)
        self.translation = image.tvec
        self.width, self.height = camera.width, camera.height

    def project(self, point: Sequence[float]) -> Optional[Tuple[float, float]]:
        """Project a world point to pixel coordinates.

        Args:
            point: World point; only its first 3 values (x, y, z) are used.

        Returns:
            The (u, v) pixel coordinates, or None if the point is behind the camera or more than MARGIN pixels
            outside the image.
        """
        r, t = self.rotation, self.translation
        x, y, z = point[0], point[1], point[2]
        zc = r[2][0] * x + r[2][1] * y + r[2][2] * z + t[2]
        if zc <= 0:
            return None
        u = self.fx * (r[0][0] * x + r[0][1] * y + r[0][2] * z + t[0]) / zc + self.cx
        v = self.fy * (r[1][0] * x + r[1][1] * y + r[1][2] * z + t[1]) / zc + self.cy
        margin = self.MARGIN
        if not (-margin <= u <= self.width + margin and -margin <= v <= self.height + margin):
            return None
        return u, v


class DenseSfmConverter:
    """Convert a COLMAP dense reconstruction into an SfMData whose landmarks are the fused points.

    Used for: the dense node, the command line and Python scripts. Meshing builds the mesh from the landmarks
    of the result, so DepthMap and DepthMapFilter are not needed.

    Why it is needed: it puts together the steps of the dense conversion:
        1. the cameras, from the undistorted model of the workspace, converted like the sparse node does
           (ColmapModelReader, SfmViewsConverter);
        2. the order of the images in fused.ply.vis, which depends on the COLMAP version
           (detect_image_order);
        3. the landmarks: every kept fused point, observed at its projection (CameraProjection) into the
           exported images it was fused from (LandmarkFactory), streamed to the file (SfmJson.write_streamed).

    Usage:
        converter = DenseSfmConverter("project/dense", input_sfm="cameraInit.sfm", max_points=500000)
        num_landmarks = converter.write("dense.sfm")
        print(converter.stats)

    Attributes:
        DEFAULT_MAX_POINTS: Default maximum number of exported points; SfMData json files with millions of
            points are slow to load in AliceVision.
        INVALID_WARNING_RATIO: Above this ratio of invalid observations, a warning says that fused.ply.vis
            probably does not match the model.
        colmap_dense_folder, cloud, input_sfm, image_folder, max_points, logger: The conversion settings.
        stats: After write(): number of observations in total, of images not exported, invalid (behind or
            outside their image), and number of points dropped (seen by less than 2 exported views).
    """

    DEFAULT_MAX_POINTS = 1000000
    INVALID_WARNING_RATIO = 0.05

    def __init__(self, colmap_dense_folder: str, fused_ply: str = "", input_sfm: str = "", image_folder: str = "",
                 max_points: int = DEFAULT_MAX_POINTS, logger: Optional[logging.Logger] = None) -> None:
        """Set up the conversion; nothing is read before write().

        Args:
            colmap_dense_folder: COLMAP dense workspace folder, holding sparse/, images/ and fused.ply.
            fused_ply: Fused point cloud path, <colmap_dense_folder>/fused.ply if empty. <fused_ply>.vis is
                read too.
            input_sfm: SfMData file from a CameraInit run on the undistorted images, whose views are reused.
            image_folder: Undistorted images folder, only used without input_sfm;
                <colmap_dense_folder>/images if both are empty.
            max_points: Maximum number of exported points, 0 for all. Points are kept evenly through the file.
            logger: Logger, the module logger if None.
        """
        self.colmap_dense_folder = colmap_dense_folder
        self.cloud = FusedPointCloud(fused_ply or os.path.join(colmap_dense_folder, "fused.ply"))
        self.input_sfm = input_sfm
        self.image_folder = image_folder or ("" if input_sfm else os.path.join(colmap_dense_folder, "images"))
        self.max_points = max_points
        self.logger = logger or logging.getLogger(__name__)
        self.stats = {"observations": 0, "notExported": 0, "invalid": 0, "dropped": 0}

    def write(self, output_sfm: str) -> int:
        """Convert the reconstruction and write the SfMData file.

        Args:
            output_sfm: Output .sfm file path.

        Returns:
            The number of landmarks written.

        Raises:
            RuntimeError: If an input file is missing, fused.ply and fused.ply.vis do not match, a camera has
                distortion or no image of the model can be exported.
        """
        num_points = self.cloud.check()
        source_sfm_data = SfmJson.load(self.input_sfm) if self.input_sfm else None

        # views, intrinsics and poses, as written by the sparse conversion
        cameras, images, _ = ColmapModelReader.read(self.colmap_dense_folder, read_points=False)
        self.logger.info("COLMAP model: {} cameras, {} images".format(len(cameras), len(images)))
        converter = SfmViewsConverter(cameras, images, self.image_folder, source_sfm_data, self.logger)
        sfm_data = converter.convert()
        view_ids = converter.view_ids
        if not view_ids:
            raise RuntimeError("No image of the COLMAP model could be exported (see the warnings above)")
        projections = {image_id: CameraProjection(cameras[images[image_id].camera_id], images[image_id])
                       for image_id in view_ids}

        order = self.detect_image_order(images, projections, self.cloud, num_points, self.logger)
        step = int(math.ceil(num_points / float(self.max_points))) if 0 < self.max_points < num_points else 1
        self.logger.info("Fused point cloud: {} points{}".format(
            num_points, ", keeping 1 point out of {}".format(step) if step > 1 else ""))

        self.stats = {"observations": 0, "notExported": 0, "invalid": 0, "dropped": 0}
        num_landmarks = SfmJson.write_streamed(output_sfm, sfm_data,
                                               self._landmarks(order, view_ids, projections, step))
        self._log_summary(sfm_data, num_landmarks)
        return num_landmarks

    @staticmethod
    def detect_image_order(images: Dict[int, Image], projections: Dict[int, CameraProjection],
                           cloud: FusedPointCloud, num_points: int, logger: logging.Logger,
                           num_samples: int = 2000) -> List[int]:
        """Find the order of the images indexed by fused.ply.vis.

        fused.ply.vis indexes the registered images of the model, in their file order for older COLMAP versions
        and sorted by image id for newer ones. The order in which most observations of a sample of points
        project into their images is kept.

        Args:
            images: COLMAP images, keyed by image id, in model file order.
            projections: Projections of the exported images, keyed by image id.
            cloud: Fused point cloud.
            num_points: Number of fused points.
            logger: Logger for the chosen order.
            num_samples: Approximate number of points tested.

        Returns:
            The COLMAP image ids, indexed like fused.ply.vis.
        """
        candidates = [list(images)]
        if sorted(images) != candidates[0]:
            candidates.append(sorted(images))
        if len(candidates) == 1 or num_points == 0:
            return candidates[0]

        step = max(1, num_points // num_samples)
        scores = [0] * len(candidates)
        for i, (point, image_indices) in enumerate(cloud):
            if i % step:
                continue
            for k, order in enumerate(candidates):
                for idx in image_indices:
                    projection = projections.get(order[idx]) if idx < len(order) else None
                    if projection and projection.project(point) is not None:
                        scores[k] += 1
        best = max(range(len(candidates)), key=lambda k: scores[k])
        logger.info("fused.ply.vis image order: {} ({} valid sample observations, {} with the other order)".format(
            ("file order", "image id order")[best], scores[best], scores[1 - best]))
        return candidates[best]

    def _landmarks(self, order: List[int], view_ids: Dict[int, str], projections: Dict[int, CameraProjection],
                   step: int) -> Iterator[Dict[str, Any]]:
        """Convert the kept fused points into landmarks, counting the observations in stats.

        Args:
            order: COLMAP image ids, indexed like fused.ply.vis.
            view_ids: COLMAP image id -> view id of the exported images.
            projections: Projections of the exported images, keyed by image id.
            step: Keep 1 point out of step.

        Yields:
            An SfMData landmark for every kept point seen by at least 2 exported views.
        """
        stats = self.stats
        # fused points have no keypoints: featureId = landmarkId. Fewer digits than the sparse conversion keep
        # large point clouds smaller (pixels to 1/1000, float32 coordinates exactly).
        factory = LandmarkFactory(format_pixel="{:.3f}".format, format_coordinate="{:.9g}".format)
        for i, (point, image_indices) in enumerate(self.cloud):
            if i % step:
                continue
            observations = []
            for idx in image_indices:
                stats["observations"] += 1
                image_id = order[idx] if idx < len(order) else None
                if image_id is None:
                    stats["invalid"] += 1
                    continue
                if image_id not in view_ids:
                    stats["notExported"] += 1
                    continue
                xy = projections[image_id].project(point)
                if xy is None:
                    stats["invalid"] += 1
                    continue
                observations.append(factory.observation(view_ids[image_id], i, xy[0], xy[1]))
            landmark = factory.landmark(i, point[3:], point[:3], observations)
            if landmark is None:
                stats["dropped"] += 1
                continue
            yield landmark

    def _log_summary(self, sfm_data: SfmData, num_landmarks: int) -> None:
        """Log the observation statistics, warning when many are invalid, and the SfMData content.

        Args:
            sfm_data: Written SfMData (without structure).
            num_landmarks: Number of landmarks written.
        """
        stats = self.stats
        self.logger.info("Observations: {} in total, {} of images not exported, {} invalid (behind or outside the "
                         "image)".format(stats["observations"], stats["notExported"], stats["invalid"]))
        if stats["invalid"] > self.INVALID_WARNING_RATIO * stats["observations"]:
            self.logger.warning("{:.0f}% of the observations are invalid: {} probably does not match the COLMAP "
                                "model of {}".format(100.0 * stats["invalid"] / stats["observations"],
                                                     self.cloud.vis_path, self.colmap_dense_folder))
        self.logger.info("SfMData: {} views, {} intrinsics, {} landmarks ({} points seen by less than 2 views "
                         "dropped)".format(len(sfm_data["views"]), len(sfm_data["intrinsics"]), num_landmarks,
                                           stats["dropped"]))


def convert(colmap_dense_folder: str, output_sfm: str, fused_ply: str = "", input_sfm: str = "",
            image_folder: str = "", max_points: int = DenseSfmConverter.DEFAULT_MAX_POINTS,
            logger: Optional[logging.Logger] = None) -> None:
    """Convert a COLMAP dense reconstruction into an SfMData file (shortcut to DenseSfmConverter.write).

    Args:
        colmap_dense_folder: COLMAP dense workspace folder, holding sparse/, images/ and fused.ply.
        output_sfm: Output .sfm file path.
        fused_ply: Fused point cloud path, <colmap_dense_folder>/fused.ply if empty.
        input_sfm: SfMData file from a CameraInit run on the undistorted images, whose views are reused.
        image_folder: Undistorted images folder, only used without input_sfm.
        max_points: Maximum number of exported points, 0 for all.
        logger: Logger, the module logger if None.
    """
    DenseSfmConverter(colmap_dense_folder, fused_ply, input_sfm, image_folder, max_points, logger).write(output_sfm)


if desc is not None:
    class ColmapDense2MeshroomSfmConvertion(ColmapConversionNode, desc.Node):
        """Meshroom node converting a COLMAP fused point cloud into an SfMData (see DenseSfmConverter).

        Used for: replacing Meshroom's SfM and depth map computation by a COLMAP dense reconstruction:
        CameraInit -> this node -> Meshing -> MeshFiltering -> Texturing, with PrepareDenseScene giving
        Texturing its images.
        """

        documentation = """
Converts a COLMAP dense reconstruction (fused.ply and fused.ply.vis, from colmap stereo_fusion) into an
AliceVision SfMData file whose landmarks are the fused points, so Meshing can build the mesh without DepthMap.

Typical graph: CameraInit (on the dense/images folder) -> this node (Input SfM) -> Meshing (Depth Maps Folder
left empty) -> MeshFiltering -> Texturing (Dense SfMData from Meshing, Images Folder from a PrepareDenseScene
run on this node's Output SfM).
"""

        inputs = [
            desc.File(
                name="input",
                label="COLMAP Dense Folder",
                description="COLMAP dense workspace folder (e.g. 'dense'), holding sparse/, images/ and fused.ply.",
                value="",
            ),
            desc.File(
                name="fusedPly",
                label="Fused Point Cloud",
                description="Fused point cloud (fused.ply, with its fused.ply.vis file). "
                            "Uses <COLMAP Dense Folder>/fused.ply if empty.",
                value="",
            ),
            desc.File(
                name="inputSfm",
                label="Input SfM",
                description="Input SfMData from a CameraInit run on the undistorted images (dense/images). "
                            "Views are matched to COLMAP images by file name.",
                value="",
            ),
            desc.File(
                name="imageFolder",
                label="Image Folder",
                description="Folder of the undistorted images, used if no Input SfM is given. "
                            "Uses <COLMAP Dense Folder>/images if empty.",
                value="",
            ),
            desc.IntParam(
                name="maxPoints",
                label="Max Points",
                description="Maximum number of exported points (0 for all). Large SfMData files are slow to load.",
                value=DenseSfmConverter.DEFAULT_MAX_POINTS,
                range=(0, 50000000, 100000),
            ),
            ColmapConversionNode.verbose_level_attribute(),
        ]

        outputs = [
            ColmapConversionNode.output_sfm_attribute(),
        ]

        def run_conversion(self, node: Any, logger: logging.Logger) -> None:
            """Run the conversion with the node attribute values.

            Args:
                node: Meshroom node, giving the attribute values.
                logger: Node logger.
            """
            DenseSfmConverter(node.input.value,
                              fused_ply=node.fusedPly.value,
                              input_sfm=node.inputSfm.value,
                              image_folder=node.imageFolder.value,
                              max_points=node.maxPoints.value,
                              logger=logger).write(node.outputSfm.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a COLMAP fused point cloud into an AliceVision SfMData file.")
    parser.add_argument("input", help="COLMAP dense workspace folder")
    parser.add_argument("output", help="output .sfm file")
    parser.add_argument("--fusedPly", default="", help="fused point cloud (default: <input>/fused.ply)")
    parser.add_argument("--inputSfm", default="", help="SfMData from Meshroom's CameraInit on <input>/images")
    parser.add_argument("--imageFolder", default="", help="undistorted image folder (default: <input>/images)")
    parser.add_argument("--maxPoints", type=int, default=DenseSfmConverter.DEFAULT_MAX_POINTS,
                        help="maximum number of points, 0 for all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert(args.input, args.output, args.fusedPly, args.inputSfm, args.imageFolder, args.maxPoints)
