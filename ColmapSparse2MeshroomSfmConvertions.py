"""
COLMAP sparse reconstruction to Meshroom SfM conversion.

Converts a COLMAP sparse reconstruction (cameras, images and points3D, in .bin or .txt format) into an
AliceVision SfMData file (.sfm, json), so a COLMAP reconstruction can feed the Meshroom dense pipeline
(PrepareDenseScene, DepthMap, Meshing, Texturing, ...) instead of Meshroom's own feature matching and SfM.

    SparseSfmConverter                    the conversion (cameras from the shared base, plus the 3D points)
    ColmapSparse2MeshroomSfmConvertion    the Meshroom node
    colmap2meshroom(), convert()          shortcuts to SparseSfmConverter, for scripts

The COLMAP reader and the conversion of the cameras are in ColmapSfmBase.py, shared with the dense node
(ColmapDense2MeshroomSfmConvertions.py); it must be in the same folder. Only the standard library is used, so
the conversion also runs as a plain script:

    python3 ColmapSparse2MeshroomSfmConvertions.py <colmap_sparse_folder> <output.sfm>
        [--inputSfm cameraInit.sfm | --imageFolder <images>] [--noLandmarks]

See README.md for the installation, the Meshroom graph, the node attributes, the conversion details and the
limitations.
"""

__version__ = "3.0"

import argparse
import logging
from typing import Any, Dict, List, Optional

if __package__:
    # Meshroom imports node files as modules of their package folder
    from .ColmapSfmBase import (ColmapConversionNode, ColmapModelReader, Image, LandmarkFactory, Point3D,
                                SfmData, SfmJson, SfmViewsConverter, desc)
else:
    # standalone: the base module is found in the folder of this script
    from ColmapSfmBase import (ColmapConversionNode, ColmapModelReader, Image, LandmarkFactory, Point3D,
                               SfmData, SfmJson, SfmViewsConverter, desc)


class SparseSfmConverter:
    """Convert a COLMAP sparse model into an AliceVision SfMData: cameras, poses and the COLMAP 3D points.

    Used for: the sparse node, the command line and Python scripts. The result replaces the output of
    Meshroom's StructureFromMotion node, so the rest of the Meshroom pipeline runs on COLMAP's cameras.

    Why it is needed: it puts together the three steps of the sparse conversion: reading the model
    (ColmapModelReader), converting the cameras like the dense node does (SfmViewsConverter), and the part
    specific to sparse models, the landmarks. Each COLMAP 3D point becomes a landmark observed at its COLMAP
    keypoints in the exported views; DepthMap uses them to estimate the depth range of each view.

    Usage:
        # with the views of a CameraInit run (recommended)
        converter = SparseSfmConverter.from_files("project/sparse/0", input_sfm="cameraInit.sfm")
        converter.write("sfm.sfm")

        # with views created from the image folder, inspecting the result before writing it
        converter = SparseSfmConverter("project/sparse/0", image_folder="project/images")
        sfm_data = converter.to_sfm_data()

    Attributes:
        colmap_folder: Model folder, or a parent folder of it (see ColmapModelReader.find_model_folder).
        image_folder: Folder of the images used by COLMAP, only used without source_sfm_data.
        source_sfm_data: SfMData from CameraInit, or None.
        include_landmarks: Export the COLMAP 3D points as landmarks.
        logger: Logger of the conversion.
    """

    def __init__(self, colmap_folder: str, image_folder: str = "", source_sfm_data: Optional[SfmData] = None,
                 include_landmarks: bool = True, logger: Optional[logging.Logger] = None) -> None:
        """Set up the conversion; nothing is read before to_sfm_data() or write().

        Args:
            colmap_folder: Model folder, or a parent folder of it (see ColmapModelReader.find_model_folder).
            image_folder: Folder of the images used by COLMAP; view paths are <image_folder>/<image name>.
                Only used without source_sfm_data.
            source_sfm_data: SfMData from CameraInit. Its version, view ids, paths and metadata and its
                sensor infos are kept; views are matched to COLMAP images by file name.
            include_landmarks: Export the COLMAP 3D points as landmarks.
            logger: Logger, the module logger if None.
        """
        self.colmap_folder = colmap_folder
        self.image_folder = image_folder
        self.source_sfm_data = source_sfm_data
        self.include_landmarks = include_landmarks
        self.logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_files(cls, colmap_folder: str, input_sfm: str = "", image_folder: str = "",
                   include_landmarks: bool = True, logger: Optional[logging.Logger] = None) -> "SparseSfmConverter":
        """Set up the conversion from file paths, as given to the node and the command line.

        Args:
            colmap_folder: Model folder, or a parent folder of it.
            input_sfm: SfMData file from CameraInit, whose views are reused (recommended).
            image_folder: Folder of the images used by COLMAP, only used without input_sfm.
            include_landmarks: Export the COLMAP 3D points as landmarks.
            logger: Logger, the module logger if None.

        Returns:
            The converter.

        Raises:
            RuntimeError: If neither input_sfm nor image_folder is given.
        """
        if not image_folder and not input_sfm:
            raise RuntimeError("Must input an image folder or an SfMData file")
        source_sfm_data = SfmJson.load(input_sfm) if input_sfm else None
        return cls(colmap_folder, image_folder, source_sfm_data, include_landmarks, logger)

    def to_sfm_data(self) -> SfmData:
        """Read the model and convert it.

        Returns:
            The SfMData: version, views, intrinsics, poses and, with include_landmarks, structure.

        Raises:
            RuntimeError: If the model is not found, the SfMData version is older than 1.2.8 or a camera model
                is not supported.
        """
        cameras, images, points3D = ColmapModelReader.read(self.colmap_folder, read_points=self.include_landmarks)
        self.logger.info("COLMAP model: {} cameras, {} images, {} points".format(
            len(cameras), len(images), len(points3D)))

        converter = SfmViewsConverter(cameras, images, self.image_folder, self.source_sfm_data, self.logger)
        sfm_data = converter.convert()
        if self.include_landmarks and points3D:
            sfm_data["structure"] = self._structure(images, points3D, converter.view_ids)

        self.logger.info("SfMData: {} views, {} intrinsics, {} landmarks".format(
            len(sfm_data["views"]), len(sfm_data["intrinsics"]), len(sfm_data.get("structure", []))))
        return sfm_data

    def write(self, output_sfm: str) -> None:
        """Convert the model and write the SfMData file.

        Args:
            output_sfm: Output .sfm file path.

        Raises:
            RuntimeError: If the conversion fails (see to_sfm_data).
        """
        SfmJson.write(output_sfm, self.to_sfm_data())

    @staticmethod
    def _structure(images: Dict[int, Image], points3D: Dict[int, Point3D],
                   view_ids: Dict[int, str]) -> List[Dict[str, Any]]:
        """Convert the COLMAP 3D points into landmarks.

        Observations of images that were not exported are ignored, as are further observations of a point in
        a view that already has one. Points seen by less than 2 exported views are dropped.

        Args:
            images: COLMAP images, holding the keypoints.
            points3D: COLMAP 3D points.
            view_ids: COLMAP image id -> view id of the exported images.

        Returns:
            The landmarks, by point id.
        """
        factory = LandmarkFactory()
        structure = []
        for point3D_id, point in sorted(points3D.items()):
            observations = {}
            for image_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
                if image_id not in view_ids or view_ids[image_id] in observations:
                    continue
                x, y = images[image_id].xys[point2D_idx]
                observations[view_ids[image_id]] = factory.observation(view_ids[image_id], point2D_idx, x, y)
            landmark = factory.landmark(point3D_id, point.rgb, point.xyz, list(observations.values()))
            if landmark is not None:
                structure.append(landmark)
        return structure


def colmap2meshroom(colmap_folder: str, image_folder: str = "", source_sfm_data: Optional[SfmData] = None,
                    include_landmarks: bool = True, logger: Optional[logging.Logger] = None) -> SfmData:
    """Convert a COLMAP sparse model into an SfMData (shortcut to SparseSfmConverter.to_sfm_data).

    Args:
        colmap_folder: Model folder, or a parent folder of it.
        image_folder: Folder of the images used by COLMAP, only used without source_sfm_data.
        source_sfm_data: SfMData from CameraInit, whose views are reused.
        include_landmarks: Export the COLMAP 3D points as landmarks.
        logger: Logger, the module logger if None.

    Returns:
        The SfMData.
    """
    return SparseSfmConverter(colmap_folder, image_folder, source_sfm_data, include_landmarks,
                              logger).to_sfm_data()


def convert(colmap_folder: str, output_sfm: str, input_sfm: str = "", image_folder: str = "",
            include_landmarks: bool = True, logger: Optional[logging.Logger] = None) -> None:
    """Convert a COLMAP sparse model into an SfMData file (shortcut to SparseSfmConverter.from_files and write).

    Args:
        colmap_folder: Model folder, or a parent folder of it.
        output_sfm: Output .sfm file path.
        input_sfm: SfMData file from CameraInit, whose views are reused (recommended).
        image_folder: Folder of the images used by COLMAP, only used without input_sfm.
        include_landmarks: Export the COLMAP 3D points as landmarks.
        logger: Logger, the module logger if None.
    """
    SparseSfmConverter.from_files(colmap_folder, input_sfm, image_folder, include_landmarks, logger).write(output_sfm)


if desc is not None:
    class ColmapSparse2MeshroomSfmConvertion(ColmapConversionNode, desc.Node):
        """Meshroom node converting a COLMAP sparse model into an SfMData (see SparseSfmConverter).

        Used for: replacing Meshroom's feature extraction, matching and StructureFromMotion by a COLMAP
        reconstruction: CameraInit -> this node -> PrepareDenseScene -> DepthMap -> ... -> Texturing.
        """

        documentation = """
Converts a COLMAP sparse reconstruction (cameras, images and points3D, in .bin or .txt format)
into an AliceVision SfMData file, so it can be used by the Meshroom dense pipeline (PrepareDenseScene, DepthMap, ...).

Connect the output of a CameraInit node to "Input SfM" to keep Meshroom's view ids and image metadata.
"""

        inputs = [
            desc.File(
                name="input",
                label="COLMAP Sparse Folder",
                description="COLMAP sparse model folder (e.g. 'sparse/0'), holding cameras, images and points3D files.",
                value="",
            ),
            desc.File(
                name="inputSfm",
                label="Input SfM",
                description="Input SfMData from CameraInit. Views are matched to COLMAP images by file name.",
                value="",
            ),
            desc.File(
                name="imageFolder",
                label="Image Folder",
                description="Folder of the images used by COLMAP (needed if no Input SfM is given).",
                value="",
            ),
            desc.BoolParam(
                name="includeLandmarks",
                label="Include Landmarks",
                description="Export COLMAP 3D points as SfMData landmarks (used e.g. by DepthMap to estimate depth ranges).",
                value=True,
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
            SparseSfmConverter.from_files(node.input.value,
                                          input_sfm=node.inputSfm.value,
                                          image_folder=node.imageFolder.value,
                                          include_landmarks=node.includeLandmarks.value,
                                          logger=logger).write(node.outputSfm.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a COLMAP sparse model into an AliceVision SfMData file.")
    parser.add_argument("input", help="COLMAP sparse model folder")
    parser.add_argument("output", help="output .sfm file")
    parser.add_argument("--inputSfm", default="", help="SfMData from Meshroom's CameraInit")
    parser.add_argument("--imageFolder", default="", help="image folder (if no --inputSfm)")
    parser.add_argument("--noLandmarks", action="store_true", help="do not export 3D points")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert(args.input, args.output, args.inputSfm, args.imageFolder, not args.noLandmarks)
