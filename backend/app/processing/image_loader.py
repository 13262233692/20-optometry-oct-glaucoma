import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


class ImageLoadingError(Exception):
    pass


@dataclass
class VolumeInfo:
    shape: Tuple[int, int, int]
    voxel_spacing: Tuple[float, float, float]
    file_format: str
    affine: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class MedicalImageLoader:
    def __init__(self):
        self._nibabel_available = False
        self._sitk_available = False
        self._init_libraries()

    def _init_libraries(self):
        try:
            import nibabel
            self._nibabel = nibabel
            self._nibabel_available = True
            logger.info("Nibabel library available for NIfTI loading")
        except ImportError:
            logger.warning("Nibabel not available, NIfTI loading disabled")

        try:
            import SimpleITK
            self._sitk = SimpleITK
            self._sitk_available = True
            logger.info("SimpleITK library available for MHA loading")
        except ImportError:
            logger.warning("SimpleITK not available, MHA loading disabled")

    def _detect_format(self, file_path: Union[str, Path]) -> str:
        path_str = str(file_path).lower()
        if path_str.endswith((".nii", ".nii.gz")):
            return "nifti"
        elif path_str.endswith((".mha", ".mhd")):
            return "mha"
        else:
            raise ImageLoadingError(f"Unsupported file format: {file_path}")

    def load(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, VolumeInfo]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ImageLoadingError(f"File not found: {file_path}")

        file_format = self._detect_format(file_path)
        logger.info(f"Loading {file_format.upper()} file: {file_path.name}")

        if file_format == "nifti":
            return self._load_nifti(file_path)
        elif file_format == "mha":
            return self._load_mha(file_path)
        else:
            raise ImageLoadingError(f"Unsupported file format: {file_format}")

    def _load_nifti(self, file_path: Path) -> Tuple[np.ndarray, VolumeInfo]:
        if not self._nibabel_available:
            raise ImageLoadingError("Nibabel library not available for NIfTI loading")

        try:
            img = self._nibabel.load(str(file_path))
            data = np.asarray(img.dataobj, dtype=np.float32)

            if data.ndim == 4 and data.shape[3] == 1:
                data = data.squeeze(axis=3)

            if data.ndim != 3:
                raise ImageLoadingError(f"Expected 3D volume, got {data.ndim}D array")

            header = img.header
            voxel_sizes = tuple(float(x) for x in header.get_zooms()[:3])
            affine = img.affine

            info = VolumeInfo(
                shape=data.shape,
                voxel_spacing=voxel_sizes,
                file_format="nifti",
                affine=affine,
                metadata=dict(header)
            )

            logger.info(f"Loaded NIfTI volume: shape={data.shape}, spacing={voxel_sizes}")
            return data, info

        except Exception as e:
            logger.error(f"Failed to load NIfTI file: {e}")
            raise ImageLoadingError(f"Failed to load NIfTI file: {e}") from e

    def _load_mha(self, file_path: Path) -> Tuple[np.ndarray, VolumeInfo]:
        if not self._sitk_available:
            raise ImageLoadingError("SimpleITK library not available for MHA loading")

        try:
            image = self._sitk.ReadImage(str(file_path))
            data = self._sitk.GetArrayFromImage(image).astype(np.float32)

            if data.ndim == 4 and data.shape[0] == 1:
                data = data.squeeze(axis=0)

            if data.ndim != 3:
                raise ImageLoadingError(f"Expected 3D volume, got {data.ndim}D array")

            spacing = tuple(float(x) for x in image.GetSpacing())
            origin = tuple(float(x) for x in image.GetOrigin())
            direction = np.array(image.GetDirection()).reshape(3, 3)

            affine = np.eye(4)
            affine[:3, :3] = direction * np.array(spacing)
            affine[:3, 3] = origin

            metadata = {
                "origin": origin,
                "spacing": spacing,
                "direction": direction,
                "size": image.GetSize(),
                "keys": image.GetMetaDataKeys()
            }

            for key in image.GetMetaDataKeys():
                try:
                    metadata[key] = image.GetMetaData(key)
                except Exception:
                    pass

            info = VolumeInfo(
                shape=data.shape,
                voxel_spacing=spacing,
                file_format="mha",
                affine=affine,
                metadata=metadata
            )

            logger.info(f"Loaded MHA volume: shape={data.shape}, spacing={spacing}")
            return data, info

        except Exception as e:
            logger.error(f"Failed to load MHA file: {e}")
            raise ImageLoadingError(f"Failed to load MHA file: {e}") from e

    def load_from_bytes(self, file_bytes: bytes, file_format: str) -> Tuple[np.ndarray, VolumeInfo]:
        if file_format.lower() in ["nifti", "nii", "nii.gz"]:
            return self._load_nifti_from_bytes(file_bytes)
        elif file_format.lower() in ["mha", "mhd"]:
            return self._load_mha_from_bytes(file_bytes)
        else:
            raise ImageLoadingError(f"Unsupported format for bytes loading: {file_format}")

    def _load_nifti_from_bytes(self, file_bytes: bytes) -> Tuple[np.ndarray, VolumeInfo]:
        if not self._nibabel_available:
            raise ImageLoadingError("Nibabel library not available")

        try:
            file_map = self._nibabel.FileHolder(fileobj=io.BytesIO(file_bytes))
            img = self._nibabel.Nifti1Image.from_file_map({"image": file_map, "header": file_map})
            data = np.asarray(img.dataobj, dtype=np.float32)

            if data.ndim == 4 and data.shape[3] == 1:
                data = data.squeeze(axis=3)

            header = img.header
            voxel_sizes = tuple(float(x) for x in header.get_zooms()[:3])

            info = VolumeInfo(
                shape=data.shape,
                voxel_spacing=voxel_sizes,
                file_format="nifti",
                affine=img.affine,
                metadata=dict(header)
            )

            return data, info
        except Exception as e:
            logger.error(f"Failed to load NIfTI from bytes: {e}")
            raise ImageLoadingError(f"Failed to load NIfTI from bytes: {e}") from e

    def _load_mha_from_bytes(self, file_bytes: bytes) -> Tuple[np.ndarray, VolumeInfo]:
        if not self._sitk_available:
            raise ImageLoadingError("SimpleITK library not available")

        try:
            image = self._sitk.ReadImage(io.BytesIO(file_bytes))
            data = self._sitk.GetArrayFromImage(image).astype(np.float32)

            if data.ndim == 4 and data.shape[0] == 1:
                data = data.squeeze(axis=0)

            spacing = tuple(float(x) for x in image.GetSpacing())
            origin = tuple(float(x) for x in image.GetOrigin())
            direction = np.array(image.GetDirection()).reshape(3, 3)

            affine = np.eye(4)
            affine[:3, :3] = direction * np.array(spacing)
            affine[:3, 3] = origin

            metadata = {"origin": origin, "spacing": spacing, "direction": direction}

            info = VolumeInfo(
                shape=data.shape,
                voxel_spacing=spacing,
                file_format="mha",
                affine=affine,
                metadata=metadata
            )

            return data, info
        except Exception as e:
            logger.error(f"Failed to load MHA from bytes: {e}")
            raise ImageLoadingError(f"Failed to load MHA from bytes: {e}") from e


_default_loader = None


def get_default_loader() -> MedicalImageLoader:
    global _default_loader
    if _default_loader is None:
        _default_loader = MedicalImageLoader()
    return _default_loader


def load_medical_image(file_path: Union[str, Path]) -> Tuple[np.ndarray, VolumeInfo]:
    return get_default_loader().load(file_path)


def get_volume_info(file_path: Union[str, Path]) -> VolumeInfo:
    _, info = get_default_loader().load(file_path)
    return info
