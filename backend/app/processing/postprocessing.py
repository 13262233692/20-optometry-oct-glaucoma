import base64
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

from ..config import get_settings
from ..models import HealthStatus, ThicknessMapData, DefectRegion
from ..utils import get_logger
from .preprocessing import PreprocessingResult

logger = get_logger(__name__)


@dataclass
class PostprocessingResult:
    segmentation_mask: np.ndarray
    probability_map: Optional[np.ndarray]
    thickness_map: Optional[np.ndarray]
    defect_regions: Optional[List[DefectRegion]]
    statistics: Dict[str, float]


def encode_segmentation_mask(
    mask: np.ndarray,
    compress: bool = True,
    compression_level: int = 9
) -> str:
    mask_uint8 = mask.astype(np.uint8)
    mask_bytes = mask_uint8.tobytes()

    if compress:
        mask_bytes = zlib.compress(mask_bytes, level=compression_level)

    encoded = base64.b64encode(mask_bytes).decode("utf-8")
    return encoded


def decode_segmentation_mask(
    encoded: str,
    shape: Tuple[int, int, int],
    compressed: bool = True
) -> np.ndarray:
    mask_bytes = base64.b64decode(encoded)

    if compressed:
        mask_bytes = zlib.decompress(mask_bytes)

    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(shape)
    return mask


def extract_thickness_map(
    segmentation_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    axis: int = 2,
    method: str = "axial_projection"
) -> np.ndarray:
    if method == "axial_projection":
        rnfl_mask = (segmentation_mask > 0).astype(np.float32)
        thickness_voxels = np.sum(rnfl_mask, axis=axis)
        physical_thickness = thickness_voxels * voxel_spacing[axis] * 1000
        return physical_thickness.astype(np.float32)

    elif method == "distance_transform":
        rnfl_mask = (segmentation_mask > 0).astype(np.uint8)
        edt = ndimage.distance_transform_edt(rnfl_mask, sampling=voxel_spacing)
        thickness_map = np.max(edt, axis=axis) * 1000
        return thickness_map.astype(np.float32)

    elif method == "top_bottom":
        rnfl_mask = (segmentation_mask > 0)

        top_surface = np.full(rnfl_mask.shape[:2], -1, dtype=np.int32)
        bottom_surface = np.full(rnfl_mask.shape[:2], -1, dtype=np.int32)

        for i in range(rnfl_mask.shape[0]):
            for j in range(rnfl_mask.shape[1]):
                slice_vals = rnfl_mask[i, j, :]
                if np.any(slice_vals):
                    nonzero = np.where(slice_vals)[0]
                    top_surface[i, j] = nonzero[0]
                    bottom_surface[i, j] = nonzero[-1]

        valid = (top_surface >= 0) & (bottom_surface >= 0)
        thickness_voxels = np.zeros_like(top_surface, dtype=np.float32)
        thickness_voxels[valid] = bottom_surface[valid] - top_surface[valid] + 1
        physical_thickness = thickness_voxels * voxel_spacing[axis] * 1000

        return physical_thickness.astype(np.float32)

    else:
        raise ValueError(f"Unknown thickness extraction method: {method}")


def create_thickness_map_data(
    thickness_map: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    mask: Optional[np.ndarray] = None
) -> ThicknessMapData:
    if mask is not None:
        thickness_for_stats = thickness_map[mask > 0]
    else:
        thickness_for_stats = thickness_map[thickness_map > 0]

    if thickness_for_stats.size == 0:
        thickness_for_stats = np.array([0.0])

    h, w = thickness_map.shape
    spacing_x, spacing_y = voxel_spacing[0] * 1000, voxel_spacing[1] * 1000

    coords_x = np.arange(w) * spacing_x
    coords_y = np.arange(h) * spacing_y

    return ThicknessMapData(
        thickness_values=thickness_map.tolist(),
        coordinates_x=coords_x.tolist(),
        coordinates_y=coords_y.tolist(),
        min_thickness=float(np.min(thickness_for_stats)),
        max_thickness=float(np.max(thickness_for_stats)),
        mean_thickness=float(np.mean(thickness_for_stats)),
        median_thickness=float(np.median(thickness_for_stats)),
        std_thickness=float(np.std(thickness_for_stats))
    )


def detect_defect_regions(
    thickness_map: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    warning_threshold: Optional[float] = None,
    danger_threshold: Optional[float] = None,
    min_region_size_pixels: int = 10
) -> List[DefectRegion]:
    settings = get_settings()
    warning_threshold = warning_threshold or settings.rnfl_thickness_warning_threshold
    danger_threshold = danger_threshold or settings.rnfl_thickness_danger_threshold

    spacing_xy = voxel_spacing[0] * voxel_spacing[1] * 100

    danger_mask = (thickness_map > 0) & (thickness_map < danger_threshold)
    warning_mask = (thickness_map > 0) & (thickness_map >= danger_threshold) & (thickness_map < warning_threshold)

    labeled_danger, num_danger = measure.label(danger_mask, connectivity=2, return_num=True)
    labeled_warning, num_warning = measure.label(warning_mask, connectivity=2, return_num=True)

    defect_regions = []
    region_id = 0

    for i in range(1, num_danger + 1):
        region_mask = (labeled_danger == i)
        region_size = int(np.sum(region_mask))

        if region_size < min_region_size_pixels:
            continue

        region_id += 1
        coords = np.where(region_mask)
        cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
        x1, y1 = int(np.min(coords[1])), int(np.min(coords[0]))
        x2, y2 = int(np.max(coords[1])), int(np.max(coords[0]))

        region_thicknesses = thickness_map[region_mask]

        defect_regions.append(DefectRegion(
            region_id=region_id,
            severity=HealthStatus.DANGER,
            center=(cx, cy),
            bounding_box=(x1, y1, x2, y2),
            area_pixels=region_size,
            area_mm2=float(region_size * spacing_xy),
            mean_thickness=float(np.mean(region_thicknesses)),
            min_thickness=float(np.min(region_thicknesses))
        ))

    for i in range(1, num_warning + 1):
        region_mask = (labeled_warning == i)
        region_size = int(np.sum(region_mask))

        if region_size < min_region_size_pixels:
            continue

        overlap_with_danger = np.any(danger_mask & region_mask)
        if overlap_with_danger:
            continue

        region_id += 1
        coords = np.where(region_mask)
        cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
        x1, y1 = int(np.min(coords[1])), int(np.min(coords[0]))
        x2, y2 = int(np.max(coords[1])), int(np.max(coords[0]))

        region_thicknesses = thickness_map[region_mask]

        defect_regions.append(DefectRegion(
            region_id=region_id,
            severity=HealthStatus.WARNING,
            center=(cx, cy),
            bounding_box=(x1, y1, x2, y2),
            area_pixels=region_size,
            area_mm2=float(region_size * spacing_xy),
            mean_thickness=float(np.mean(region_thicknesses)),
            min_thickness=float(np.min(region_thicknesses))
        ))

    defect_regions.sort(key=lambda r: (r.severity.value, -r.area_mm2))
    return defect_regions


def compute_segmentation_statistics(
    segmentation_mask: np.ndarray,
    thickness_map: Optional[np.ndarray],
    voxel_spacing: Tuple[float, float, float],
    defect_regions: Optional[List[DefectRegion]] = None
) -> Dict[str, float]:
    voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]

    rnfl_voxels = int(np.sum(segmentation_mask > 0))
    total_voxels = segmentation_mask.size

    stats = {
        "rnfl_volume_voxels": float(rnfl_voxels),
        "rnfl_volume_mm3": float(rnfl_voxels * voxel_volume),
        "total_volume_voxels": float(total_voxels),
        "rnfl_fraction": float(rnfl_voxels / total_voxels) if total_voxels > 0 else 0.0
    }

    if thickness_map is not None:
        valid_thickness = thickness_map[thickness_map > 0]
        if valid_thickness.size > 0:
            stats.update({
                "thickness_min": float(np.min(valid_thickness)),
                "thickness_max": float(np.max(valid_thickness)),
                "thickness_mean": float(np.mean(valid_thickness)),
                "thickness_median": float(np.median(valid_thickness)),
                "thickness_std": float(np.std(valid_thickness)),
                "thickness_p10": float(np.percentile(valid_thickness, 10)),
                "thickness_p25": float(np.percentile(valid_thickness, 25)),
                "thickness_p75": float(np.percentile(valid_thickness, 75)),
                "thickness_p90": float(np.percentile(valid_thickness, 90))
            })

    if defect_regions:
        danger_count = sum(1 for r in defect_regions if r.severity == HealthStatus.DANGER)
        warning_count = sum(1 for r in defect_regions if r.severity == HealthStatus.WARNING)
        total_defect_area = sum(r.area_mm2 for r in defect_regions)
        danger_area = sum(r.area_mm2 for r in defect_regions if r.severity == HealthStatus.DANGER)

        stats.update({
            "defect_region_count": float(len(defect_regions)),
            "danger_region_count": float(danger_count),
            "warning_region_count": float(warning_count),
            "total_defect_area_mm2": float(total_defect_area),
            "danger_area_mm2": float(danger_area)
        })

    return stats


def determine_overall_health(
    thickness_map: np.ndarray,
    defect_regions: Optional[List[DefectRegion]] = None,
    warning_threshold: Optional[float] = None,
    danger_threshold: Optional[float] = None,
    normal_min: Optional[float] = None
) -> Tuple[HealthStatus, float]:
    settings = get_settings()
    warning_threshold = warning_threshold or settings.rnfl_thickness_warning_threshold
    danger_threshold = danger_threshold or settings.rnfl_thickness_danger_threshold
    normal_min = normal_min or settings.rnfl_normal_thickness_min

    valid_thickness = thickness_map[thickness_map > 0]

    if valid_thickness.size == 0:
        return HealthStatus.UNKNOWN, 0.0

    mean_thickness = float(np.mean(valid_thickness))
    min_thickness = float(np.min(valid_thickness))
    below_danger = float(np.sum(valid_thickness < danger_threshold)) / valid_thickness.size
    below_warning = float(np.sum(valid_thickness < warning_threshold)) / valid_thickness.size

    danger_regions = []
    warning_regions = []
    if defect_regions:
        danger_regions = [r for r in defect_regions if r.severity == HealthStatus.DANGER]
        warning_regions = [r for r in defect_regions if r.severity == HealthStatus.WARNING]

    total_defect_area = sum(r.area_mm2 for r in defect_regions) if defect_regions else 0.0
    total_rnfl_area = float(np.sum(thickness_map > 0)) * settings.input_volume_size[0] * settings.input_volume_size[1] / 100
    defect_fraction = total_defect_area / total_rnfl_area if total_rnfl_area > 0 else 0.0

    has_large_danger = any(r.area_mm2 > 0.5 for r in danger_regions)
    has_multiple_danger = len(danger_regions) >= 2
    has_significant_thinning = min_thickness < danger_threshold

    confidence = 0.0

    if has_large_danger or has_multiple_danger or (has_significant_thinning and below_danger > 0.1):
        health = HealthStatus.DANGER
        confidence = 0.7 + min(0.3, below_danger * 0.5 + defect_fraction * 0.5)
    elif len(danger_regions) >= 1 or below_danger > 0.05 or below_warning > 0.2:
        health = HealthStatus.WARNING
        confidence = 0.6 + min(0.3, below_warning * 0.3 + defect_fraction * 0.3)
    elif mean_thickness >= normal_min and below_warning < 0.05:
        health = HealthStatus.NORMAL
        confidence = 0.8 + min(0.2, (mean_thickness - normal_min) / 40.0)
    else:
        health = HealthStatus.UNKNOWN
        confidence = 0.5

    confidence = max(0.0, min(1.0, confidence))
    return health, confidence


def refine_segmentation(
    probability_map: np.ndarray,
    threshold: float = 0.5,
    min_object_size: int = 100,
    closing_radius: int = 2
) -> np.ndarray:
    binary_mask = (probability_map > threshold).astype(np.uint8)

    if closing_radius > 0:
        struct = morphology.ball(closing_radius)
        binary_mask = morphology.closing(binary_mask, struct)

    if min_object_size > 0:
        binary_mask = morphology.remove_small_objects(
            binary_mask.astype(bool),
            min_size=min_object_size,
            connectivity=2
        ).astype(np.uint8)

    if closing_radius > 0:
        struct = morphology.ball(closing_radius // 2)
        binary_mask = morphology.opening(binary_mask, struct)

    return binary_mask


def resize_to_original(
    mask: np.ndarray,
    original_shape: Tuple[int, int, int],
    order: int = 0
) -> np.ndarray:
    if mask.shape == original_shape:
        return mask

    factors = tuple(o / s for o, s in zip(original_shape, mask.shape))
    resized = ndimage.zoom(mask, factors, order=order, mode="nearest")

    if resized.shape != original_shape:
        resized = resized[:original_shape[0], :original_shape[1], :original_shape[2]]
        pad_width = [(0, max(0, o - r)) for o, r in zip(original_shape, resized.shape)]
        if any(p[1] > 0 for p in pad_width):
            resized = np.pad(resized, pad_width, mode="constant", constant_values=0)

    return resized.astype(mask.dtype)
