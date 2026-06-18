from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from skimage import filters, restoration

from ..config import get_settings
from ..utils import get_logger
from .image_loader import VolumeInfo
from .resampler import (
    PhysicalSpaceResampler,
    ResampleResult,
    ResampleTransform,
    get_physical_resampler
)

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    volume: np.ndarray
    original_shape: Tuple[int, int, int]
    target_shape: Tuple[int, int, int]
    original_voxel_spacing: Tuple[float, float, float]
    resampled_voxel_spacing: Tuple[float, float, float]
    normalization_params: Tuple[float, float]
    resample_transform: ResampleTransform = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)


def resize_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    order: int = 1,
    preserve_range: bool = True
) -> np.ndarray:
    if volume.shape == target_shape:
        return volume

    factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    resized = ndimage.zoom(volume, factors, order=order, mode="reflect")

    if resized.shape != target_shape:
        resized = resized[:target_shape[0], :target_shape[1], :target_shape[2]]
        pad_width = [(0, max(0, t - r)) for t, r in zip(target_shape, resized.shape)]
        if any(p[1] > 0 for p in pad_width):
            resized = np.pad(resized, pad_width, mode="constant", constant_values=0)

    if preserve_range:
        original_min, original_max = volume.min(), volume.max()
        if original_max > original_min:
            resized = np.clip(resized, original_min, original_max)

    return resized.astype(volume.dtype)


def normalize_volume(
    volume: np.ndarray,
    method: str = "zscore",
    percentile_range: Tuple[float, float] = (0.5, 99.5),
    output_range: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[np.ndarray, Tuple[float, float]]:
    volume = volume.astype(np.float32)

    if method == "zscore":
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 1e-8:
            normalized = (volume - mean) / std
        else:
            normalized = volume - mean
        params = (mean, std)

    elif method == "minmax":
        p_min, p_max = np.percentile(volume, percentile_range)
        if p_max - p_min > 1e-8:
            normalized = (volume - p_min) / (p_max - p_min)
            normalized = normalized * (output_range[1] - output_range[0]) + output_range[0]
            normalized = np.clip(normalized, output_range[0], output_range[1])
        else:
            normalized = np.full_like(volume, (output_range[0] + output_range[1]) / 2)
        params = (p_min, p_max)

    elif method == "percentile":
        p_low, p_high = np.percentile(volume, percentile_range)
        normalized = np.clip(volume, p_low, p_high)
        if p_high - p_low > 1e-8:
            normalized = (normalized - p_low) / (p_high - p_low)
        params = (p_low, p_high)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32), params


def denoise_volume(
    volume: np.ndarray,
    method: str = "gaussian",
    sigma: float = 1.0,
    **kwargs
) -> np.ndarray:
    if method == "gaussian":
        denoised = ndimage.gaussian_filter(volume, sigma=sigma)
    elif method == "median":
        size = kwargs.get("size", 3)
        denoised = filters.median(volume, np.ones((size, size, size)))
    elif method == "bilateral":
        denoised = restoration.denoise_bilateral(
            volume,
            sigma_color=kwargs.get("sigma_color", 0.05),
            sigma_spatial=kwargs.get("sigma_spatial", 1.0),
            channel_axis=None
        )
    elif method == "wavelet":
        denoised = restoration.denoise_wavelet(
            volume,
            sigma=sigma,
            channel_axis=None
        )
    elif method == "none":
        denoised = volume
    else:
        raise ValueError(f"Unknown denoising method: {method}")

    return denoised.astype(volume.dtype)


def bias_correction(
    volume: np.ndarray,
    shrink_factor: int = 4,
    convergence_threshold: float = 1e-6,
    max_iterations: int = 50
) -> np.ndarray:
    try:
        import SimpleITK as sitk

        sitk_image = sitk.GetImageFromArray(volume)
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetConvergenceThreshold(convergence_threshold)
        corrector.SetMaximumNumberOfIterations([max_iterations] * 3)

        try:
            shrink = sitk.ShrinkImageFilter()
            shrink.SetShrinkFactors([shrink_factor] * 3)
            downsampled = shrink.Execute(sitk_image)
            corrected = corrector.Execute(downsampled)

            expand = sitk.ResampleImageFilter()
            expand.SetSize(sitk_image.GetSize())
            expand.SetOutputSpacing(sitk_image.GetSpacing())
            expand.SetOutputOrigin(sitk_image.GetOrigin())
            expand.SetOutputDirection(sitk_image.GetDirection())
            expand.SetInterpolator(sitk.sitkLinear)
            full_corrected = expand.Execute(corrected)
        except Exception:
            full_corrected = corrector.Execute(sitk_image)

        corrected_array = sitk.GetArrayFromImage(full_corrected)
        return corrected_array.astype(volume.dtype)

    except ImportError:
        logger.warning("SimpleITK not available for bias correction, skipping")
        return volume
    except Exception as e:
        logger.warning(f"Bias correction failed: {e}, skipping")
        return volume


def estimate_noise(volume: np.ndarray) -> float:
    gradient_magnitude = np.sqrt(
        np.sum(np.array(np.gradient(volume)) ** 2, axis=0)
    )
    return float(np.median(np.abs(gradient_magnitude - np.median(gradient_magnitude))) / 0.6745)


class OCTPreprocessor:
    def __init__(
        self,
        target_shape: Optional[Tuple[int, int, int]] = None,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        normalization_method: str = "minmax",
        denoising_method: str = "gaussian",
        denoising_sigma: float = 0.8,
        apply_bias_correction: bool = True,
        use_physical_resampling: bool = True,
        enable_pre_crop: bool = True
    ):
        settings = get_settings()
        self.target_shape = target_shape or settings.input_volume_size
        self.target_spacing = tuple(
            target_spacing or PhysicalSpaceResampler.DEFAULT_TARGET_SPACING
        )
        self.normalization_method = normalization_method
        self.denoising_method = denoising_method
        self.denoising_sigma = denoising_sigma
        self.apply_bias_correction = apply_bias_correction
        self.use_physical_resampling = use_physical_resampling
        self.enable_pre_crop = enable_pre_crop

        self._resampler: Optional[PhysicalSpaceResampler] = None

    def _get_resampler(self) -> PhysicalSpaceResampler:
        if self._resampler is None:
            self._resampler = get_physical_resampler(
                target_shape=self.target_shape,
                target_spacing=self.target_spacing,
                enable_pre_crop=self.enable_pre_crop,
                spacing_tolerance=0.05,
                resample_order=1,
                mask_resample_order=0,
                padding_mode="edge",
                align_strategy="center"
            )
        return self._resampler

    def preprocess(
        self,
        volume: np.ndarray,
        volume_info: VolumeInfo
    ) -> PreprocessingResult:
        logger.info(
            f"Starting preprocessing: shape={volume.shape}, "
            f"spacing={volume_info.voxel_spacing} mm"
        )

        original_shape = volume.shape
        original_spacing = volume_info.voxel_spacing

        self._validate_input_volume(volume, original_spacing)

        processed = volume.astype(np.float32)
        metadata = {"steps": []}

        if self.apply_bias_correction:
            logger.info("Step 1/5: Applying N4 bias field correction")
            processed = bias_correction(processed)
            metadata["steps"].append("bias_correction")

        noise_level = estimate_noise(processed)
        logger.info(f"Estimated noise level: {noise_level:.4f}")

        if noise_level > 0.01:
            logger.info(f"Step 2/5: Applying {self.denoising_method} denoising (sigma={self.denoising_sigma})")
            processed = denoise_volume(
                processed,
                method=self.denoising_method,
                sigma=self.denoising_sigma
            )
            metadata["steps"].append(f"denoise:{self.denoising_method}")
        else:
            logger.info("Step 2/5: Skipping denoising (noise level acceptable)")
            metadata["steps"].append("denoise:skipped")

        if self.use_physical_resampling:
            logger.info(
                f"Step 3/5: Physical-space adaptive resampling "
                f"({original_shape}@{original_spacing} → "
                f"{self.target_shape}@{self.target_spacing})"
            )
            resampler = self._get_resampler()
            resample_result: ResampleResult = resampler.resample(
                processed,
                original_spacing=original_spacing,
                is_mask=False
            )
            processed = resample_result.volume
            resample_transform = resample_result.transform
            metadata["steps"].append("physical_resample")
            metadata["resample_scale_factors"] = resample_transform.resample_scale_factors
            metadata["padding"] = resample_transform.padding_amounts
            metadata["cropping"] = resample_transform.cropping_slices
        else:
            logger.info(f"Step 3/5: Simple resize (no physical alignment) → {self.target_shape}")
            processed = resize_volume(processed, self.target_shape, order=1)
            resample_transform = None
            metadata["steps"].append("simple_resize")

        logger.info(f"Step 4/5: {self.normalization_method} normalization → [-1, 1]")
        processed, norm_params = normalize_volume(
            processed,
            method=self.normalization_method,
            percentile_range=(0.5, 99.5),
            output_range=(-1.0, 1.0)
        )
        metadata["steps"].append(f"normalize:{self.normalization_method}")

        logger.info(f"Step 5/5: Shape validation")
        if processed.shape != self.target_shape:
            logger.warning(
                f"Unexpected shape {processed.shape}, forcing to {self.target_shape}"
            )
            processed = PhysicalSpaceResampler._force_shape(
                processed, self.target_shape, order=1
            )

        metadata["final_shape"] = processed.shape
        metadata["final_spacing"] = self.target_spacing
        metadata["steps"].append("complete")

        logger.info(
            f"Preprocessing complete: {original_shape}@{original_spacing} → "
            f"{processed.shape}@{self.target_spacing} | "
            f"steps: {'→'.join(metadata['steps'])}"
        )

        return PreprocessingResult(
            volume=processed,
            original_shape=original_shape,
            target_shape=self.target_shape,
            original_voxel_spacing=original_spacing,
            resampled_voxel_spacing=self.target_spacing,
            normalization_params=norm_params,
            resample_transform=resample_transform,
            metadata=metadata
        )

    def restore_mask_to_original_space(
        self,
        mask_in_network_space: np.ndarray,
        preprocess_result: PreprocessingResult,
        is_probability_map: bool = False
    ) -> np.ndarray:
        """
        将神经网络输出的分割掩码/概率图逆变换回原始物理空间。
        这是保证厚度计算和缺陷定位在临床坐标系下准确的关键步骤。
        """
        if mask_in_network_space.shape != self.target_shape:
            raise ValueError(
                f"Mask shape {mask_in_network_space.shape} does not match "
                f"target shape {self.target_shape}"
            )

        transform = preprocess_result.resample_transform
        original_spacing = preprocess_result.original_voxel_spacing

        if transform is None or not self.use_physical_resampling:
            logger.warning(
                "No resample transform available, using simple resize (geometry may be distorted)"
            )
            return resize_volume(
                mask_in_network_space,
                preprocess_result.original_shape,
                order=0 if (not is_probability_map and np.issubdtype(mask_in_network_space.dtype, np.unsignedinteger))
                else 1
            )

        resampler = self._get_resampler()
        restored = resampler.inverse_transform(
            mask_in_network_space,
            transform,
            is_mask=(not is_probability_map)
        )

        logger.debug(
            f"Restored mask: {mask_in_network_space.shape} → {restored.shape} "
            f"(spacing {self.target_spacing} → {original_spacing})"
        )
        return restored

    def to_tensor(self, preprocessed: PreprocessingResult) -> np.ndarray:
        tensor = preprocessed.volume[np.newaxis, np.newaxis, ...]
        return tensor.astype(np.float32)

    @staticmethod
    def _validate_input_volume(
        volume: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> None:
        if volume.ndim != 3:
            raise ValueError(
                f"Input must be 3D OCT volume, got {volume.ndim}D with shape {volume.shape}"
            )

        for axis, sz in enumerate(volume.shape):
            if sz < 4:
                raise ValueError(
                    f"Volume too small on axis {axis}: size={sz} (min=4). "
                    f"Full shape: {volume.shape}"
                )

        for axis, sp in enumerate(spacing):
            if sp <= 0:
                logger.warning(
                    f"Invalid voxel spacing on axis {axis}: {sp} mm, "
                    f"using default 0.005 mm"
                )
            elif sp > 1.0:
                logger.warning(
                    f"Suspiciously large voxel spacing on axis {axis}: {sp} mm "
                    f"(likely wrong units, should be mm not meters)"
                )
            elif sp < 1e-6:
                logger.warning(
                    f"Suspiciously small voxel spacing on axis {axis}: {sp} mm "
                    f"(likely wrong units)"
                )


_default_preprocessor = None


def get_default_preprocessor() -> OCTPreprocessor:
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = OCTPreprocessor()
    return _default_preprocessor


def preprocess_oct_volume(
    volume: np.ndarray,
    volume_info: VolumeInfo,
    target_shape: Optional[Tuple[int, int, int]] = None
) -> PreprocessingResult:
    preprocessor = OCTPreprocessor(target_shape=target_shape)
    return preprocessor.preprocess(volume, volume_info)
