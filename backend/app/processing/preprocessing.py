from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from skimage import filters, restoration

from ..config import get_settings
from ..utils import get_logger
from .image_loader import VolumeInfo

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    volume: np.ndarray
    original_shape: Tuple[int, int, int]
    target_shape: Tuple[int, int, int]
    voxel_spacing: Tuple[float, float, float]
    normalization_params: Tuple[float, float]


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
        normalization_method: str = "minmax",
        denoising_method: str = "gaussian",
        denoising_sigma: float = 0.8,
        apply_bias_correction: bool = True
    ):
        settings = get_settings()
        self.target_shape = target_shape or settings.input_volume_size
        self.normalization_method = normalization_method
        self.denoising_method = denoising_method
        self.denoising_sigma = denoising_sigma
        self.apply_bias_correction = apply_bias_correction

    def preprocess(
        self,
        volume: np.ndarray,
        volume_info: VolumeInfo
    ) -> PreprocessingResult:
        logger.info(f"Starting preprocessing: original shape={volume.shape}")

        original_shape = volume.shape
        voxel_spacing = volume_info.voxel_spacing

        processed = volume.astype(np.float32)

        if self.apply_bias_correction:
            logger.info("Applying N4 bias field correction")
            processed = bias_correction(processed)

        noise_level = estimate_noise(processed)
        logger.info(f"Estimated noise level: {noise_level:.4f}")

        if noise_level > 0.01:
            logger.info(f"Applying {self.denoising_method} denoising with sigma={self.denoising_sigma}")
            processed = denoise_volume(
                processed,
                method=self.denoising_method,
                sigma=self.denoising_sigma
            )

        logger.info(f"Resizing to target shape: {self.target_shape}")
        processed = resize_volume(processed, self.target_shape, order=1)

        logger.info(f"Applying {self.normalization_method} normalization")
        processed, norm_params = normalize_volume(
            processed,
            method=self.normalization_method,
            percentile_range=(0.5, 99.5),
            output_range=(-1.0, 1.0)
        )

        logger.info(f"Preprocessing complete: output shape={processed.shape}")

        return PreprocessingResult(
            volume=processed,
            original_shape=original_shape,
            target_shape=self.target_shape,
            voxel_spacing=voxel_spacing,
            normalization_params=norm_params
        )

    def to_tensor(self, preprocessed: PreprocessingResult) -> np.ndarray:
        tensor = preprocessed.volume[np.newaxis, np.newaxis, ...]
        return tensor.astype(np.float32)


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
