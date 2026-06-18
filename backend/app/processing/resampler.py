import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ResampleTransform:
    original_shape: Tuple[int, int, int]
    original_spacing: Tuple[float, float, float]
    target_spacing: Tuple[float, float, float]

    after_resample_shape: Tuple[int, int, int]
    resample_scale_factors: Tuple[float, float, float]

    padding_amounts: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    cropping_slices: Tuple[slice, slice, slice]

    final_shape: Tuple[int, int, int]
    final_origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    metadata: dict = field(default_factory=dict)


@dataclass
class ResampleResult:
    volume: np.ndarray
    transform: ResampleTransform


class PhysicalSpaceResampler:
    """
    物理空间自适应重采样器（解决不同设备 OCT 数据的形状/分辨率不兼容问题）

    核心设计原则：
    1. 先重采样到目标物理间距（保证 RNFL 厚度等几何特征在物理空间不失真）
    2. 再进行 Padding/Cropping 对齐到网络标准输入尺寸
    3. 完整记录逆变换参数，确保分割结果能精确映射回原始空间
    """

    DEFAULT_TARGET_SPACING = (0.005, 0.005, 0.0035)

    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (128, 128, 64),
        target_spacing: Optional[Tuple[float, float, float]] = None,
        spacing_tolerance: float = 0.05,
        resample_order: int = 1,
        mask_resample_order: int = 0,
        padding_mode: str = "edge",
        padding_constant_value: float = 0.0,
        align_strategy: str = "center",
        enable_pre_crop: bool = True,
        pre_crop_background_threshold: float = 1e-6
    ):
        self.target_shape = target_shape
        self.target_spacing = tuple(target_spacing or self.DEFAULT_TARGET_SPACING)
        self.spacing_tolerance = spacing_tolerance
        self.resample_order = resample_order
        self.mask_resample_order = mask_resample_order
        self.padding_mode = padding_mode
        self.padding_constant_value = padding_constant_value
        self.align_strategy = align_strategy
        self.enable_pre_crop = enable_pre_crop
        self.pre_crop_background_threshold = pre_crop_background_threshold

        if len(target_shape) != 3:
            raise ValueError(f"target_shape must be 3D, got {len(target_shape)}D")
        if len(self.target_spacing) != 3:
            raise ValueError(f"target_spacing must be 3D, got {len(self.target_spacing)}D")

    def _needs_resampling(
        self, original_spacing: Tuple[float, float, float]
    ) -> bool:
        for orig, target in zip(original_spacing, self.target_spacing):
            if orig <= 0 or target <= 0:
                return True
            rel_error = abs(orig - target) / target
            if rel_error > self.spacing_tolerance:
                return True
        return False

    MAX_SCALE_FACTOR = 8.0
    MIN_SCALE_FACTOR = 1.0 / 8.0
    MAX_DIM_SIZE = 1024
    MAX_TOTAL_VOXELS = 256 * 256 * 256

    def _calculate_resample_shape(
        self,
        original_shape: Tuple[int, int, int],
        original_spacing: Tuple[float, float, float]
    ) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
        raw_ratios = []
        for orig_sp, tgt_sp in zip(original_spacing, self.target_spacing):
            if orig_sp <= 0 or tgt_sp <= 0:
                raw_ratios.append(1.0)
            else:
                raw_ratios.append(orig_sp / tgt_sp)

        global_scale_cap = 1.0
        for rs, orig_size in zip(raw_ratios, original_shape):
            proposed = orig_size * rs
            if proposed > self.MAX_DIM_SIZE:
                global_scale_cap = min(global_scale_cap, self.MAX_DIM_SIZE / proposed)

        proposed_shape = tuple(
            max(1, int(round(orig_size * rr * global_scale_cap)))
            for orig_size, rr in zip(original_shape, raw_ratios)
        )
        total_voxels = float(np.prod(proposed_shape))
        if total_voxels > self.MAX_TOTAL_VOXELS:
            volume_scale = (self.MAX_TOTAL_VOXELS / total_voxels) ** (1.0 / 3.0)
            proposed_shape = tuple(
                max(1, int(round(s * volume_scale))) for s in proposed_shape
            )
            global_scale_cap *= volume_scale

        new_shape = []
        scale_factors = []
        for i, (orig_size, orig_sp, tgt_sp) in enumerate(
            zip(original_shape, original_spacing, self.target_spacing)
        ):
            if orig_sp <= 0 or tgt_sp <= 0:
                new_shape.append(orig_size)
                scale_factors.append(1.0)
                continue

            ratio = orig_sp / tgt_sp
            if ratio > self.MAX_SCALE_FACTOR or ratio < self.MIN_SCALE_FACTOR:
                logger.warning(
                    f"Axis {i}: scale factor {ratio:.4f} outside safe range "
                    f"[{self.MIN_SCALE_FACTOR:.4f}, {self.MAX_SCALE_FACTOR:.2f}] "
                    f"(orig_spacing={orig_sp}, target={tgt_sp} mm). "
                    f"Likely unit mismatch. Using direct shape alignment."
                )
                new_size = proposed_shape[i]
            else:
                new_size = proposed_shape[i]

            new_size = max(1, min(new_size, self.MAX_DIM_SIZE))
            scale_factor = new_size / orig_size
            new_shape.append(new_size)
            scale_factors.append(scale_factor)

        return tuple(new_shape), tuple(scale_factors)

    def _pre_crop_foreground(
        self,
        volume: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
        if not self.enable_pre_crop:
            slices = tuple(slice(0, s) for s in volume.shape)
            return volume, slices

        threshold = self.pre_crop_background_threshold
        if np.issubdtype(volume.dtype, np.integer):
            foreground_mask = volume > 0
        else:
            foreground_mask = np.abs(volume) > threshold

        if not np.any(foreground_mask):
            slices = tuple(slice(0, s) for s in volume.shape)
            return volume, slices

        any_axis_0 = np.any(foreground_mask, axis=(1, 2))
        any_axis_1 = np.any(foreground_mask, axis=(0, 2))
        any_axis_2 = np.any(foreground_mask, axis=(0, 1))

        indices_0 = np.where(any_axis_0)[0]
        indices_1 = np.where(any_axis_1)[0]
        indices_2 = np.where(any_axis_2)[0]

        margin = 2
        s0 = slice(max(0, indices_0[0] - margin), min(volume.shape[0], indices_0[-1] + margin + 1))
        s1 = slice(max(0, indices_1[0] - margin), min(volume.shape[1], indices_1[-1] + margin + 1))
        s2 = slice(max(0, indices_2[0] - margin), min(volume.shape[2], indices_2[-1] + margin + 1))

        crop_slices = (s0, s1, s2)
        cropped = volume[crop_slices]

        logger.info(
            f"Pre-crop: {volume.shape} → {cropped.shape} "
            f"(saved {1 - np.prod(cropped.shape) / np.prod(volume.shape):.1%} space)"
        )
        return cropped, crop_slices

    def _calculate_padding_cropping(
        self,
        current_shape: Tuple[int, int, int]
    ) -> Tuple[
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        Tuple[slice, slice, slice]
    ]:
        padding = []
        slices = []

        for curr_size, tgt_size in zip(current_shape, self.target_shape):
            diff = tgt_size - curr_size

            if diff >= 0:
                if self.align_strategy == "center":
                    pad_before = diff // 2
                    pad_after = diff - pad_before
                elif self.align_strategy == "start":
                    pad_before = 0
                    pad_after = diff
                else:
                    pad_before = diff
                    pad_after = 0

                padding.append((pad_before, pad_after))
                slices.append(slice(0, curr_size))
            else:
                crop_amount = -diff
                if self.align_strategy == "center":
                    crop_start = crop_amount // 2
                elif self.align_strategy == "start":
                    crop_start = 0
                else:
                    crop_start = crop_amount
                crop_end = crop_start + tgt_size

                padding.append((0, 0))
                slices.append(slice(crop_start, crop_end))

        return tuple(padding), tuple(slices)

    def resample(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        is_mask: bool = False
    ) -> ResampleResult:
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D with shape {volume.shape}")

        original_shape = volume.shape
        original_dtype = volume.dtype
        work_volume = volume.astype(np.float32) if not is_mask else volume.astype(np.uint8)

        pre_crop_slices = (slice(None), slice(None), slice(None))
        if self.enable_pre_crop and not is_mask:
            work_volume, pre_crop_slices = self._pre_crop_foreground(work_volume)
            logger.debug(f"Pre-crop slices: {pre_crop_slices}")

        order = self.mask_resample_order if is_mask else self.resample_order

        after_resample_shape = work_volume.shape
        resample_scale_factors = (1.0, 1.0, 1.0)
        resampled = work_volume

        if self._needs_resampling(original_spacing):
            after_resample_shape, resample_scale_factors = self._calculate_resample_shape(
                work_volume.shape, original_spacing
            )

            logger.info(
                f"Physical resampling: {work_volume.shape} → {after_resample_shape} "
                f"(spacing: {original_spacing} → {self.target_spacing} mm)"
            )

            resampled = ndimage.zoom(
                work_volume,
                resample_scale_factors,
                order=order,
                mode="reflect"
            )

            if resampled.shape != after_resample_shape:
                resampled = self._force_shape(resampled, after_resample_shape, order)
        else:
            logger.info(f"Skipping resample: spacing within tolerance ({original_spacing})")

        padding, crop_slices = self._calculate_padding_cropping(resampled.shape)

        if any(sum(p) > 0 for p in padding):
            if is_mask or self.padding_mode == "constant":
                pad_val = 0 if is_mask else self.padding_constant_value
                aligned = np.pad(
                    resampled,
                    padding,
                    mode="constant",
                    constant_values=pad_val
                )
            else:
                aligned = np.pad(resampled, padding, mode=self.padding_mode)
            logger.info(f"Padding applied: {padding} → shape={aligned.shape}")
        else:
            aligned = resampled

        final_volume = aligned
        if any(s != slice(0, s) for s, sz in zip(crop_slices, aligned.shape)):
            final_volume = aligned[crop_slices]
            logger.info(f"Cropping applied: {crop_slices} → shape={final_volume.shape}")

        if final_volume.shape != self.target_shape:
            final_volume = self._force_shape(final_volume, self.target_shape, order)
            logger.warning(f"Force-aligned to target shape: {final_volume.shape}")

        if is_mask:
            final_volume = (final_volume > 0.5).astype(np.uint8)
        elif np.issubdtype(original_dtype, np.floating):
            final_volume = final_volume.astype(original_dtype)

        _s0 = pre_crop_slices[0].start if pre_crop_slices[0].start is not None else 0
        _s1 = pre_crop_slices[1].start if pre_crop_slices[1].start is not None else 0
        _s2 = pre_crop_slices[2].start if pre_crop_slices[2].start is not None else 0
        origin_offset = (
            -_s0 * original_spacing[0],
            -_s1 * original_spacing[1],
            -_s2 * original_spacing[2]
        )

        transform = ResampleTransform(
            original_shape=original_shape,
            original_spacing=original_spacing,
            target_spacing=self.target_spacing,
            after_resample_shape=after_resample_shape,
            resample_scale_factors=resample_scale_factors,
            padding_amounts=padding,
            cropping_slices=crop_slices,
            final_shape=final_volume.shape,
            final_origin_offset=origin_offset,
            metadata={
                "pre_crop_slices": pre_crop_slices,
                "is_mask": is_mask,
                "resample_order": order
            }
        )

        return ResampleResult(volume=final_volume, transform=transform)

    @staticmethod
    def _normalize_slice(s: slice, size: int) -> Tuple[int, int]:
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else size
        start = max(0, min(start, size))
        stop = max(0, min(stop, size))
        return start, stop

    def inverse_transform(
        self,
        processed_volume: np.ndarray,
        transform: ResampleTransform,
        is_mask: bool = True
    ) -> np.ndarray:
        if processed_volume.shape != transform.final_shape:
            raise ValueError(
                f"Shape mismatch: expected {transform.final_shape}, got {processed_volume.shape}"
            )

        order = self.mask_resample_order if is_mask else self.resample_order
        original_shape = transform.original_shape

        inverse = processed_volume.astype(np.float32)

        crop_slices = transform.cropping_slices
        padding = transform.padding_amounts
        pre_crop_slices = transform.metadata.get(
            "pre_crop_slices", (slice(None), slice(None), slice(None))
        )

        crop_info = []
        for i in range(3):
            cs = crop_slices[i]
            if isinstance(cs, slice):
                c_start, c_stop = self._normalize_slice(cs, inverse.shape[i] + padding[i][0] + padding[i][1])
            else:
                c_start, c_stop = 0, inverse.shape[i]
            crop_info.append((c_start, c_stop, c_stop - c_start))

        padded_shape_list = []
        for i in range(3):
            sz = crop_info[i][2] + padding[i][0] + padding[i][1]
            padded_shape_list.append(sz)
        padded_shape = tuple(padded_shape_list)
        uncropped = np.zeros(padded_shape, dtype=np.float32)

        inner_slices_list = []
        for i in range(3):
            p_before, p_after = padding[i]
            c_len = crop_info[i][2]
            inner_start = p_before
            inner_stop = min(p_before + c_len, padded_shape[i])
            inner_slices_list.append(slice(inner_start, inner_stop))
        inner_slices = tuple(inner_slices_list)

        copy_slices_list = []
        for i in range(3):
            copy_slices_list.append(slice(0, inner_slices[i].stop - inner_slices[i].start))
        copy_slices = tuple(copy_slices_list)

        uncropped[inner_slices] = inverse[copy_slices]
        inverse = uncropped

        unpadded_list = []
        for i in range(3):
            p_before, p_after = padding[i]
            s = inverse.shape[i]
            unpadded_list.append(slice(p_before, max(p_before, s - p_after)))
        unpadded_slices = tuple(unpadded_list)
        inverse = inverse[unpadded_slices]

        scale_factors = transform.resample_scale_factors
        if any(abs(sf - 1.0) > 1e-6 for sf in scale_factors):
            inverse_scale = tuple(1.0 / sf for sf in scale_factors)
            inverse = ndimage.zoom(inverse, inverse_scale, order=order, mode="reflect")

        if inverse.shape != original_shape and any(abs(sf - 1.0) <= 1e-6 for sf in scale_factors) and inverse.shape != padded_shape_list:
            target_shape_after_resample = transform.after_resample_shape
            if inverse.shape != target_shape_after_resample:
                inverse = self._force_shape(inverse, target_shape_after_resample, order)

        restored = np.zeros(original_shape, dtype=np.float32)

        pc_info = []
        for i in range(3):
            s = pre_crop_slices[i]
            if isinstance(s, slice):
                pc_start, pc_stop = self._normalize_slice(s, original_shape[i])
            else:
                pc_start, pc_stop = 0, original_shape[i]
            pc_info.append((pc_start, pc_stop, pc_stop - pc_start))

        valid_src_list = []
        for i in range(3):
            src_len = min(pc_info[i][2], inverse.shape[i])
            valid_src_list.append(slice(0, src_len))

        dst_start_list = []
        final_src_list = []
        for i in range(3):
            dst_start = pc_info[i][0]
            dst_stop = min(dst_start + valid_src_list[i].stop, original_shape[i])
            dst_start_list.append(slice(dst_start, dst_stop))
            final_src_list.append(slice(0, dst_stop - dst_start))

        final_src = tuple(final_src_list)
        dst_slices = tuple(dst_start_list)

        restored[dst_slices] = inverse[final_src]

        if is_mask:
            restored = (restored > 0.5).astype(np.uint8)

        return restored

    @staticmethod
    def _force_shape(
        volume: np.ndarray,
        target_shape: Tuple[int, int, int],
        order: int
    ) -> np.ndarray:
        factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
        result = ndimage.zoom(volume, factors, order=order, mode="reflect")

        if result.shape != target_shape:
            result = result[:target_shape[0], :target_shape[1], :target_shape[2]]
            pad_widths = []
            for i in range(3):
                diff = target_shape[i] - result.shape[i]
                if diff > 0:
                    pad_before = diff // 2
                    pad_after = diff - pad_before
                    pad_widths.append((pad_before, pad_after))
                else:
                    pad_widths.append((0, 0))
            if any(p[0] > 0 or p[1] > 0 for p in pad_widths):
                result = np.pad(result, pad_widths, mode="edge")

        return result


_resampler_cache: dict = {}


def get_physical_resampler(
    target_shape: Tuple[int, int, int],
    target_spacing: Optional[Tuple[float, float, float]] = None,
    **kwargs
) -> PhysicalSpaceResampler:
    cache_key = (target_shape, tuple(target_spacing or ()))
    if cache_key not in _resampler_cache:
        _resampler_cache[cache_key] = PhysicalSpaceResampler(
            target_shape=target_shape,
            target_spacing=target_spacing,
            **kwargs
        )
    return _resampler_cache[cache_key]
