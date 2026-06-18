import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import numpy as np
import pytest
import torch


def test_imports():
    from app.config import get_settings
    from app.models import create_resnet3d_model, HealthStatus
    from app.processing import (
        resize_volume,
        normalize_volume,
        extract_thickness_map,
        detect_defect_regions
    )
    from app.utils import get_device, validate_file_extension

    assert get_settings is not None
    assert create_resnet3d_model is not None
    assert HealthStatus is not None
    assert resize_volume is not None
    assert normalize_volume is not None
    assert extract_thickness_map is not None
    assert detect_defect_regions is not None
    assert get_device is not None
    assert validate_file_extension is not None


def test_settings():
    from app.config import get_settings
    settings = get_settings()
    assert settings.app_name == "Glaucoma OCT AI Platform"
    assert settings.num_classes == 2
    assert settings.in_channels == 1
    assert len(settings.input_volume_size) == 3


def test_file_extension_validation():
    from app.utils import validate_file_extension

    assert validate_file_extension("test.nii") is True
    assert validate_file_extension("test.nii.gz") is True
    assert validate_file_extension("test.mha") is True
    assert validate_file_extension("test.mhd") is True
    assert validate_file_extension("test.dcm") is False
    assert validate_file_extension("test.png") is False


def test_resize_volume():
    from app.processing import resize_volume

    volume = np.random.rand(64, 64, 32).astype(np.float32)
    target_shape = (128, 128, 64)
    resized = resize_volume(volume, target_shape, order=1)

    assert resized.shape == target_shape
    assert resized.dtype == np.float32


def test_normalize_volume():
    from app.processing import normalize_volume

    volume = np.random.rand(64, 64, 32).astype(np.float32) * 100 + 50
    normalized, params = normalize_volume(volume, method="minmax", output_range=(-1.0, 1.0))

    assert normalized.shape == volume.shape
    assert normalized.min() >= -1.0
    assert normalized.max() <= 1.0
    assert len(params) == 2


def test_extract_thickness_map():
    from app.processing import extract_thickness_map

    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    mask[10:20, 10:20, 5:15] = 1

    voxel_spacing = (0.01, 0.01, 0.01)
    thickness_map = extract_thickness_map(mask, voxel_spacing, axis=2)

    assert thickness_map.shape == (64, 64)
    assert thickness_map.dtype == np.float32

    expected_thickness = 10 * 0.01 * 1000
    assert np.abs(thickness_map[15, 15] - expected_thickness) < 1.0


def test_detect_defect_regions():
    from app.processing import detect_defect_regions
    from app.models import HealthStatus

    thickness_map = np.full((64, 64), 100.0, dtype=np.float32)
    thickness_map[20:30, 20:30] = 40.0
    thickness_map[40:45, 40:45] = 60.0
    voxel_spacing = (0.01, 0.01, 0.01)

    defects = detect_defect_regions(
        thickness_map,
        voxel_spacing,
        warning_threshold=70.0,
        danger_threshold=50.0,
        min_region_size_pixels=5
    )

    assert len(defects) >= 2
    severity_counts = {d.severity: 0 for d in defects}
    for d in defects:
        severity_counts[d.severity] += 1

    assert severity_counts.get(HealthStatus.DANGER, 0) >= 1
    assert severity_counts.get(HealthStatus.WARNING, 0) >= 1


def test_resnet3d_model_creation():
    from app.models import create_resnet3d_model

    model = create_resnet3d_model(
        in_channels=1,
        num_classes=2,
        model_name="resnet3d_18"
    )

    assert model is not None
    assert model.get_parameters_count() > 0

    input_tensor = torch.randn(1, 1, 64, 64, 32)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (1, 2, 64, 64, 32)


def test_resnet3d_prediction():
    from app.models import create_resnet3d_model

    model = create_resnet3d_model(
        in_channels=1,
        num_classes=2,
        model_name="resnet3d_18"
    )

    input_tensor = torch.randn(1, 1, 64, 64, 32)
    model.eval()
    with torch.no_grad():
        prediction = model.predict(input_tensor)

    assert prediction.shape == (1, 2, 64, 64, 32)
    assert torch.all(prediction >= 0.0)
    assert torch.all(prediction <= 1.0)


def test_health_status_enum():
    from app.models import HealthStatus

    assert HealthStatus.NORMAL.value == "normal"
    assert HealthStatus.WARNING.value == "warning"
    assert HealthStatus.DANGER.value == "danger"
    assert HealthStatus.UNKNOWN.value == "unknown"


def test_physical_resampler_import():
    from app.processing.resampler import (
        PhysicalSpaceResampler,
        ResampleTransform,
        ResampleResult,
        get_physical_resampler
    )
    assert PhysicalSpaceResampler is not None
    assert ResampleTransform is not None
    assert ResampleResult is not None
    assert get_physical_resampler is not None


def test_physical_resampler_basic():
    from app.processing.resampler import PhysicalSpaceResampler

    volume = np.random.rand(200, 200, 128).astype(np.float32)
    original_spacing = (0.005, 0.005, 0.0035)
    target_shape = (128, 128, 64)

    resampler = PhysicalSpaceResampler(
        target_shape=target_shape,
        target_spacing=(0.005, 0.005, 0.0035),
        enable_pre_crop=True
    )

    result = resampler.resample(volume, original_spacing, is_mask=False)
    assert result.volume.shape == target_shape
    assert result.transform is not None
    assert result.transform.original_shape == volume.shape
    assert result.transform.final_shape == target_shape


def test_physical_resampler_different_spacing():
    from app.processing.resampler import PhysicalSpaceResampler

    volume = np.random.rand(150, 150, 256).astype(np.float32)
    original_spacing = (0.006, 0.006, 0.002)
    target_shape = (128, 128, 64)
    target_spacing = (0.005, 0.005, 0.0035)

    resampler = PhysicalSpaceResampler(
        target_shape=target_shape,
        target_spacing=target_spacing,
        enable_pre_crop=False
    )

    result = resampler.resample(volume, original_spacing, is_mask=False)
    assert result.volume.shape == target_shape
    assert len(result.transform.resample_scale_factors) == 3


def test_physical_resampler_roundtrip_inverse():
    from app.processing.resampler import PhysicalSpaceResampler

    volume = np.random.rand(180, 180, 128).astype(np.float32)
    original_spacing = (0.0048, 0.0048, 0.0036)
    target_shape = (128, 128, 64)
    target_spacing = (0.005, 0.005, 0.0035)

    resampler = PhysicalSpaceResampler(
        target_shape=target_shape,
        target_spacing=target_spacing,
        enable_pre_crop=False
    )

    result = resampler.resample(volume, original_spacing, is_mask=False)
    assert result.volume.shape == target_shape

    restored = resampler.inverse_transform(
        result.volume,
        result.transform,
        is_mask=False
    )
    assert restored.shape == volume.shape


def test_physical_resampler_mask_roundtrip():
    from app.processing.resampler import PhysicalSpaceResampler

    mask = np.zeros((200, 200, 128), dtype=np.uint8)
    mask[50:150, 50:150, 40:80] = 1
    original_spacing = (0.005, 0.005, 0.0035)
    target_shape = (128, 128, 64)
    target_spacing = (0.005, 0.005, 0.0035)

    resampler = PhysicalSpaceResampler(
        target_shape=target_shape,
        target_spacing=target_spacing,
        enable_pre_crop=False
    )

    result = resampler.resample(mask, original_spacing, is_mask=True)
    assert result.volume.shape == target_shape
    assert np.issubdtype(result.volume.dtype, np.unsignedinteger)

    restored = resampler.inverse_transform(
        result.volume,
        result.transform,
        is_mask=True
    )
    assert restored.shape == mask.shape
    assert np.issubdtype(restored.dtype, np.unsignedinteger)


def test_oct_preprocessor_with_physical_resampling():
    from app.processing.preprocessing import OCTPreprocessor
    from app.processing.image_loader import VolumeInfo

    volume = np.random.rand(200, 200, 128).astype(np.float32) * 100 + 50
    original_spacing = (0.005, 0.005, 0.0035)
    target_shape = (128, 128, 64)

    volume_info = VolumeInfo(
        shape=volume.shape,
        voxel_spacing=original_spacing,
        file_format="nifti"
    )

    preprocessor = OCTPreprocessor(
        target_shape=target_shape,
        use_physical_resampling=True,
        enable_pre_crop=True,
        apply_bias_correction=False
    )

    result = preprocessor.preprocess(volume, volume_info)
    assert result.volume.shape == target_shape
    assert result.original_voxel_spacing == original_spacing
    assert result.resampled_voxel_spacing == preprocessor.target_spacing
    assert result.resample_transform is not None
    assert hasattr(result, 'normalization_params')


def test_restore_mask_to_original_space():
    from app.processing.preprocessing import OCTPreprocessor
    from app.processing.image_loader import VolumeInfo

    volume = np.random.rand(180, 180, 128).astype(np.float32) * 100 + 50
    original_shape = volume.shape
    original_spacing = (0.006, 0.006, 0.0025)
    target_shape = (128, 128, 64)

    volume_info = VolumeInfo(
        shape=original_shape,
        voxel_spacing=original_spacing,
        file_format="nifti"
    )

    preprocessor = OCTPreprocessor(
        target_shape=target_shape,
        target_spacing=(0.005, 0.005, 0.0035),
        use_physical_resampling=True,
        enable_pre_crop=False,
        apply_bias_correction=False
    )

    preprocessed = preprocessor.preprocess(volume, volume_info)

    fake_mask_network = np.random.rand(*target_shape).astype(np.float32)
    fake_mask_network[40:80, 40:80, 20:40] = 0.9

    restored_mask = preprocessor.restore_mask_to_original_space(
        fake_mask_network,
        preprocessed,
        is_probability_map=False
    )
    assert restored_mask.shape == original_shape

    restored_prob = preprocessor.restore_mask_to_original_space(
        fake_mask_network,
        preprocessed,
        is_probability_map=True
    )
    assert restored_prob.shape == original_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
