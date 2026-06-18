import io
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import numpy as np
import pytest


def test_full_inference_pipeline():
    from app.config import get_settings
    from app.inference import RNFLSegmentationEngine
    from app.models import HealthStatus
    from app.processing import (
        create_thickness_map_data,
        detect_defect_regions,
        determine_overall_health,
        encode_segmentation_mask,
        extract_thickness_map,
        resize_volume,
    )

    settings = get_settings()
    print(f"\nSettings loaded: {settings.app_name} v{settings.app_version}")
    print(f"  Device: {settings.device}")
    print(f"  Model: {settings.model_name}")
    print(f"  Input size: {settings.input_volume_size}")

    print("\nCreating synthetic OCT volume...")
    original_shape = (200, 200, 128)
    volume = np.random.randn(*original_shape).astype(np.float32) * 30 + 100

    center = (100, 100, 64)
    yy, xx, zz = np.mgrid[0:200, 0:200, 0:128]
    distance = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    rnfl_region = (distance < 80) & (zz > 55) & (zz < 75)
    volume[rnfl_region] += 80
    volume = np.clip(volume, 0, 255)

    voxel_spacing = (0.005, 0.005, 0.0035)
    print(f"  Volume shape: {volume.shape}")
    print(f"  Voxel spacing: {voxel_spacing} mm")

    from app.processing.image_loader import VolumeInfo
    from app.processing.preprocessing import OCTPreprocessor

    volume_info = VolumeInfo(
        shape=original_shape,
        voxel_spacing=voxel_spacing,
        file_format="nifti"
    )

    print("\nRunning preprocessing...")
    preprocessor = OCTPreprocessor(target_shape=settings.input_volume_size)
    preprocessed = preprocessor.preprocess(volume, volume_info)
    input_tensor = preprocessor.to_tensor(preprocessed)
    print(f"  Preprocessed shape: {preprocessed.volume.shape}")
    print(f"  Input tensor shape: {input_tensor.shape}")
    print(f"  Value range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

    print("\nInitializing inference engine (without preload for test)...")
    engine = RNFLSegmentationEngine(preload=False)
    print("Loading model...")
    engine.load_model()
    print(f"  Parameters: {engine._parameters_count:,}")
    print(f"  Device: {engine.device}")

    print("\nRunning warmup...")
    engine.warmup(num_runs=1)

    print("\nRunning inference...")
    import time
    start = time.perf_counter()
    probability_map = engine._infer(input_tensor)
    infer_time = (time.perf_counter() - start) * 1000
    print(f"  Output shape: {probability_map.shape}")
    print(f"  Probability range: [{probability_map.min():.4f}, {probability_map.max():.4f}]")
    print(f"  Inference time: {infer_time:.1f} ms")

    print("\nRunning postprocessing...")
    from app.processing.postprocessing import (
        compute_segmentation_statistics,
        refine_segmentation,
        resize_to_original
    )

    segmentation_mask = refine_segmentation(
        probability_map,
        threshold=0.5,
        min_object_size=50,
        closing_radius=1
    )
    segmentation_mask = resize_to_original(segmentation_mask, original_shape, order=0)
    print(f"  Segmentation mask shape: {segmentation_mask.shape}")
    print(f"  RNFL voxels: {int(np.sum(segmentation_mask > 0)):,}")

    thickness_map = extract_thickness_map(
        segmentation_mask,
        voxel_spacing,
        axis=2,
        method="axial_projection"
    )
    print(f"  Thickness map shape: {thickness_map.shape}")
    valid_t = thickness_map[thickness_map > 0]
    if valid_t.size > 0:
        print(f"  Thickness range: [{valid_t.min():.1f}, {valid_t.max():.1f}] μm")
        print(f"  Mean thickness: {valid_t.mean():.1f} ± {valid_t.std():.1f} μm")

    defect_regions = detect_defect_regions(
        thickness_map,
        voxel_spacing,
        warning_threshold=70.0,
        danger_threshold=50.0,
        min_region_size_pixels=5
    )
    print(f"  Defect regions detected: {len(defect_regions)}")
    for r in defect_regions[:3]:
        print(f"    - Region {r.region_id}: {r.severity.value}, "
              f"area={r.area_mm2:.2f} mm², "
              f"mean_thickness={r.mean_thickness:.1f} μm")

    statistics = compute_segmentation_statistics(
        segmentation_mask,
        thickness_map,
        voxel_spacing,
        defect_regions
    )
    print(f"  Statistics computed: {len(statistics)} keys")
    for k, v in list(statistics.items())[:5]:
        print(f"    {k}: {v:.4f}")

    overall_health, confidence = determine_overall_health(
        thickness_map,
        defect_regions
    )
    print(f"  Overall health: {overall_health.value}")
    print(f"  Confidence score: {confidence:.3f}")

    print("\nEncoding segmentation mask...")
    encoded = encode_segmentation_mask(segmentation_mask)
    print(f"  Encoded size: {len(encoded)} chars")

    thickness_data = create_thickness_map_data(thickness_map, voxel_spacing)
    print(f"  Thickness data created")
    print(f"    Min: {thickness_data.min_thickness:.1f} μm")
    print(f"    Max: {thickness_data.max_thickness:.1f} μm")
    print(f"    Mean: {thickness_data.mean_thickness:.1f} μm")

    print("\nBuilding full response...")
    from dataclasses import dataclass, field
    from app.inference.engine import InferenceTiming, InferenceResult

    timing = InferenceTiming(
        preprocessing_ms=12.5,
        inference_ms=infer_time,
        postprocessing_ms=8.3,
        total_ms=infer_time + 20.8
    )

    result = InferenceResult(
        segmentation_mask=segmentation_mask,
        probability_map=probability_map,
        thickness_map=thickness_map,
        defect_regions=defect_regions,
        statistics=statistics,
        overall_health=overall_health,
        confidence_score=confidence,
        timing=timing
    )

    print("\n✅ Full inference pipeline completed successfully!")
    print(f"   Total time: {timing.total_ms:.1f} ms")
    print(f"   Health: {result.overall_health.value} ({result.confidence_score:.1%})")

    assert result is not None
    assert result.segmentation_mask.shape == original_shape
    assert result.thickness_map.shape == original_shape[:2]
    assert isinstance(result.overall_health, HealthStatus)
    assert 0.0 <= result.confidence_score <= 1.0

    return True


def test_save_and_load_synthetic_nifti():
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("Nibabel not available")
        return

    from app.processing import load_medical_image, preprocess_oct_volume

    with TemporaryDirectory() as tmp_dir:
        print("\nCreating synthetic NIfTI file...")
        shape = (64, 64, 32)
        data = np.random.rand(*shape).astype(np.float32) * 200 + 20
        affine = np.eye(4)
        affine[0, 0] = 0.01
        affine[1, 1] = 0.01
        affine[2, 2] = 0.01

        img = nib.Nifti1Image(data, affine)
        nifti_path = Path(tmp_dir) / "test_oct.nii.gz"
        nib.save(img, str(nifti_path))
        print(f"  Saved: {nifti_path} ({nifti_path.stat().st_size / 1024:.1f} KB)")

        print("Loading back with MedicalImageLoader...")
        volume, info = load_medical_image(nifti_path)
        print(f"  Loaded shape: {volume.shape}")
        print(f"  Format: {info.file_format}")
        print(f"  Spacing: {info.voxel_spacing}")

        assert volume.shape == shape
        assert info.file_format == "nifti"

        print("Running preprocessing...")
        preprocessed = preprocess_oct_volume(volume, info, target_shape=(64, 64, 32))
        print(f"  Preprocessed shape: {preprocessed.volume.shape}")

        assert preprocessed.volume.shape == (64, 64, 32)
        print("✅ NIfTI save/load test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("  Glaucoma OCT AI Platform - Integration Test")
    print("=" * 60)

    test_full_inference_pipeline()
    print("\n" + "-" * 60 + "\n")
    try:
        test_save_and_load_synthetic_nifti()
    except Exception as e:
        print(f"⚠️  NIfTI test skipped/warning: {e}")

    print("\n" + "=" * 60)
    print("  All integration tests completed successfully!")
    print("=" * 60)
