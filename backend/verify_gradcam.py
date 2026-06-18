import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def synthesize_oct_volume(shape=(200, 200, 128)):
    from app.processing.image_loader import VolumeInfo

    volume = np.random.randn(*shape).astype(np.float32) * 15 + 80
    yy, xx, zz = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    cy, cx, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    dist_r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    disc_mask = dist_r < min(shape[0], shape[1]) * 0.15
    rnfl_cup_mask = (dist_r > min(shape[0], shape[1]) * 0.18) & \
                    (dist_r < min(shape[0], shape[1]) * 0.38) & \
                    (np.abs(zz - cz) < shape[2] * 0.12)
    vessel_mask = (np.abs(xx - (cx - 40)) < 3) & \
                  (np.abs(yy - cy) < 80) & \
                  (np.abs(zz - cz) < shape[2] * 0.15)

    volume[disc_mask] += 15
    volume[rnfl_cup_mask] += 60
    volume[vessel_mask] -= 30
    volume = np.clip(volume, 0, 255)

    volume_info = VolumeInfo(
        shape=shape,
        voxel_spacing=(0.005, 0.005, 0.0035),
        file_format="nifti"
    )
    return volume, volume_info


def save_temp_nifti(volume, volume_info, tmp_dir):
    try:
        import nibabel as nib
    except ImportError:
        return None

    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = volume_info.voxel_spacing[i]

    nifti_path = Path(tmp_dir) / "test_oct_xai.nii.gz"
    img = nib.Nifti1Image(volume, affine)
    nib.save(img, str(nifti_path))
    return nifti_path


def main():
    import tempfile

    from app.config import get_settings
    from app.inference.engine import RNFLSegmentationEngine, InferenceTiming

    print("=" * 70)
    print("  3D Grad-CAM XAI - Full Pipeline Verification")
    print("=" * 70)

    settings = get_settings()
    print(f"\n[1/5] Config:")
    print(f"  enable_xai = {settings.enable_xai}")
    print(f"  xai_class_index = {settings.xai_class_index}")
    print(f"  xai_target_layers = {settings.xai_target_layers}")

    print("\n[2/5] Synthesizing pathological OCT volume:")
    volume, volume_info = synthesize_oct_volume(shape=(200, 200, 128))
    print(f"  Input: {volume.shape} @ spacing={volume_info.voxel_spacing} mm")
    print(f"  Range: [{volume.min():.1f}, {volume.max():.1f}]")

    print("\n[3/5] Loading inference engine:")
    engine = RNFLSegmentationEngine(preload=False)
    engine.load_model()
    print(f"  Parameters: {engine._parameters_count:,}")
    print(f"  Device: {engine.device}")

    timing = InferenceTiming()
    t_total = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp_dir:
        nifti_path = save_temp_nifti(volume, volume_info, tmp_dir)

        if nifti_path is not None:
            print(f"\n[4/5] Running full inference with Grad-CAM XAI:")
            try:
                response = engine.run(
                    file_path=nifti_path,
                    patient_id="XAI-PAT-001",
                    study_id="XAI-STUDY-001",
                    return_segmentation=True,
                    return_thickness=True,
                    return_defects=True
                )
                timing.total_ms = (time.perf_counter() - t_total) * 1000

                print(f"\n[5/5] Results:")
                print(f"  Overall health: {response.overall_health.value}")
                print(f"  Confidence: {response.confidence_score:.3f}")
                print(f"  Preprocessing: {response.preprocessing_time_ms:.1f} ms")
                print(f"  Inference: {response.inference_time_ms:.1f} ms")
                print(f"  XAI (Grad-CAM): {response.xai_time_ms:.1f} ms")
                print(f"  Postprocessing: {response.postprocessing_time_ms:.1f} ms")
                print(f"  Total: {timing.total_ms:.1f} ms")

                print(f"\n  Segmentation: shape={response.segmentation.segmentation_mask_shape}")
                print(f"  Thickness map: {len(response.thickness_map.thickness_values)}x"
                      f"{len(response.thickness_map.thickness_values[0])}")

                if response.gradcam_heatmap is not None:
                    gc = response.gradcam_heatmap
                    print(f"\n  === Grad-CAM 3D XAI Result ===")
                    print(f"  Target layer(s): {gc.target_layer}")
                    print(f"  Attributed class: {gc.class_index} (1=RNFL, 0=Background)")
                    print(f"  Class score: {gc.class_score:.4f}")
                    print(f"  Mean activation: {gc.mean_activation:.4f}")
                    print(f"  Max activation: {gc.max_activation:.4f}")

                    axial = np.array(gc.axial_projection, dtype=np.float32)
                    coronal = np.array(gc.coronal_projection, dtype=np.float32)
                    sagittal = np.array(gc.sagittal_projection, dtype=np.float32)

                    print(f"\n  3D Heatmap projections:")
                    print(f"    Axial (XY) max-proj: {axial.shape}, "
                          f"range=[{axial.min():.3f}, {axial.max():.3f}]")
                    print(f"    Coronal (XZ) max-proj: {coronal.shape}, "
                          f"range=[{coronal.min():.3f}, {coronal.max():.3f}]")
                    print(f"    Sagittal (YZ) max-proj: {sagittal.shape}, "
                          f"range=[{sagittal.min():.3f}, {sagittal.max():.3f}]")

                    high_thresh = 0.7
                    axial_high = np.count_nonzero(axial > high_thresh)
                    print(f"\n  Clinical interpretability:")
                    print(f"    High-attention pixels (>{high_thresh}) in axial view: "
                          f"{axial_high:,} / {axial.size:,}")
                    if axial_high > 0:
                        yy, xx = np.where(axial > high_thresh)
                        cy, cx = np.mean(yy), np.mean(xx)
                        print(f"    Center of high attention: "
                              f"(x={cx:.1f}, y={cy:.1f}) in axial plane")
                        print(f"    --> Highlights location of RNFL abnormality driving AI decision")

                    assert axial.shape == volume.shape[:2], \
                        f"Axial projection should match original XY: {axial.shape} vs {volume.shape[:2]}"
                    assert 0.0 <= axial.min() <= axial.max() <= 1.0, "Heatmap not normalized to [0,1]"
                    assert gc.max_activation > 0.1, "Should produce meaningful activation"

                    print("\n" + "=" * 70)
                    print("  [SUCCESS] Grad-CAM 3D XAI integrated end-to-end!")
                    print("  Doctors can now see which specific 3D regions drove AI's decision")
                    print("=" * 70)
                    return 0
                else:
                    print("\n  [WARNING] gradcam_heatmap is None. XAI may be disabled or failed.")
                    return 1

            except Exception as e:
                import traceback
                print(f"\n  [FAILED] Engine run error: {e}")
                traceback.print_exc()
                return 1
        else:
            print("[SKIP] nibabel not available, testing engine._generate_gradcam directly")
            from app.processing import OCTPreprocessor

            preprocessor = OCTPreprocessor(
                target_shape=settings.input_volume_size,
                use_physical_resampling=True,
                apply_bias_correction=False
            )
            preprocessed, input_tensor = engine._preprocess(volume, volume_info)

            t_xai = time.perf_counter()
            gradcam_result = engine._generate_gradcam(input_tensor, target_shape=preprocessed.target_shape)
            xai_ms = (time.perf_counter() - t_xai) * 1000
            print(f"  Grad-CAM time: {xai_ms:.1f} ms")

            if gradcam_result is not None:
                print(f"  Heatmap 3D shape: {gradcam_result.heatmap_3d.shape}")
                print(f"  Layers used: {gradcam_result.target_layer_name}")
                print(f"  Class score: {gradcam_result.class_score:.4f}")
                return 0
            else:
                print("[FAIL] Grad-CAM returned None")
                return 1


if __name__ == "__main__":
    sys.exit(main())
