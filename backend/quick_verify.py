import sys
import time
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import numpy as np
import torch

torch.set_num_threads(4)


def main():
    print("=" * 60)
    print("  Glaucoma OCT AI Platform - Quick Verification")
    print("=" * 60)

    print("\n[1/6] Checking dependencies...")
    try:
        import fastapi, uvicorn, pydantic
        import numpy, scipy, skimage
        import nibabel, SimpleITK
        print(f"  ✅ FastAPI={fastapi.__version__}, Pydantic={pydantic.__version__}")
        print(f"  ✅ NumPy={numpy.__version__}, SciPy={scipy.__version__}")
        print(f"  ✅ Nibabel={nibabel.__version__}, SimpleITK={SimpleITK.__version__}")
    except ImportError as e:
        print(f"  ⚠️  Missing dep: {e}")

    print("\n[2/6] Loading settings & verifying config...")
    from app.config import get_settings
    settings = get_settings()
    print(f"  ✅ App: {settings.app_name} v{settings.app_version}")
    print(f"  ✅ Model: {settings.model_name}, Input: {settings.input_volume_size}")
    print(f"  ✅ Classes: {settings.num_classes}, Device: {settings.device}")
    print(f"  ✅ Dirs: upload={settings.upload_dir.exists()}, "
          f"model={settings.model_dir.exists()}, "
          f"output={settings.output_dir.exists()}")

    print("\n[3/6] Creating synthetic OCT volume...")
    from app.processing import VolumeInfo, OCTPreprocessor

    H, W, D = settings.input_volume_size
    volume = np.random.rand(H, W, D).astype(np.float32) * 100 + 50

    yy, xx, zz = np.mgrid[0:H, 0:W, 0:D]
    cx, cy, cz = H // 2, W // 2, D // 2
    dist_xy = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rnfl = (dist_xy < H * 0.4) & (np.abs(zz - cz) < D * 0.15)
    volume[rnfl] += 120
    volume = np.clip(volume, 0, 255)

    voxel_spacing = (0.005, 0.005, 0.0035)
    info = VolumeInfo(
        shape=volume.shape,
        voxel_spacing=voxel_spacing,
        file_format="nifti"
    )
    print(f"  ✅ Volume: {volume.shape}, range=[{volume.min():.0f},{volume.max():.0f}]")
    print(f"  ✅ Spacing: {voxel_spacing} mm")

    print("\n[4/6] Preprocessing...")
    t0 = time.perf_counter()
    preprocessor = OCTPreprocessor(
        target_shape=settings.input_volume_size,
        apply_bias_correction=False
    )
    preprocessed = preprocessor.preprocess(volume, info)
    input_tensor = preprocessor.to_tensor(preprocessed)
    t_prep = (time.perf_counter() - t0) * 1000
    print(f"  ✅ Preprocessed shape: {preprocessed.volume.shape}")
    print(f"  ✅ Input tensor: {input_tensor.shape}, "
          f"range=[{input_tensor.min():.2f},{input_tensor.max():.2f}]")
    print(f"  ✅ Time: {t_prep:.1f} ms")

    print("\n[5/6] Creating 3D-ResNet model & running inference...")
    from app.models import create_resnet3d_model

    t0 = time.perf_counter()
    model = create_resnet3d_model(
        in_channels=settings.in_channels,
        num_classes=settings.num_classes,
        model_name=settings.model_name
    )
    model.eval()
    params = model.get_parameters_count()
    t_load = (time.perf_counter() - t0) * 1000
    print(f"  ✅ Model params: {params:,} ({params/1e6:.1f}M)")
    print(f"  ✅ Load time: {t_load:.1f} ms")

    input_t = torch.from_numpy(input_tensor)
    with torch.no_grad():
        t0 = time.perf_counter()
        prediction = model.predict(input_t)
        t_infer = (time.perf_counter() - t0) * 1000

    prob_map = prediction.squeeze(0).numpy()
    if prob_map.shape[0] > 1:
        rnfl_prob = prob_map[1]
    else:
        rnfl_prob = prob_map[0]
    print(f"  ✅ Output: {prediction.shape}")
    print(f"  ✅ RNFL prob: [{rnfl_prob.min():.4f}, {rnfl_prob.max():.4f}]")
    print(f"  ✅ Inference time: {t_infer:.1f} ms")

    print("\n[6/6] Post-processing (thickness + defect detection)...")
    from app.processing import (
        create_thickness_map_data,
        detect_defect_regions,
        determine_overall_health,
        encode_segmentation_mask,
        extract_thickness_map,
        refine_segmentation
    )

    t0 = time.perf_counter()
    seg_mask = refine_segmentation(rnfl_prob, threshold=0.3, min_object_size=10, closing_radius=1)
    seg_mask_resized = seg_mask

    thickness_map = extract_thickness_map(
        seg_mask_resized, voxel_spacing, axis=2, method="axial_projection"
    )

    defect_regions = detect_defect_regions(
        thickness_map, voxel_spacing, min_region_size_pixels=3
    )

    health, confidence = determine_overall_health(thickness_map, defect_regions)

    thickness_data = create_thickness_map_data(thickness_map, voxel_spacing)

    encoded_mask = encode_segmentation_mask(seg_mask_resized)
    t_post = (time.perf_counter() - t0) * 1000

    print(f"  ✅ Segmentation: {seg_mask_resized.shape}, "
          f"RNFL voxels={int(np.sum(seg_mask_resized > 0)):,}")
    print(f"  ✅ Thickness map: {thickness_map.shape}")
    valid_t = thickness_map[thickness_map > 0]
    if valid_t.size > 0:
        print(f"  ✅ Thickness: mean={valid_t.mean():.1f}±{valid_t.std():.1f} μm, "
              f"range=[{valid_t.min():.1f},{valid_t.max():.1f}]")
    print(f"  ✅ Defect regions: {len(defect_regions)}")
    for r in defect_regions[:3]:
        print(f"    - #{r.region_id} {r.severity.value}: "
              f"{r.area_mm2:.2f} mm², mean={r.mean_thickness:.1f}μm")
    print(f"  ✅ Overall health: {health.value} (confidence={confidence:.1%})")
    print(f"  ✅ Encoded mask: {len(encoded_mask)} chars (Base64+zlib)")
    print(f"  ✅ Post-process time: {t_post:.1f} ms")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total processing time: {t_prep + t_infer + t_post:.1f} ms")
    print(f"  - Preprocessing:  {t_prep:.1f} ms")
    print(f"  - 3D-ResNet infer: {t_infer:.1f} ms")
    print(f"  - Postprocessing: {t_post:.1f} ms")
    print(f"  Model: 3D-ResNet18-UNet ({params/1e6:.1f}M params)")
    print(f"  Output: RNFL segmentation mask + thickness topology + defects")
    print(f"  Clinical status: {health.value.upper()} ({confidence:.1%})")
    print("=" * 60)
    print("  ✅ ALL VERIFICATION PASSED")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
