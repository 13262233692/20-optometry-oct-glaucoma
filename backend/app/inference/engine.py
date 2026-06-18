import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..config import get_settings
from ..models import (
    HealthStatus,
    InferenceResponse,
    RNFLSegmentationResult,
    create_resnet3d_model
)
from ..processing import (
    OCTPreprocessor,
    PreprocessingResult,
    compute_segmentation_statistics,
    create_thickness_map_data,
    decode_segmentation_mask,
    detect_defect_regions,
    determine_overall_health,
    encode_segmentation_mask,
    extract_thickness_map,
    load_medical_image,
    refine_segmentation
)
from ..utils import (
    get_device,
    get_logger,
    get_precision_context,
    optimize_model_for_inference
)

logger = get_logger(__name__)


@dataclass
class InferenceTiming:
    preprocessing_ms: float = 0.0
    inference_ms: float = 0.0
    postprocessing_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class InferenceResult:
    segmentation_mask: np.ndarray
    probability_map: np.ndarray
    thickness_map: Optional[np.ndarray]
    defect_regions: Optional[list]
    statistics: Dict[str, float]
    overall_health: HealthStatus
    confidence_score: float
    timing: InferenceTiming = field(default_factory=InferenceTiming)


class RNFLSegmentationEngine:
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        preload: bool = True
    ):
        settings = get_settings()
        self.settings = settings

        self.model_name = model_name or settings.model_name
        self.model_path = model_path or settings.model_path
        self.device_str = device or settings.device
        self.precision = precision or settings.precision

        self.device = get_device(self.device_str)
        self._raw_model = None
        self._optimized_model = None
        self._parameters_count = 0
        self.preprocessor = OCTPreprocessor()
        self.is_loaded = False
        self.is_warmed_up = False

        self.stats = {
            "total_requests": 0,
            "active_requests": 0,
            "total_inference_time_ms": 0.0
        }

        if preload:
            self.load_model()
            self.warmup()

    @property
    def model(self):
        return self._optimized_model if self._optimized_model is not None else self._raw_model

    def load_model(self) -> None:
        logger.info(f"Loading 3D-ResNet model: {self.model_name} on {self.device}")

        self._raw_model = create_resnet3d_model(
            in_channels=self.settings.in_channels,
            num_classes=self.settings.num_classes,
            model_name=self.model_name
        )
        self._parameters_count = self._raw_model.get_parameters_count()

        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Loading weights from: {self.model_path}")
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self._raw_model.load_state_dict(state_dict, strict=False)
                logger.info("Model weights loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}. Using initialized weights.")
        else:
            logger.warning("No pretrained weights found. Using random initialization.")

        input_size = (
            1, self.settings.in_channels,
            *self.settings.input_volume_size
        )

        try:
            self._optimized_model = optimize_model_for_inference(
                self._raw_model,
                device=self.device,
                input_size=input_size,
                precision=self.precision
            )
            has_predict = hasattr(self._optimized_model, "predict")
            if not has_predict:
                logger.info("Optimized model lacks predict method, using raw model for predict")
                self._optimized_model = None
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}. Using raw model.")
            self._optimized_model = None

        self.is_loaded = True
        logger.info(f"Model loaded successfully. Parameters: {self._parameters_count:,}")

    def _model_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self._optimized_model is not None:
            try:
                return self._raw_model.predict(input_tensor)
            except Exception:
                output = self._optimized_model(input_tensor)
                if self.settings.num_classes > 1:
                    return torch.softmax(output, dim=1)
                else:
                    return torch.sigmoid(output)
        else:
            return self._raw_model.predict(input_tensor)

    def warmup(self, num_runs: int = 3) -> None:
        if not self.is_loaded:
            self.load_model()

        logger.info("Warming up model for inference...")

        input_shape = (
            1, self.settings.in_channels,
            *self.settings.input_volume_size
        )

        dummy_input = torch.randn(input_shape, device=self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            dummy_input = dummy_input.half()

        with torch.no_grad():
            for i in range(num_runs):
                with get_precision_context(self.precision, self.device):
                    _ = self._model_predict(dummy_input)
                torch.cuda.synchronize() if self.device.type == "cuda" else None

        self.is_warmed_up = True
        logger.info("Model warmup completed")

    def _preprocess(
        self,
        volume: np.ndarray,
        volume_info
    ) -> Tuple[PreprocessingResult, np.ndarray]:
        preprocessed = self.preprocessor.preprocess(volume, volume_info)
        input_tensor = self.preprocessor.to_tensor(preprocessed)
        return preprocessed, input_tensor

    def _infer(self, input_tensor: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(input_tensor).to(self.device)

        if self.precision == "fp16" and self.device.type == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            with get_precision_context(self.precision, self.device):
                output = self._model_predict(tensor)

        probability_map = output.squeeze(0).cpu().numpy()
        if probability_map.shape[0] > 1:
            probability_map = probability_map[1]
        else:
            probability_map = probability_map[0]

        return probability_map

    def _postprocess(
        self,
        probability_map: np.ndarray,
        preprocessed: PreprocessingResult,
        return_segmentation: bool = True,
        return_thickness: bool = True,
        return_defects: bool = True
    ) -> InferenceResult:
        original_shape = preprocessed.original_shape
        original_spacing = preprocessed.original_voxel_spacing

        seg_mask_network_space = refine_segmentation(
            probability_map,
            threshold=0.5,
            min_object_size=100,
            closing_radius=2
        )

        logger.info(
            f"Post-processing: restoring masks to original physical space "
            f"({seg_mask_network_space.shape}@{preprocessed.resampled_voxel_spacing} → "
            f"{original_shape}@{original_spacing})"
        )

        try:
            segmentation_mask = self.preprocessor.restore_mask_to_original_space(
                seg_mask_network_space,
                preprocessed,
                is_probability_map=False
            )
            probability_map_restored = self.preprocessor.restore_mask_to_original_space(
                probability_map,
                preprocessed,
                is_probability_map=True
            )
        except Exception as e:
            logger.error(
                f"Inverse transform failed ({e}), falling back to nearest-neighbor resize. "
                f"Geometric accuracy may be compromised."
            )
            from ..processing import resize_volume
            segmentation_mask = resize_volume(
                seg_mask_network_space, original_shape, order=0
            )
            probability_map_restored = resize_volume(
                probability_map, original_shape, order=1
            )

        if segmentation_mask.shape != original_shape:
            logger.warning(
                f"Shape mismatch after inverse transform: "
                f"{segmentation_mask.shape} vs expected {original_shape}. Correcting."
            )
            from ..processing import resize_volume
            segmentation_mask = resize_volume(
                segmentation_mask, original_shape, order=0
            )
            probability_map_restored = resize_volume(
                probability_map_restored, original_shape, order=1
            )

        thickness_map = None
        defect_regions = None
        statistics = {}
        overall_health = HealthStatus.UNKNOWN
        confidence_score = 0.0

        if return_thickness:
            thickness_map = extract_thickness_map(
                segmentation_mask,
                original_spacing,
                axis=2,
                method="axial_projection"
            )
            logger.info(
                f"Thickness map computed in original physical space: "
                f"shape={thickness_map.shape}, spacing_Z={original_spacing[2]*1000:.2f} μm/slice"
            )

        if return_defects and thickness_map is not None:
            defect_regions = detect_defect_regions(
                thickness_map,
                original_spacing
            )

        statistics = compute_segmentation_statistics(
            segmentation_mask,
            thickness_map,
            original_spacing,
            defect_regions
        )
        statistics["inverse_transform_success"] = 1.0

        if thickness_map is not None:
            overall_health, confidence_score = determine_overall_health(
                thickness_map,
                defect_regions
            )

        return InferenceResult(
            segmentation_mask=segmentation_mask,
            probability_map=probability_map_restored,
            thickness_map=thickness_map,
            defect_regions=defect_regions,
            statistics=statistics,
            overall_health=overall_health,
            confidence_score=confidence_score
        )

    def run(
        self,
        file_path: Path,
        patient_id: Optional[str] = None,
        study_id: Optional[str] = None,
        return_segmentation: bool = True,
        return_thickness: bool = True,
        return_defects: bool = True
    ) -> InferenceResponse:
        if not self.is_loaded:
            self.load_model()
        if not self.is_warmed_up:
            self.warmup()

        request_id = str(uuid.uuid4())
        self.stats["active_requests"] += 1
        self.stats["total_requests"] += 1

        timing = InferenceTiming()
        total_start = time.perf_counter()

        try:
            logger.info(f"Processing request {request_id}: {file_path.name}")

            volume, volume_info = load_medical_image(file_path)
            logger.info(f"Loaded volume: shape={volume.shape}, spacing={volume_info.voxel_spacing}")

            preprocess_start = time.perf_counter()
            preprocessed, input_tensor = self._preprocess(volume, volume_info)
            timing.preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000

            inference_start = time.perf_counter()
            probability_map = self._infer(input_tensor)
            timing.inference_ms = (time.perf_counter() - inference_start) * 1000

            postprocess_start = time.perf_counter()
            result = self._postprocess(
                probability_map,
                preprocessed,
                return_segmentation,
                return_thickness,
                return_defects
            )
            timing.postprocessing_ms = (time.perf_counter() - postprocess_start) * 1000

            result.timing = timing
            timing.total_ms = (time.perf_counter() - total_start) * 1000
            self.stats["total_inference_time_ms"] += timing.total_ms

            response = self._build_response(
                request_id=request_id,
                file_id=file_path.stem,
                patient_id=patient_id,
                study_id=study_id,
                result=result,
                timing=timing,
                return_segmentation=return_segmentation,
                return_thickness=return_thickness,
                return_defects=return_defects,
                original_shape=preprocessed.original_shape,
                voxel_spacing=preprocessed.original_voxel_spacing
            )

            logger.info(
                f"Request {request_id} completed: "
                f"health={result.overall_health.value}, "
                f"confidence={result.confidence_score:.3f}, "
                f"total_time={timing.total_ms:.1f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}", exc_info=True)
            raise
        finally:
            self.stats["active_requests"] -= 1

    def _build_response(
        self,
        request_id: str,
        file_id: str,
        patient_id: Optional[str],
        study_id: Optional[str],
        result: InferenceResult,
        timing: InferenceTiming,
        return_segmentation: bool,
        return_thickness: bool,
        return_defects: bool,
        original_shape: Tuple[int, int, int],
        voxel_spacing: Tuple[float, float, float]
    ) -> InferenceResponse:
        segmentation = None
        if return_segmentation:
            encoded_mask = encode_segmentation_mask(result.segmentation_mask)
            segmentation = RNFLSegmentationResult(
                segmentation_mask_encoding=encoded_mask,
                segmentation_mask_shape=result.segmentation_mask.shape,
                rnfl_volume_voxels=int(result.statistics.get("rnfl_volume_voxels", 0)),
                rnfl_volume_mm3=float(result.statistics.get("rnfl_volume_mm3", 0.0))
            )

        thickness_map_data = None
        if return_thickness and result.thickness_map is not None:
            thickness_map_data = create_thickness_map_data(
                result.thickness_map,
                voxel_spacing
            )

        defect_regions = None
        if return_defects and result.defect_regions is not None:
            defect_regions = result.defect_regions

        model_info = {
            "name": self.model_name,
            "version": "1.0.0",
            "input_size": "x".join(str(x) for x in self.settings.input_volume_size),
            "num_classes": str(self.settings.num_classes),
            "device": str(self.device),
            "precision": self.precision
        }

        return InferenceResponse(
            request_id=request_id,
            file_id=file_id,
            patient_id=patient_id,
            study_id=study_id,
            status="completed",
            overall_health=result.overall_health,
            confidence_score=result.confidence_score,
            inference_time_ms=timing.inference_ms,
            preprocessing_time_ms=timing.preprocessing_ms,
            postprocessing_time_ms=timing.postprocessing_ms,
            segmentation=segmentation,
            thickness_map=thickness_map_data,
            defect_regions=defect_regions,
            statistics=result.statistics,
            model_info=model_info
        )

    def get_model_info(self):
        from ..models import ModelInfo
        return ModelInfo(
            name=self.model_name,
            version="1.0.0",
            input_size=self.settings.input_volume_size,
            num_classes=self.settings.num_classes,
            device=str(self.device),
            precision=self.precision,
            is_loaded=self.is_loaded,
            parameters_count=self._parameters_count
        )

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "average_inference_time_ms": (
                self.stats["total_inference_time_ms"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0.0
            )
        }


_inference_engine: Optional[RNFLSegmentationEngine] = None


def get_inference_engine() -> RNFLSegmentationEngine:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = RNFLSegmentationEngine(preload=True)
    return _inference_engine


def warmup_engine() -> None:
    engine = get_inference_engine()
    if not engine.is_warmed_up:
        engine.warmup()
