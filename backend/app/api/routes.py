import time
from pathlib import Path
from typing import Dict, Optional

import torch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..config import Settings, get_settings
from ..inference import RNFLSegmentationEngine, get_inference_engine, warmup_engine, _inference_engine
from ..models import (
    InferenceRequest,
    InferenceResponse,
    OCTUploadResponse,
    SystemStatus
)
from ..processing import get_volume_info, ImageLoadingError
from ..utils import (
    clean_temp_files,
    get_logger,
    save_uploaded_file,
    validate_file_extension
)

logger = get_logger(__name__)
router = APIRouter()

_start_time = time.perf_counter()


@router.on_event("startup")
async def startup_event():
    logger.info("Starting Glaucoma OCT AI Platform...")
    try:
        engine = get_inference_engine()
        warmup_engine()
        logger.info("Model loaded and warmed up successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)


@router.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Glaucoma OCT AI Platform...")
    global _inference_engine
    _inference_engine = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@router.get("/", summary="根路径", response_model=Dict[str, str])
async def root():
    return {
        "name": "Glaucoma OCT AI Platform",
        "version": "1.0.0",
        "status": "running",
        "description": "眼科医学 AI 平台 - 青光眼辅助诊断系统"
    }


@router.get("/health", summary="健康检查")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@router.get("/status", summary="系统状态", response_model=SystemStatus)
async def get_system_status(
    settings: Settings = Depends(get_settings),
    engine: RNFLSegmentationEngine = Depends(get_inference_engine)
):
    uptime = time.perf_counter() - _start_time
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    memory_usage = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_usage = memory_allocated / (1024 * 1024)

    stats = engine.get_stats()

    return SystemStatus(
        app_name=settings.app_name,
        app_version=settings.app_version,
        status="running",
        uptime_seconds=uptime,
        model=engine.get_model_info(),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        memory_usage_mb=memory_usage,
        active_requests=int(stats.get("active_requests", 0)),
        total_requests_processed=int(stats.get("total_requests", 0))
    )


@router.post(
    "/upload",
    summary="上传 OCT 图像文件",
    response_model=OCTUploadResponse,
    status_code=201
)
async def upload_oct_file(
    file: UploadFile = File(..., description="OCT 三维体素文件 (NIfTI/MHA 格式)"),
    settings: Settings = Depends(get_settings)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not validate_file_extension(file.filename, settings.allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(settings.allowed_extensions)}"
        )

    try:
        saved_path, file_size = await save_uploaded_file(
            file,
            settings.upload_dir,
            settings.max_upload_size
        )
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save file")

    try:
        volume_info = get_volume_info(saved_path)
    except ImageLoadingError as e:
        clean_temp_files([saved_path])
        raise HTTPException(status_code=400, detail=f"Invalid medical image: {e}")
    except Exception as e:
        clean_temp_files([saved_path])
        logger.error(f"Failed to read uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read medical image")

    file_id = saved_path.stem
    file_format = volume_info.file_format.upper()

    return OCTUploadResponse(
        file_id=file_id,
        file_name=file.filename,
        file_size=file_size,
        file_format=file_format,
        volume_shape=volume_info.shape,
        voxel_spacing=volume_info.voxel_spacing
    )


@router.post(
    "/infer",
    summary="执行 RNFL 分割推理",
    response_model=InferenceResponse,
    status_code=200
)
async def run_inference(
    request: InferenceRequest,
    settings: Settings = Depends(get_settings),
    engine: RNFLSegmentationEngine = Depends(get_inference_engine)
):
    file_path = settings.upload_dir / f"{request.file_id}.nii.gz"
    if not file_path.exists():
        file_path = settings.upload_dir / f"{request.file_id}.nii"
    if not file_path.exists():
        file_path = settings.upload_dir / f"{request.file_id}.mha"
    if not file_path.exists():
        file_path = settings.upload_dir / f"{request.file_id}.mhd"

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found for ID: {request.file_id}"
        )

    try:
        result = engine.run(
            file_path=file_path,
            patient_id=request.patient_id,
            study_id=request.study_id,
            return_segmentation=request.return_segmentation_mask,
            return_thickness=request.return_thickness_map,
            return_defects=request.return_defect_regions
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return result


@router.post(
    "/upload-and-infer",
    summary="上传并推理 (一站式)",
    response_model=InferenceResponse,
    status_code=200
)
async def upload_and_infer(
    patient_id: Optional[str] = None,
    study_id: Optional[str] = None,
    return_segmentation_mask: bool = True,
    return_thickness_map: bool = True,
    return_defect_regions: bool = True,
    file: UploadFile = File(..., description="OCT 三维体素文件"),
    settings: Settings = Depends(get_settings),
    engine: RNFLSegmentationEngine = Depends(get_inference_engine)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not validate_file_extension(file.filename, settings.allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(settings.allowed_extensions)}"
        )

    saved_path = None
    try:
        saved_path, file_size = await save_uploaded_file(
            file,
            settings.upload_dir,
            settings.max_upload_size
        )
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save file")

    try:
        result = engine.run(
            file_path=saved_path,
            patient_id=patient_id,
            study_id=study_id,
            return_segmentation=return_segmentation_mask,
            return_thickness=return_thickness_map,
            return_defects=return_defect_regions
        )
    except ImageLoadingError as e:
        if saved_path:
            clean_temp_files([saved_path])
        raise HTTPException(status_code=400, detail=f"Invalid medical image: {e}")
    except Exception as e:
        if saved_path:
            clean_temp_files([saved_path])
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return result


@router.delete(
    "/files/{file_id}",
    summary="删除上传的文件",
    status_code=204
)
async def delete_file(
    file_id: str,
    settings: Settings = Depends(get_settings)
):
    deleted = False
    for ext in settings.allowed_extensions:
        file_path = settings.upload_dir / f"{file_id}{ext}"
        if file_path.exists():
            try:
                file_path.unlink()
                deleted = True
                logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {e}")

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"File not found for ID: {file_id}"
        )

    return None


@router.get(
    "/models/info",
    summary="获取模型信息"
)
async def get_model_info(
    engine: RNFLSegmentationEngine = Depends(get_inference_engine)
):
    return engine.get_model_info()


@router.post(
    "/models/reload",
    summary="重新加载模型"
)
async def reload_model(
    engine: RNFLSegmentationEngine = Depends(get_inference_engine)
):
    try:
        engine.load_model()
        engine.warmup()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to reload model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")
