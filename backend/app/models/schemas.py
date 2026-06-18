from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict


class HealthStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    DANGER = "danger"
    UNKNOWN = "unknown"


class OCTUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    file_id: str = Field(..., description="上传文件唯一标识")
    file_name: str = Field(..., description="原始文件名")
    file_size: int = Field(..., description="文件大小 (字节)")
    file_format: str = Field(..., description="文件格式: NIfTI 或 MHA")
    volume_shape: Tuple[int, int, int] = Field(..., description="体素矩阵形状 (H, W, D)")
    voxel_spacing: Tuple[float, float, float] = Field(..., description="体素间距 (mm)")
    upload_time: datetime = Field(default_factory=datetime.utcnow, description="上传时间")


class InferenceRequest(BaseModel):
    file_id: str = Field(..., description="要推理的文件ID")
    patient_id: Optional[str] = Field(None, description="患者ID")
    study_id: Optional[str] = Field(None, description="检查ID")
    return_segmentation_mask: bool = Field(default=True, description="是否返回分割掩码")
    return_thickness_map: bool = Field(default=True, description="是否返回厚度图")
    return_defect_regions: bool = Field(default=True, description="是否返回缺损区域")


class ThicknessMapData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    thickness_values: List[List[float]] = Field(..., description="厚度值二维数组 (μm)")
    coordinates_x: List[float] = Field(..., description="X 轴坐标 (mm)")
    coordinates_y: List[float] = Field(..., description="Y 轴坐标 (mm)")
    min_thickness: float = Field(..., description="最小厚度 (μm)")
    max_thickness: float = Field(..., description="最大厚度 (μm)")
    mean_thickness: float = Field(..., description="平均厚度 (μm)")
    median_thickness: float = Field(..., description="中位厚度 (μm)")
    std_thickness: float = Field(..., description="厚度标准差 (μm)")


class DefectRegion(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    region_id: int = Field(..., description="区域ID")
    severity: HealthStatus = Field(..., description="严重程度")
    center: Tuple[int, int] = Field(..., description="区域中心坐标 (x, y)")
    bounding_box: Tuple[int, int, int, int] = Field(..., description="边界框 (x1, y1, x2, y2)")
    area_pixels: int = Field(..., description="区域面积 (像素)")
    area_mm2: float = Field(..., description="区域面积 (mm²)")
    mean_thickness: float = Field(..., description="区域平均厚度 (μm)")
    min_thickness: float = Field(..., description="区域最小厚度 (μm)")


class RNFLSegmentationResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    segmentation_mask_encoding: str = Field(..., description="Base64 编码的分割掩码")
    segmentation_mask_shape: Tuple[int, int, int] = Field(..., description="掩码形状")
    rnfl_volume_voxels: int = Field(..., description="RNFL 体素数量")
    rnfl_volume_mm3: float = Field(..., description="RNFL 体积 (mm³)")


class GradCAMHeatmapResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    axial_projection: List[List[float]] = Field(
        ..., description="轴位（Z轴）最大投影热力图 [0,1] 归一化"
    )
    coronal_projection: List[List[float]] = Field(
        ..., description="冠状位（Y轴）最大投影热力图 [0,1] 归一化"
    )
    sagittal_projection: List[List[float]] = Field(
        ..., description="矢状位（X轴）最大投影热力图 [0,1] 归一化"
    )
    target_layer: str = Field(..., description="Grad-CAM 目标卷积层名称")
    class_index: int = Field(..., description="归因类别索引 (1=RNFL)")
    class_score: float = Field(..., description="归因类别置信度 [0,1]")
    mean_activation: float = Field(..., description="3D 热力图平均激活值")
    max_activation: float = Field(..., description="3D 热力图最大激活值")


class InferenceResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=()
    )

    request_id: str = Field(..., description="请求唯一标识")
    file_id: str = Field(..., description="文件ID")
    patient_id: Optional[str] = Field(None, description="患者ID")
    study_id: Optional[str] = Field(None, description="检查ID")
    status: str = Field(..., description="推理状态")
    overall_health: HealthStatus = Field(..., description="整体健康状态")
    confidence_score: float = Field(..., description="置信度分数 (0-1)")
    inference_time_ms: float = Field(..., description="推理耗时 (毫秒)")
    preprocessing_time_ms: float = Field(..., description="预处理耗时 (毫秒)")
    postprocessing_time_ms: float = Field(..., description="后处理耗时 (毫秒)")
    xai_time_ms: float = Field(default=0.0, description="XAI Grad-CAM 特征归因耗时 (毫秒)")

    segmentation: Optional[RNFLSegmentationResult] = Field(None, description="分割结果")
    thickness_map: Optional[ThicknessMapData] = Field(None, description="厚度分布拓扑图")
    defect_regions: Optional[List[DefectRegion]] = Field(None, description="高危缺损区域列表")
    gradcam_heatmap: Optional[GradCAMHeatmapResponse] = Field(
        None, description="Grad-CAM 3D 特征归因热力图（XAI 可解释性）"
    )

    statistics: Dict[str, float] = Field(default_factory=dict, description="统计数据")
    model_info: Dict[str, str] = Field(default_factory=dict, description="模型信息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="完成时间")


class ModelInfo(BaseModel):
    name: str = Field(..., description="模型名称")
    version: str = Field(..., description="模型版本")
    input_size: Tuple[int, int, int] = Field(..., description="输入尺寸")
    num_classes: int = Field(..., description="输出类别数")
    device: str = Field(..., description="运行设备")
    precision: str = Field(..., description="运算精度")
    is_loaded: bool = Field(..., description="是否已加载")
    parameters_count: int = Field(..., description="参数量")


class SystemStatus(BaseModel):
    app_name: str = Field(..., description="应用名称")
    app_version: str = Field(..., description="应用版本")
    status: str = Field(..., description="系统状态")
    uptime_seconds: float = Field(..., description="运行时长 (秒)")
    model: ModelInfo = Field(..., description="模型信息")
    gpu_available: bool = Field(..., description="GPU 是否可用")
    gpu_name: Optional[str] = Field(None, description="GPU 名称")
    memory_usage_mb: Optional[float] = Field(None, description="显存使用 (MB)")
    active_requests: int = Field(default=0, description="当前活跃请求数")
    total_requests_processed: int = Field(default=0, description="总处理请求数")
