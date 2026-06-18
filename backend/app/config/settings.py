import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    app_name: str = Field(default="Glaucoma OCT AI Platform", description="应用名称")
    app_version: str = Field(default="1.0.0", description="应用版本")
    api_prefix: str = Field(default="/api/v1", description="API 前缀")
    debug: bool = Field(default=False, description="调试模式")

    host: str = Field(default="0.0.0.0", description="服务监听地址")
    port: int = Field(default=8000, description="服务监听端口")

    base_dir: Path = Field(default_factory=_resolve_base_dir)
    upload_dir: Optional[Path] = Field(default=None, description="上传文件目录")
    model_dir: Optional[Path] = Field(default=None, description="模型文件目录")
    output_dir: Optional[Path] = Field(default=None, description="输出文件目录")

    max_upload_size: int = Field(default=500 * 1024 * 1024, description="最大上传文件大小 (字节)")
    allowed_extensions: Tuple[str, ...] = Field(
        default=(".nii", ".nii.gz", ".mha", ".mhd"),
        description="允许的文件扩展名"
    )

    model_name: str = Field(default="resnet3d_18", description="模型名称")
    model_path: Optional[Path] = Field(default=None, description="预训练模型权重路径")
    device: str = Field(default="auto", description="推理设备: auto/cpu/cuda")
    precision: str = Field(default="fp32", description="推理精度: fp32/fp16")

    input_volume_size: Tuple[int, int, int] = Field(
        default=(128, 128, 64),
        description="输入体素尺寸 (H, W, D)"
    )
    num_classes: int = Field(default=2, description="分割类别数 (背景+RNFL)")
    in_channels: int = Field(default=1, description="输入通道数")

    enable_xai: bool = Field(default=True, description="是否启用 XAI 可解释性分析 (Grad-CAM 3D)")
    xai_class_index: int = Field(default=1, description="XAI 归因类别 (1=RNFL, 0=背景)")
    xai_target_layers: Optional[str] = Field(
        default=None,
        description="Grad-CAM 目标层逗号分隔列表，None=使用默认多层融合"
    )

    rnfl_normal_thickness_min: float = Field(default=80.0, description="RNFL 正常厚度下限 (μm)")
    rnfl_normal_thickness_max: float = Field(default=120.0, description="RNFL 正常厚度上限 (μm)")
    rnfl_thickness_warning_threshold: float = Field(
        default=70.0,
        description="RNFL 厚度警告阈值 (μm)"
    )
    rnfl_thickness_danger_threshold: float = Field(
        default=50.0,
        description="RNFL 厚度危险阈值 (μm)"
    )

    @model_validator(mode="after")
    def _validate_and_create_dirs(self):
        base = self.base_dir
        if self.upload_dir is None:
            self.upload_dir = base / "data" / "uploads"
        if self.model_dir is None:
            self.model_dir = base / "data" / "models"
        if self.output_dir is None:
            self.output_dir = base / "data" / "outputs"

        for directory in [self.upload_dir, self.model_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()
