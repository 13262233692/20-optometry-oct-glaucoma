from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class GradCAMResult:
    heatmap_3d: np.ndarray
    heatmap_axial_max: np.ndarray
    heatmap_coronal_max: np.ndarray
    heatmap_sagittal_max: np.ndarray
    target_layer_name: str
    class_index: int
    class_score: float
    metadata: Dict[str, float] = field(default_factory=dict)


class GradCAM3D:
    """
    3D 版本的 Grad-CAM（Gradient-weighted Class Activation Mapping）

    算法原理：
    1. 在前向传播时截获目标卷积层的特征图激活 A^k
    2. 对目标类别分数 y^c 做反向传播，获得该层的梯度 ∂y^c/∂A^k
    3. 对梯度做全局平均池化 (GAP) 得到神经元重要性权重 α^c_k
    4. 加权求和 L^c = ReLU(Σ α^c_k * A^k) 得到类激活热力图
    5. 三线性插值上采样到原图尺寸

    针对医学分割网络，目标层选择策略：
    - encoder.layer4[-1]: 深层语义，定位大范围病灶（如视神经盘凹陷）
    - decoders[-1].conv2: 解码器末端，高分辨率细粒度（如血管旁变薄）
    """

    DEFAULT_TARGET_LAYERS = [
        "model.encoder.layer4",
        "model.decoders.3",
        "model.final_conv.3"
    ]

    def __init__(
        self,
        model: nn.Module,
        target_layer_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.target_layer_names = target_layer_names or self.DEFAULT_TARGET_LAYERS

        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()

    def _resolve_layer(self, layer_path: str) -> nn.Module:
        modules = layer_path.split(".")
        current = self.model
        for name in modules:
            if hasattr(current, name):
                current = getattr(current, name)
            else:
                raise AttributeError(f"Layer '{layer_path}' not found at '{name}'")
        if isinstance(current, nn.Sequential):
            current = current[-1]
            logger.info(f"Resolved Sequential '{layer_path}' to its last block")
        return current

    def _register_hooks(self) -> None:
        self._remove_hooks()
        for layer_name in self.target_layer_names:
            try:
                layer = self._resolve_layer(layer_name)
                act_handle = layer.register_forward_hook(
                    self._get_activation_hook(layer_name)
                )
                grad_handle = layer.register_full_backward_hook(
                    self._get_gradient_hook(layer_name)
                )
                self._handles.extend([act_handle, grad_handle])
                logger.info(f"Registered Grad-CAM hooks on: {layer_name}")
            except Exception as e:
                logger.warning(f"Failed to register hooks on {layer_name}: {e}")

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations.clear()
        self._gradients.clear()

    def _get_activation_hook(self, layer_name: str):
        def hook(module, input, output):
            self._activations[layer_name] = output.detach()
        return hook

    def _get_gradient_hook(self, layer_name: str):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self._gradients[layer_name] = grad_output[0].detach()
        return hook

    @staticmethod
    def _compute_cam(
        activations: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        if activations.ndim != 5:
            raise ValueError(f"Expected 5D activations (N,C,D,H,W), got {activations.shape}")

        grads = gradients
        if grads.shape != activations.shape:
            grads = F.interpolate(
                grads,
                size=activations.shape[2:],
                mode="trilinear",
                align_corners=False
            )

        weights = torch.mean(grads, dim=(2, 3, 4), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        return cam

    @staticmethod
    def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
        cam_min = torch.amin(cam, dim=(2, 3, 4), keepdim=True)
        cam_max = torch.amax(cam, dim=(2, 3, 4), keepdim=True)
        denom = cam_max - cam_min
        denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
        return (cam - cam_min) / denom

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_index: Optional[int] = None,
        target_shape: Optional[Tuple[int, int, int]] = None,
        fusion: str = "mean"
    ) -> GradCAMResult:
        if input_tensor.ndim != 5:
            raise ValueError(
                f"Expected 5D input (N,C,D,H,W), got {input_tensor.shape}"
            )
        if input_tensor.shape[0] != 1:
            raise ValueError(f"Only batch_size=1 supported, got {input_tensor.shape[0]}")

        was_training = self.model.training
        self.model.eval()

        input_var = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        try:
            logits = self.model(input_var)
        except Exception as e:
            self.model.train(was_training)
            raise RuntimeError(f"Forward pass failed: {e}")

        if isinstance(logits, list):
            logits = logits[0]

        num_classes = logits.shape[1]
        if class_index is None:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1) if num_classes > 1 else torch.sigmoid(logits)
                class_index = int(torch.argmax(probs[:, 1:]) + 1) if num_classes > 1 else 0
        class_index = max(0, min(class_index, num_classes - 1))

        if num_classes > 1:
            target = logits[:, class_index].sum()
        else:
            target = logits[:, 0].sum()

        try:
            target.backward(retain_graph=False)
        except Exception as e:
            self.model.train(was_training)
            raise RuntimeError(f"Backward pass failed: {e}")

        cams = []
        valid_layers = []
        for layer_name in self.target_layer_names:
            if layer_name not in self._activations or layer_name not in self._gradients:
                logger.warning(f"No activations/gradients for {layer_name}, skipping")
                continue
            try:
                cam = self._compute_cam(
                    self._activations[layer_name],
                    self._gradients[layer_name]
                )
                cams.append(cam)
                valid_layers.append(layer_name)
            except Exception as e:
                logger.warning(f"Failed to compute CAM for {layer_name}: {e}")

        if not cams:
            self.model.train(was_training)
            raise RuntimeError("No valid CAMs computed from any layer")

        if len(cams) == 1:
            fused_cam = cams[0]
            target_layer_name = valid_layers[0]
        else:
            max_shape = max(c.shape[2:] for c in cams)
            resized = [
                F.interpolate(c, size=max_shape, mode="trilinear", align_corners=False)
                for c in cams
            ]
            stacked = torch.cat(resized, dim=0)
            if fusion == "mean":
                fused_cam = torch.mean(stacked, dim=0, keepdim=True)
            elif fusion == "max":
                fused_cam = torch.max(stacked, dim=0, keepdim=True)[0]
            else:
                fused_cam = torch.mean(stacked, dim=0, keepdim=True)
            target_layer_name = "+".join(valid_layers)

        norm_cam = self._normalize_cam(fused_cam)

        final_shape = target_shape or tuple(input_tensor.shape[2:])
        if tuple(norm_cam.shape[2:]) != final_shape:
            norm_cam = F.interpolate(
                norm_cam,
                size=final_shape,
                mode="trilinear",
                align_corners=False
            )

        heatmap_3d = norm_cam.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        heatmap_axial_max = np.max(heatmap_3d, axis=2)
        heatmap_coronal_max = np.max(heatmap_3d, axis=1)
        heatmap_sagittal_max = np.max(heatmap_3d, axis=0)

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1) if num_classes > 1 else torch.sigmoid(logits)
            class_score = float(probs[:, class_index].mean().cpu().numpy())

        self.model.train(was_training)

        result = GradCAMResult(
            heatmap_3d=heatmap_3d,
            heatmap_axial_max=heatmap_axial_max,
            heatmap_coronal_max=heatmap_coronal_max,
            heatmap_sagittal_max=heatmap_sagittal_max,
            target_layer_name=target_layer_name,
            class_index=class_index,
            class_score=class_score,
            metadata={
                "num_layers_fused": len(valid_layers),
                "fusion_method": fusion,
                "mean_activation": float(np.mean(heatmap_3d)),
                "max_activation": float(np.max(heatmap_3d))
            }
        )

        logger.info(
            f"Grad-CAM generated: class={class_index} (score={class_score:.3f}), "
            f"layers={target_layer_name}, heatmap_shape={heatmap_3d.shape}"
        )

        return result

    def __del__(self):
        self._remove_hooks()


def compute_gradcam3d(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_index: Optional[int] = None,
    target_shape: Optional[Tuple[int, int, int]] = None,
    target_layers: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> GradCAMResult:
    cam = GradCAM3D(model, target_layer_names=target_layers, device=device)
    try:
        return cam.generate(input_tensor, class_index, target_shape)
    finally:
        cam._remove_hooks()


@dataclass
class GradCAMHeatmapData:
    axial_projection: List[List[float]]
    coronal_projection: List[List[float]]
    sagittal_projection: List[List[float]]
    target_layer: str
    class_index: int
    class_score: float
    mean_activation: float
    max_activation: float


def get_gradcam_heatmap_data(
    result: GradCAMResult,
    voxel_spacing: Optional[Tuple[float, float, float]] = None
) -> GradCAMHeatmapData:
    axial = result.heatmap_axial_max.astype(np.float64)
    coronal = result.heatmap_coronal_max.astype(np.float64)
    sagittal = result.heatmap_sagittal_max.astype(np.float64)

    return GradCAMHeatmapData(
        axial_projection=axial.tolist(),
        coronal_projection=coronal.tolist(),
        sagittal_projection=sagittal.tolist(),
        target_layer=result.target_layer_name,
        class_index=result.class_index,
        class_score=round(float(result.class_score), 4),
        mean_activation=round(float(result.metadata.get("mean_activation", 0.0)), 4),
        max_activation=round(float(result.metadata.get("max_activation", 0.0)), 4)
    )
