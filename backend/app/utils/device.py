from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


@contextmanager
def get_precision_context(precision: str = "fp32", device: Optional[torch.device] = None):
    if precision == "fp16" and device and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            yield
    elif precision == "bf16" and device and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            yield
    else:
        with torch.autocast(device_type="cpu", enabled=False):
            yield


def optimize_model_for_inference(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int, int],
    precision: str = "fp32"
) -> nn.Module:
    model.eval()
    model = model.to(device)

    if precision == "fp16" and device.type == "cuda":
        model = model.half()
    elif precision == "bf16" and device.type == "cuda":
        model = model.to(torch.bfloat16)

    try:
        dummy_input = torch.randn(input_size, device=device)
        if precision == "fp16" and device.type == "cuda":
            dummy_input = dummy_input.half()
        elif precision == "bf16" and device.type == "cuda":
            dummy_input = dummy_input.to(torch.bfloat16)

        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input, strict=False)
            optimized_model = torch.jit.freeze(traced_model)
        return optimized_model
    except Exception:
        for param in model.parameters():
            param.requires_grad = False
        return model
