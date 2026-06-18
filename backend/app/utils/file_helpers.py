import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import UploadFile

from ..config import get_settings


def validate_file_extension(
    filename: str,
    allowed_extensions: Optional[Tuple[str, ...]] = None
) -> bool:
    settings = get_settings()
    allowed = allowed_extensions or settings.allowed_extensions

    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in allowed)


async def save_uploaded_file(
    file: UploadFile,
    upload_dir: Optional[Path] = None,
    max_size: Optional[int] = None
) -> Tuple[Path, int]:
    settings = get_settings()
    upload_dir = upload_dir or settings.upload_dir
    max_size = max_size or settings.max_upload_size

    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    saved_filename = f"{file_id}{file_ext}"
    saved_path = upload_dir / saved_filename

    file_size = 0
    with open(saved_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            file_size += len(chunk)
            if file_size > max_size:
                saved_path.unlink(missing_ok=True)
                raise ValueError(f"File too large. Maximum size: {max_size / 1024 / 1024:.1f}MB")
            f.write(chunk)

    return saved_path, file_size


def clean_temp_files(file_paths: List[Path]) -> None:
    for file_path in file_paths:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass
