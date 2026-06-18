import os
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import uvicorn
from app.config import get_settings
from app.utils import setup_logger

settings = get_settings()
logger = setup_logger("run_server")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Glaucoma OCT AI Platform - Server Startup")
    logger.info("=" * 60)
    logger.info(f"  App Name: {settings.app_name}")
    logger.info(f"  Version: {settings.app_version}")
    logger.info(f"  Host: {settings.host}")
    logger.info(f"  Port: {settings.port}")
    logger.info(f"  Debug: {settings.debug}")
    logger.info(f"  Device: {settings.device}")
    logger.info(f"  Model: {settings.model_name}")
    logger.info(f"  Input Size: {settings.input_volume_size}")
    logger.info("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1,
        log_level="info" if settings.debug else "warning"
    )
