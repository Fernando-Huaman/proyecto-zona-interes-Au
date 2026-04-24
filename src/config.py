"""
Configuración global del proyecto.
Incluye logging y rutas principales.
"""
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)
DATA_PROCESSED.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.info("Configuración cargada correctamente")