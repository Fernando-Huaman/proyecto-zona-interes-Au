"""
Configuración global del proyecto
"""
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Rutas
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_INTERIM = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Crear todas las carpetas necesarias
for folder in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, LOGS_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.info("Estructura de carpetas creada correctamente")
