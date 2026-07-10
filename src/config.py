"""
Configuración global del proyecto
"""
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Rutas de datos
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_INTERIM = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# Rutas de salida
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
CONFIGS_DIR = BASE_DIR / "configs"

# Rutas de modelos y outputs por sprint
MODELS_DIR = BASE_DIR / "notebooks" / "outputs_sprint3" / "models"
OUTPUTS_SPRINT2 = BASE_DIR / "notebooks" / "outputs_sprint2"
OUTPUTS_SPRINT3 = BASE_DIR / "notebooks" / "outputs_sprint3"
OUTPUTS_SPRINT4 = BASE_DIR / "notebooks" / "outputs_sprint4"

# Columnas excluidas de features (constante global)
EXCLUDE_COLS = ['Au_ppm', 'target_Au', 'East', 'North', 'Level', 'is_synthetic']

# Parámetros del modelo final (XGBoost Optuna + Mitigación A)
SEED = 42
N_FOLDS = 5
METRIC = "f1"

# Parámetros XGBoost Optuna (Sprint 3 — mejor modelo)
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': SEED,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
}

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
