"""
Evaluación de los modelos
Incluye: hold-out, validación cruzada, bootstrap IC 95%, evaluación por slice.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, mean_absolute_error, mean_squared_error,
    r2_score, root_mean_squared_error, f1_score, precision_score,
    recall_score, average_precision_score, roc_auc_score, brier_score_loss,
    accuracy_score
)
from src.config import logger, SEED
import os
from datetime import datetime
import numpy as np
import shutil

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULT_FILE = os.path.join(RESULTS_DIR, "evaluacion_resultados.txt")

def limpiar_archivo_resultados():
    """Limpia el archivo antes de una nueva evaluación"""
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(f"RESULTADOS DE EVALUACIÓN - ZONA DE INTERÉS Au\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*90 + "\n\n")

def limpiar_carpeta_results():
    """Elimina todos los archivos results"""
    if os.path.exists(RESULTS_DIR):
        try:
            shutil.rmtree(RESULTS_DIR)      # Borra todo
            os.makedirs(RESULTS_DIR)        # Vuelve a crear la carpeta vacía
            print("Carpeta results/ limpiada completamente")
        except Exception as e:
            print(f"Error al limpiar results/: {e}")
    else:
        os.makedirs(RESULTS_DIR)

def evaluar_holdout(modelo, X_test_scaled, y_test, nombre_modelo):
    """Evaluación en conjunto de prueba"""
    y_pred = modelo.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    
    header = f"""
{'='*80}
RESULTADOS HOLD-OUT - {nombre_modelo}
{'='*80}
"""
    print(header + report)
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(header + report + "\n\n")
    return report


def evaluar_cross_validation(modelo, X_train_scaled, y_train, nombre_modelo, cv=5):
    """Validación Cruzada 5-Fold"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    cv_results = cross_validate(modelo, X_train_scaled, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    
    header = f"""
{'='*80}
VALIDACIÓN CRUZADA (5-Fold) - {nombre_modelo}
{'='*80}
"""
    print(header)
    contenido = header
    
    for metric in scoring:
        mean = cv_results[f'test_{metric}'].mean()
        std = cv_results[f'test_{metric}'].std()
        linea = f"  {metric:12} → {mean:.4f} ± {std:.4f}\n"
        print(linea.strip())
        contenido += linea
    
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(contenido + "\n\n")
    
    return cv_results

def evaluar_regresion(modelo, X_test_scaled, y_test_real_ppm, nombre_modelo):
    """Evaluación para modelos de regresión"""
    y_pred_log = modelo.predict(X_test_scaled)
    y_pred_ppm = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_test_real_ppm, y_pred_ppm)
    rmse = root_mean_squared_error(y_test_real_ppm, y_pred_ppm)
    r2 = r2_score(y_test_real_ppm, y_pred_ppm)
    
    header = f"""
{'='*80}
RESULTADOS REGRESIÓN - {nombre_modelo}
{'='*80}
"""
    report = f"MAE  (ppm): {mae:.4f}\nRMSE (ppm): {rmse:.4f}\nR²         : {r2:.4f}\n"
    print(header + report)
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(header + report + "\n\n")
    return mae, rmse, r2


# ============================================================
# FUNCIONES AVANZADAS — Sprint 4
# ============================================================

def evaluar_global_bootstrap(y_true, y_prob, y_pred, n_boot=1000, conf=0.95, seed=SEED):
    """
    Evaluación global con bootstrap IC al nivel de confianza especificado.

    Retorna dict con métricas puntuales + IC para F1, Precision, Recall,
    PR-AUC, ROC-AUC, Brier.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = (1 - conf) / 2

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    metrics = {
        'n': n,
        'n_pos': int(y_true.sum()),
        'n_neg': int((1 - y_true).sum()),
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1':        f1_score(y_true, y_pred, zero_division=0),
        'PR_AUC':    average_precision_score(y_true, y_prob),
        'ROC_AUC':   roc_auc_score(y_true, y_prob),
        'Brier':     brier_score_loss(y_true, y_prob),
    }

    boot_keys = ['F1', 'Precision', 'Recall', 'PR_AUC', 'ROC_AUC', 'Brier']
    boot_metrics = {k: [] for k in boot_keys}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]

        if len(np.unique(yt)) < 2:
            continue

        boot_metrics['F1'].append(f1_score(yt, yp, zero_division=0))
        boot_metrics['Precision'].append(precision_score(yt, yp, zero_division=0))
        boot_metrics['Recall'].append(recall_score(yt, yp, zero_division=0))
        boot_metrics['PR_AUC'].append(average_precision_score(yt, ypr))
        boot_metrics['ROC_AUC'].append(roc_auc_score(yt, ypr))
        boot_metrics['Brier'].append(brier_score_loss(yt, ypr))

    for key in boot_keys:
        vals = np.array(boot_metrics[key])
        if len(vals) > 0:
            metrics[f'{key}_CI_low']  = float(np.percentile(vals, alpha * 100))
            metrics[f'{key}_CI_high'] = float(np.percentile(vals, (1 - alpha) * 100))
            metrics[f'{key}_std']     = float(np.std(vals))
        else:
            metrics[f'{key}_CI_low']  = np.nan
            metrics[f'{key}_CI_high'] = np.nan
            metrics[f'{key}_std']     = np.nan

    return metrics


def evaluar_slice(y_true, y_prob, y_pred, n_boot=1000, conf=0.95, seed=SEED):
    """
    Evalúa desempeño por subpoblación (slice) con bootstrap IC 95%.

    Requiere al menos 5 muestras y 2 clases. Retorna dict con métricas
    + IC para F1, PR-AUC, ROC-AUC.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return {k: np.nan for k in [
            'n', 'n_pos', 'Precision', 'Recall', 'F1',
            'F1_CI_low', 'F1_CI_high',
            'PR_AUC', 'PR_AUC_CI_low', 'PR_AUC_CI_high',
            'ROC_AUC', 'ROC_AUC_CI_low', 'ROC_AUC_CI_high', 'Brier'
        ]}

    metrics = {
        'n': len(y_true),
        'n_pos': int(y_true.sum()),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1':        f1_score(y_true, y_pred, zero_division=0),
        'PR_AUC':    average_precision_score(y_true, y_prob),
        'ROC_AUC':   roc_auc_score(y_true, y_prob),
        'Brier':     brier_score_loss(y_true, y_prob),
    }

    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = (1 - conf) / 2
    boot = {'F1': [], 'PR_AUC': [], 'ROC_AUC': []}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot['F1'].append(f1_score(yt, yp, zero_division=0))
        boot['PR_AUC'].append(average_precision_score(yt, ypr))
        boot['ROC_AUC'].append(roc_auc_score(yt, ypr))

    for key in boot:
        vals = np.array(boot[key])
        if len(vals) > 0:
            metrics[f'{key}_CI_low']  = float(np.percentile(vals, alpha * 100))
            metrics[f'{key}_CI_high'] = float(np.percentile(vals, (1 - alpha) * 100))
        else:
            metrics[f'{key}_CI_low']  = np.nan
            metrics[f'{key}_CI_high'] = np.nan

    return metrics


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    Busca el umbral óptimo que maximiza la métrica indicada.
    Usado en Mitigación A (umbral por slice).
    """
    best_score, best_thresh = -1, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        yp = (y_prob >= t).astype(int)
        if metric == 'f1':
            s = f1_score(y_true, yp, zero_division=0)
        elif metric == 'recall':
            s = recall_score(y_true, yp, zero_division=0)
        else:
            s = precision_score(y_true, yp, zero_division=0)
        if s > best_score:
            best_score, best_thresh = s, t
    return best_thresh, best_score