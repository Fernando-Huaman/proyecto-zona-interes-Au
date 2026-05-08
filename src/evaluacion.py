"""
Evaluación de los modelos
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from src.config import logger
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