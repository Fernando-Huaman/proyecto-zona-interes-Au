"""
Evaluación de los modelos
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from src.config import logger
import os
from datetime import datetime

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULT_FILE = os.path.join(RESULTS_DIR, "evaluacion_resultados.txt")


def limpiar_archivo_resultados():
    """Limpia el archivo antes de una nueva evaluación"""
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(f"RESULTADOS DE EVALUACIÓN - ZONA DE INTERÉS Au\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*90 + "\n\n")


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
