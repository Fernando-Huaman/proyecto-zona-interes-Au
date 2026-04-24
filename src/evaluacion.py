"""
Evaluación separada: Hold-out + Validación Cruzada
Guarda resultados en results/evaluacion_resultados.txt
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from src.config import logger
import os
from datetime import datetime

# Crear carpeta de resultados
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def guardar_resultados_txt(contenido: str, nombre_archivo: str = "evaluacion_resultados.txt"):
    """Guarda los resultados en un archivo TXT"""
    ruta = os.path.join(RESULTS_DIR, nombre_archivo)
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"FECHA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
        f.write(contenido)
        f.write("\n\n")
    logger.info(f"Resultados guardados en: {ruta}")


def evaluar_holdout(modelo, X_test_scaled, y_test, nombre_modelo):
    """Evaluación en conjunto de prueba (Hold-out)"""
    y_pred = modelo.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Evaluando Hold-out: {nombre_modelo}")
    print(f"\n{'='*70}")
    print(f"RESULTADOS HOLD-OUT - {nombre_modelo}")
    print(f"{'='*70}")
    print(report)
    
    # Guardar en TXT
    contenido = f"RESULTADOS HOLD-OUT - {nombre_modelo}\n{report}"
    guardar_resultados_txt(contenido, "evaluacion_resultados.txt")
    
    return report


def evaluar_cross_validation(modelo, X_train_scaled, y_train, nombre_modelo, cv=5):
    """Validación Cruzada Stratificada"""
    logger.info(f"Validación Cruzada para: {nombre_modelo}")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    cv_results = cross_validate(
        modelo, X_train_scaled, y_train,
        cv=skf, scoring=scoring, n_jobs=-1
    )
    
    print(f"\n{'='*70}")
    print(f"VALIDACIÓN CRUZADA (5-Fold) - {nombre_modelo}")
    print(f"{'='*70}")
    
    contenido_cv = f"VALIDACIÓN CRUZADA (5-Fold) - {nombre_modelo}\n"
    
    for metric in scoring:
        mean = cv_results[f'test_{metric}'].mean()
        std = cv_results[f'test_{metric}'].std()
        linea = f"  {metric:12} → {mean:.4f} ± {std:.4f}\n"
        print(linea.strip())
        contenido_cv += linea
    
    guardar_resultados_txt(contenido_cv, "evaluacion_resultados.txt")
    
    return cv_results