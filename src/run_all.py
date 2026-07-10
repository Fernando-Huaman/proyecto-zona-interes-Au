"""
Pipeline completo — Zona de Interés Au

Ejecuta los pasos del pipeline con scripts (.py):
  1. Ingesta de datos
  2. Preprocesamiento y balanceo
  3. Baseline de REGRESIÓN (guarda el mejor regresor → mejor_regresor.pkl)
  4. Modelo Final de CLASIFICACIÓN (XGBoost Optuna + Mitigación A):
     métricas con IC 95% + gráficos de clasificación
  5. Mapa final (XGBoost → probabilidad/color + Mejor Regresor → ley ppm/tamaño)

NOTA: Los notebooks (EDA, sprint2, sprint3, sprint4) se ejecutan de forma
INDEPENDIENTE y no forman parte de este pipeline. El paso 4 requiere los
archivos *_fe.csv generados por notebooks/sprint2.ipynb (Feature Engineering).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import logger, DATA_INTERIM
import subprocess


def run_command(cmd, description):
    logger.info(f"Ejecutando: {description}")
    print(f"\n{'─'*70}")
    print(f"  {description}...")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✓ Completado")
    else:
        print(f"  ✗ Error: {result.stderr[:200]}")
    return result.returncode == 0


def ejecutar_pipeline_completo():
    print("=" * 90)
    print("  EJECUCIÓN COMPLETA DEL PIPELINE — ZONA DE INTERÉS Au")
    print("  Modelo final: XGBoost Optuna + Mitigación A (F1 = 0.9808)")
    print("=" * 90)
    print("  NOTA: Los notebooks (EDA, sprint2, sprint3, sprint4) se ejecutan aparte.")
    print("        El modelo final requiere los *_fe.csv generados por sprint2.ipynb.")
    print("=" * 90)

    # 1. Ingesta
    run_command("python -m src.ingesta", "1. Ingesta de datos")

    # 2. Preprocesamiento
    run_command("python -m src.preprocesamiento", "2. Preprocesamiento y creación de target")

    # 3. Baseline de regresión (guarda el mejor regresor)
    run_command("python -m src.modelo_baseline",
                "3. Baseline de Regresión + Guardado del mejor regresor")

    # 4. Modelo Final (clasificación: métricas + gráficos + Mitigación A)
    if not (DATA_INTERIM / "data_prueba_fe.csv").exists():
        print("\n  ⚠ No se encontraron los datos con Feature Engineering (*_fe.csv).")
        print("    Ejecuta primero el notebook: notebooks/sprint2.ipynb")
        print("    Se omite el modelo final y el mapa.")
    else:
        run_command("python -m src.modelo_final",
                    "4. Modelo Final — XGBoost Optuna + Mitigación A (métricas + gráficos)")

        # 5. Mapa final (clasificación final + mejor regresor)
        print("\n  Generando mapa final (XGBoost + Mejor Regresor)...")
        try:
            from src.mapas import generar_mapa_modelo_final
            generar_mapa_modelo_final()
            print("  ✓ Mapa del modelo final generado")
        except Exception as e:
            print(f"  ✗ Error generando mapa: {e}")

    # Resumen
    print("\n" + "=" * 90)
    print("  ¡PIPELINE COMPLETO FINALIZADO!")
    print("=" * 90)
    print("  Archivos generados:")
    print("   • results/evaluacion_resultados.txt")
    print("   • results/metricas_modelo_final.csv")
    print("   • results/umbrales_mitigacion_a.csv")
    print("   • results/ModeloFinal_*.png (matriz confusión, ROC, PR, métricas)")
    print("   • results/Comparacion_regre*.png (regresión baseline)")
    print("   • results/mapa_XGBoost_Optuna_MitigacionA.png")
    print("   • notebooks/outputs_sprint3/models/mejor_regresor.pkl")
    print("   • notebooks/outputs_sprint3/models/modelo_final_xgboost.pkl")
    print("   • logs/pipeline.log")
    print("=" * 90)


if __name__ == "__main__":
    ejecutar_pipeline_completo()
