import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import logger
import subprocess

def run_command(cmd, description):
    logger.info(f"Ejecutando: {description}")
    print(f"\n{description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("Completado")
    else:
        print("Error:", result.stderr)
    return result.returncode == 0

def ejecutar_pipeline_completo():
    print("="*90)
    print("EJECUCIÓN COMPLETA DEL PIPELINE - ZONA DE INTERÉS Au")
    print("="*90)
    
    # 1. Ingesta
    run_command("python -m src.ingesta", "1. Ingesta de datos")
    
    # 2. Preprocesamiento
    run_command("python -m src.preprocesamiento", "2. Preprocesamiento y creación de target")
    
    # 3. EDA Notebook (antes del Baseline)
    print("\nEjecutando Notebook de EDA...")
    cmd_eda = (
        "cd /workspaces/proyecto-zona-interes-Au && "
        "jupyter nbconvert --execute --to notebook --inplace "
        "--ExecutePreprocessor.timeout=300 notebooks/EDA.ipynb"
    )
    run_command(cmd_eda, "3. Notebook EDA")
    
    # 4. Baseline + Evaluación
    run_command("python -m src.modelo_baseline", "4. Modelos Baseline + Validación Cruzada")
    
    print("\n" + "="*90)
    print("¡PIPELINE COMPLETO FINALIZADO CON ÉXITO!")
    print("Archivos generados:")
    print("   • results/evaluacion_resultados.txt")
    print("   • notebooks/EDA.ipynb (ejecutado con gráficos)")
    print("   • logs/pipeline.log")
    print("="*90)

if __name__ == "__main__":
    ejecutar_pipeline_completo()