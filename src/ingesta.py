"""
Ingesta: Carga el dataset original
"""
from src.config import logger
import pandas as pd

def cargar_datos():
    """Carga el archivo CSV original"""
    ruta = "data/raw/DataSet_final.csv"
    logger.info(f"Iniciando ingesta desde: {ruta}")
    
    df = pd.read_csv(ruta)
    logger.info(f"Dataset cargado → {df.shape[0]} filas, {df.shape[1]} columnas")
    
    df.to_csv("data/interim/00_data_raw_cargado.csv", index=False)
    logger.info("Copia raw guardada en data/interim/")
    return df

if __name__ == "__main__":
    cargar_datos()