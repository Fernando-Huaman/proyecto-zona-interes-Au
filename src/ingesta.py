from src.config import logger
import pandas as pd

def cargar_datos():
    ruta = "data/raw/DataSet_final.csv"
    logger.info(f"Cargando dataset desde: {ruta}")
    
    df = pd.read_csv(ruta)
    logger.info(f"Dataset cargado → {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Guardar en interim
    df.to_csv("data/interim/00_data_raw_cargado.csv", index=False)
    logger.info("Archivo guardado en data/interim/")
    return df

if __name__ == "__main__":
    cargar_datos()
