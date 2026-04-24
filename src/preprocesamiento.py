"""
Preprocesamiento: valores censurados y target
"""
import pandas as pd
import numpy as np
from src.config import logger

def parse_censored(value):
    """Convierte valores censurados geoquímicos"""
    if isinstance(value, str):
        if value.startswith('<'):
            try: return float(value[1:]) / 2
            except: return np.nan
        if value.startswith('>'):
            try: return float(value[1:]) * 1.1
            except: return np.nan
    try:
        return float(value)
    except:
        return np.nan

def tratar_valores_censurados(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando tratamiento de valores censurados...")
    cols = [col for col in df.columns if col not in ['Codigo','East','North','Level','Datum','Zone']]
    
    for col in cols:
        df[col] = df[col].apply(parse_censored)
    
    df = df.drop(columns=['Codigo','Datum','Zone'], errors='ignore')
    logger.info(f"Tratamiento completado → Dimensiones: {df.shape}")
    return df

def crear_target(df: pd.DataFrame) -> pd.DataFrame:
    df['target_Au'] = (df['Au_ppm'] > 0.10).astype(int)
    logger.info("Variable objetivo 'target_Au' creada")
    logger.info(f"Distribución:\n{df['target_Au'].value_counts(normalize=True).round(3)*100}")
    return df

def guardar_datos_procesados(df: pd.DataFrame):
    ruta = "data/processed/data_procesada.csv"
    df.to_csv(ruta, index=False)
    logger.info(f"Datos procesados guardados en: {ruta}")

if __name__ == "__main__":
    from src.ingesta import cargar_datos
    df = cargar_datos()
    df = tratar_valores_censurados(df)
    df = crear_target(df)
    guardar_datos_procesados(df)