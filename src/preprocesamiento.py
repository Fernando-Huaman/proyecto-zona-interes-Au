import pandas as pd
import numpy as np
from src.config import logger

def parse_censored(value):
    if isinstance(value, str):
        if value.startswith('<'): return float(value[1:]) / 2
        if value.startswith('>'): return float(value[1:]) * 1.1
    try: return float(value)
    except: return np.nan

def tratar_valores_censurados(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Tratando valores censurados...")
    cols = [col for col in df.columns if col not in ['Codigo','East','North','Level','Datum','Zone']]
    for col in cols:
        df[col] = df[col].apply(parse_censored)
    df = df.drop(columns=['Codigo','Datum','Zone'], errors='ignore')
    logger.info(f"Tratamiento completado → {df.shape}")
    return df

def crear_target(df: pd.DataFrame) -> pd.DataFrame:
    df['target_Au'] = (df['Au_ppm'] > 0.10).astype(int)
    logger.info("Variable objetivo creada")
    return df

def guardar_datos_procesados(df: pd.DataFrame):
    ruta = "data/processed/data_procesada.csv"
    df.to_csv(ruta, index=False)
    logger.info(f"Datos procesados guardados en {ruta}")

if __name__ == "__main__":
    from src.ingesta import cargar_datos
    df = cargar_datos()
    df = tratar_valores_censurados(df)
    df = crear_target(df)
    guardar_datos_procesados(df)
