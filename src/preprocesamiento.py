import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import FunctionTransformer
from src.config import logger, DATA_PROCESSED, DATA_INTERIM
import os

def parse_censored(value):
    """Maneja valores censurados (<0.01, >10, etc.)"""
    if isinstance(value, str):
        if value.startswith('<'):
            return float(value[1:]) / 2
        if value.startswith('>'):
            return float(value[1:]) * 1.1
    try:
        return float(value)
    except:
        return np.nan


def tratar_valores_censurados(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Tratando valores censurados...")
    cols = [col for col in df.columns 
            if col not in ['Codigo', 'East', 'North', 'Level', 'Datum', 'Zone']]
    for col in cols:
        df[col] = df[col].apply(parse_censored)
    df = df.drop(columns=['Codigo', 'Datum', 'Zone'], errors='ignore')
    logger.info(f"Tratamiento completado → {df.shape}")
    return df


def crear_target(df: pd.DataFrame) -> pd.DataFrame:
    df['target_Au'] = (df['Au_ppm'] > 0.10).astype(int)
    logger.info(f"Variable objetivo creada → {df['target_Au'].value_counts().to_dict()}")
    return df


def log1p_transform(X):
    return np.log1p(X)


def inverse_log1p_transform(X):
    return np.expm1(X)


def balancear_y_dividir(df: pd.DataFrame, random_state: int = 42, ratio: str = 'auto'):
    logger.info("=== Iniciando balanceo con datos sintéticos (solo train) ===")

    try:
        from imblearn.over_sampling import BorderlineSMOTE
    except ImportError as e:
        logger.error(f"Error importando imbalanced-learn: {e}")
        guardar_datos_procesados(df)
        return None, None

    feature_cols = [col for col in df.columns 
                    if col not in ['Au_ppm', 'target_Au', 'East', 'North', 'Level']]
    coord_cols = ['East', 'North', 'Level']

    X = df[feature_cols].copy()
    y = df['target_Au']
    coords = df[coord_cols]

    # Imputación
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

    # Split estratificado
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords, test_size=0.20, stratify=y, random_state=random_state
    )

    # Transformación log antes de SMOTE
    log_transformer = FunctionTransformer(log1p_transform, inverse_func=inverse_log1p_transform)
    X_train_log = log_transformer.fit_transform(X_train)
    X_test_log = log_transformer.transform(X_test)

    # Balanceo con BorderlineSMOTE
    smote = BorderlineSMOTE(sampling_strategy=ratio, random_state=random_state, kind='borderline-1')
    X_train_bal_log, y_train_bal = smote.fit_resample(X_train_log, y_train)
    X_train_bal = log_transformer.inverse_transform(X_train_bal_log)

    # DataFrame balanceado
    df_train_bal = pd.DataFrame(X_train_bal, columns=feature_cols)
    df_train_bal['target_Au'] = y_train_bal.values
    df_train_bal['is_synthetic'] = [0] * len(X_train) + [1] * (len(df_train_bal) - len(X_train))

    # Restaurar Au_ppm y coordenadas originales en muestras reales
    original_idx = X_train.index
    df_train_bal.loc[:len(X_train)-1, 'Au_ppm'] = df.loc[original_idx, 'Au_ppm'].values
    df_train_bal.loc[:len(X_train)-1, coord_cols] = df.loc[original_idx, coord_cols].values

    df_train_bal['Au_ppm'] = df_train_bal['Au_ppm'].fillna(0.0)

    # Test set (original)
    df_test = pd.DataFrame(X_test.values, columns=feature_cols)
    df_test['target_Au'] = y_test.values
    df_test['Au_ppm'] = df.loc[X_test.index, 'Au_ppm'].values
    df_test[coord_cols] = coords_test.values
    df_test['is_synthetic'] = 0

    # Guardar versiones base
    os.makedirs(DATA_INTERIM, exist_ok=True)
    df_train_bal.to_csv(DATA_INTERIM / "data_entrenamiento_balanceado.csv", index=False)
    df_test.to_csv(DATA_INTERIM / "data_prueba.csv", index=False)

    return df_train_bal, df_test


def guardar_datos_procesados(df: pd.DataFrame):
    # Eliminar columna fantasma si existe
    if 'Unnamed: 4' in df.columns:
        df = df.drop(columns=['Unnamed: 4'])
        logger.info("Columna 'Unnamed: 4' eliminada correctamente")
    else:
        logger.info("No se encontró columna 'Unnamed: 4'")    
    
    ruta = DATA_PROCESSED / "data_procesada.csv"
    df.to_csv(ruta, index=False)
    logger.info(f"Datos procesados completos guardados en {ruta}")
    return df


def procesar_datos_completos():
    """Función principal que ejecuta todo el pipeline"""
    from src.ingesta import cargar_datos
    
    df = cargar_datos()
    df = tratar_valores_censurados(df)
    df = crear_target(df)
    df = guardar_datos_procesados(df)
    
    # Balanceo y división
    df_train_bal, df_test = balancear_y_dividir(df, random_state=42, ratio='auto')
    
    if df_train_bal is not None and df_test is not None:
        logger.info("Datos preprocesados y balanceados guardados correctamente.")
    
    return df_train_bal, df_test


if __name__ == "__main__":
    procesar_datos_completos()
