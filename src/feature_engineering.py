"""
Feature Engineering
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from src.config import logger

def safe_log1p(X):
    """Log transform seguro para datos geoquímicos"""
    return np.log1p(np.clip(X, 0, None))

def crear_features_geoquimicos(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando Feature Engineering...")
    df_fe = df.copy()
    
    element_cols = [col for col in df_fe.columns 
                    if col not in ['Au_ppm', 'target_Au', 'East', 'North', 'Level', 
                                  'is_synthetic', 'Codigo', 'target_regression']]
    
    # 1. Transformaciones logarítmicas
    for col in element_cols:
        if df_fe[col].min() >= -1e-6:
            df_fe[f'{col}_log'] = safe_log1p(df_fe[col])
    
    # 2. Ratios clave (Pathfinders de Oro)
    ratios = {
        'Au_Cu_ratio': ('Au_ppm', 'Cu_ppm'),
        'Au_Ag_ratio': ('Au_ppm', 'Ag_ppm'),
        'Au_As_ratio': ('Au_ppm', 'As_ppm'),
        'Au_Sb_ratio': ('Au_ppm', 'Sb_ppm'),
        'As_Sb_ratio': ('As_ppm', 'Sb_ppm'),
        'Cu_Ag_ratio': ('Cu_ppm', 'Ag_ppm'),
        'Pb_Zn_ratio': ('Pb_ppm', 'Zn_ppm'),
        'Bi_As_ratio': ('Bi_ppm', 'As_ppm'),
    }
    
    for name, (num, den) in ratios.items():
        if num in df_fe.columns and den in df_fe.columns:
            df_fe[name] = df_fe[num] / (df_fe[den].clip(lower=1e-6))
    
    # 3. Agregaciones de pathfinders
    pathfinders = ['As_ppm', 'Sb_ppm', 'Cu_ppm', 'Ag_ppm', 'Bi_ppm', 'Pb_ppm', 'Zn_ppm']
    available = [col for col in pathfinders if col in df_fe.columns]
    if available:
        df_fe['pathfinder_sum'] = df_fe[available].sum(axis=1)
        df_fe['pathfinder_mean'] = df_fe[available].mean(axis=1)
        df_fe['pathfinder_max'] = df_fe[available].max(axis=1)
   
    logger.info(f"FE completado: {df.shape[1]} → {df_fe.shape[1]} columnas")
    return df_fe


def get_fe_pipeline():
    return Pipeline([
        ('log_transform', FunctionTransformer(safe_log1p)),
        ('scaler', StandardScaler())
    ])


def aplicar_feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame):
    df_train_fe = crear_features_geoquimicos(df_train)
    df_test_fe = crear_features_geoquimicos(df_test)
    return df_train_fe, df_test_fe