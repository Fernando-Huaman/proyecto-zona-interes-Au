"""
Modelos Baseline
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.config import logger
from src.evaluacion import limpiar_archivo_resultados, limpiar_carpeta_results, evaluar_holdout, evaluar_cross_validation, evaluar_regresion
from src.graficos import generar_graficos_desempeno, generar_graficos_regresion
from src.mapas import generar_mapas

def ejecutar_baseline():
    logger.info("Iniciando Modelos Baseline...")
    
    # Limpiar resultados anteriores
    limpiar_archivo_resultados()
    print("Archivo de resultados limpiado\n")
    
    # Limpieza completa de results
    limpiar_carpeta_results()
    print("Carpeta de resultados limpiado\n")

    # Cargar datos
    try:
        df_train = pd.read_csv("data/interim/data_entrenamiento_balanceado.csv")
        df_test = pd.read_csv("data/interim/data_prueba.csv")
        logger.info("Cargados datos balanceados desde data/interim/")
    except FileNotFoundError:
        logger.error("No se encontraron archivos en data/interim/. Ejecuta primero preprocesamiento.py")
        return

    # Features (excluir columnas no químicas y el indicador sintético)
    exclude_cols = ['Au_ppm', 'target_Au', 'East', 'North', 'Level', 'is_synthetic']
    features = [col for col in df_train.columns if col not in exclude_cols]
    
    X_train = df_train[features]
    y_train = df_train['target_Au']
    X_test = df_test[features]
    y_test = df_test['target_Au']

    logger.info(f"Train balanceado: {X_train.shape[0]} filas | Test original: {X_test.shape[0]} filas")
    logger.info(f"Distribución train: {y_train.value_counts().to_dict()}")

    # Escalado (solo features químicas)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # MODELOS DE CLASIFICACIÓN
    print(f"\n{'#'*90}")
    print("ENTRENANDO MODELOS DE CLASIFICACIÓN")
    print(f"{'#'*90}\n")

    modelos_clf = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Regresion_Logistica': LogisticRegression(max_iter=1000, random_state=42),
        'Arbol_Decision': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }

    for nombre, modelo in modelos_clf.items():
        print(f"\n{'#'*90}")
        print(f"ENTRENANDO: {nombre.upper()}")
        print(f"{'#'*90}\n")
        
        modelo.fit(X_train_scaled, y_train)        
        evaluar_holdout(modelo, X_test_scaled, y_test, nombre)
        evaluar_cross_validation(modelo, X_train_scaled, y_train, nombre)

    # MODELOS DE REGRESIÓN
    print(f"\n{'#'*90}")
    print("ENTRENANDO MODELOS DE REGRESIÓN")
    print(f"{'#'*90}\n")
    
    # Solo usamos muestras ORIGINALES del train (is_synthetic == 0)
    df_train_pos = df_train[
        (df_train['target_Au'] == 1) & 
        (df_train['is_synthetic'] == 0) & 
        (df_train['Au_ppm'].notna())
    ].copy()

    UPPER_LIMIT = 2.20  # Outliers

    df_train_pos['Au_ppm_clipped'] = df_train_pos['Au_ppm'].clip(upper=UPPER_LIMIT)

    X_pos = df_train_pos[features]
    y_pos = np.log1p(df_train_pos['Au_ppm_clipped'])
    X_pos_scaled = scaler.transform(X_pos)
    
    modelos_reg = {
        'KNN_Regressor': KNeighborsRegressor(n_neighbors=5),
        'Ridge': Ridge(alpha=1.0),
        'Arbol_Decision_Regressor': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random_Forest_Regressor': RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    }
    
    for nombre, modelo in modelos_reg.items():
        print(f"ENTRENANDO: {nombre}")
        modelo.fit(X_pos_scaled, y_pos)
        evaluar_regresion(modelo, X_pos_scaled, df_train_pos['Au_ppm_clipped'], nombre)

    generar_graficos_desempeno(modelos_clf, X_test_scaled, y_test)
    generar_graficos_regresion(modelos_reg, X_pos_scaled, df_train_pos['Au_ppm_clipped'])

    generar_mapas(modelos_clf, modelos_reg, df_test, scaler, features)
    
    print(f"\n{'#'*90}")
    print("BASELINE COMPLETO")
    print(f"Resultados guardados en → results/evaluacion_resultados.txt")
    print(f"{'#'*90}")

if __name__ == "__main__":
    ejecutar_baseline()
