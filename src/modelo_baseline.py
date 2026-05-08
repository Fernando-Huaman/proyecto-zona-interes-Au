"""
Modelos Baseline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.config import logger
from src.evaluacion import limpiar_archivo_resultados, evaluar_holdout, evaluar_cross_validation, evaluar_regresion
from src.ranking import generar_rankings_y_visualizaciones
from src.graficos import generar_graficos_desempeno, generar_graficos_regresion

def ejecutar_baseline():
    logger.info("Iniciando Modelos Baseline...")
    
    # Limpiar resultados anteriores
    limpiar_archivo_resultados()
    print("Archivo de resultados limpiado\n")
    
    # Cargar datos procesados
    df = pd.read_csv("data/processed/data_procesada.csv")
    
    features = [col for col in df.columns if col not in ['Au_ppm', 'target_Au', 'East', 'North', 'Level']]
    X = df[features]
    y = df['target_Au']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
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
    
    df_pos = df[df['target_Au'] == 1].copy()
    X_pos = df_pos[features]
    y_pos = np.log1p(df_pos['Au_ppm'])
    X_pos_scaled = scaler.transform(X_pos)
    
    modelos_reg = {
        'KNN_Regressor': KNeighborsRegressor(n_neighbors=5),
        'Ridge': Ridge(alpha=1.0),
        'Arbol_Decision_Regressor': DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random_Forest_Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    for nombre, modelo in modelos_reg.items():
        print(f"ENTRENANDO: {nombre}")
        modelo.fit(X_pos_scaled, y_pos)
        evaluar_regresion(modelo, X_pos_scaled, df_pos['Au_ppm'], nombre)

    generar_rankings_y_visualizaciones(modelos_clf, modelos_reg, df, scaler, features)

    generar_graficos_desempeno(modelos_clf, X_test_scaled, y_test)
    generar_graficos_regresion(modelos_reg, X_pos_scaled, df_pos['Au_ppm'])
    
    print(f"\n{'#'*90}")
    print("BASELINE COMPLETO")
    print(f"Resultados guardados en → results/evaluacion_resultados.txt")
    print(f"{'#'*90}")

if __name__ == "__main__":
    ejecutar_baseline()
