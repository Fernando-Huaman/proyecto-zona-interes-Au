"""
Modelos Baseline de REGRESIÓN

Entrena únicamente los modelos de regresión (estimación de ley Au ppm),
compara su desempeño y guarda el MEJOR regresor como .pkl para
reutilizarlo en el mapa final junto al modelo de clasificación final.

La clasificación ya no se ejecuta aquí: el modelo final
(src/modelo_final.py — XGBoost Optuna + Mitigación A) genera sus
propias métricas y gráficos de clasificación.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import logger, MODELS_DIR
from src.evaluacion import limpiar_archivo_resultados, limpiar_carpeta_results, evaluar_regresion
from src.graficos import generar_graficos_regresion

# Límite superior para outliers de Au_ppm
UPPER_LIMIT = 2.20


def ejecutar_baseline():
    logger.info("Iniciando Modelos Baseline de Regresión...")

    # Limpiar resultados anteriores
    limpiar_archivo_resultados()
    print("Archivo de resultados limpiado\n")

    # Limpieza completa de results
    limpiar_carpeta_results()
    print("Carpeta de resultados limpiado\n")

    # Cargar datos
    try:
        df_train = pd.read_csv("data/interim/data_entrenamiento_balanceado.csv")
        logger.info("Cargados datos balanceados desde data/interim/")
    except FileNotFoundError:
        logger.error("No se encontraron archivos en data/interim/. Ejecuta primero preprocesamiento.py")
        return

    # Features (excluir columnas no químicas y el indicador sintético)
    exclude_cols = ['Au_ppm', 'target_Au', 'East', 'North', 'Level', 'is_synthetic']
    features = [col for col in df_train.columns if col not in exclude_cols]

    # MODELOS DE REGRESIÓN
    print(f"\n{'#'*90}")
    print("ENTRENANDO MODELOS DE REGRESIÓN (estimación de ley Au ppm)")
    print(f"{'#'*90}\n")

    # Solo usamos muestras ORIGINALES positivas del train (is_synthetic == 0)
    df_train_pos = df_train[
        (df_train['target_Au'] == 1) &
        (df_train['is_synthetic'] == 0) &
        (df_train['Au_ppm'].notna())
    ].copy()

    df_train_pos['Au_ppm_clipped'] = df_train_pos['Au_ppm'].clip(upper=UPPER_LIMIT)

    logger.info(f"Muestras positivas originales para regresión: {len(df_train_pos)}")

    # Escalado (ajustado sobre las muestras positivas, datos SIN Feature Engineering)
    X_pos = df_train_pos[features]
    y_pos = np.log1p(df_train_pos['Au_ppm_clipped'])

    scaler = StandardScaler()
    X_pos_scaled = scaler.fit_transform(X_pos)

    modelos_reg = {
        'KNN_Regressor': KNeighborsRegressor(n_neighbors=5),
        'Ridge': Ridge(alpha=1.0),
        'Arbol_Decision_Regressor': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random_Forest_Regressor': RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    }

    metricas_reg = {}
    for nombre, modelo in modelos_reg.items():
        print(f"ENTRENANDO: {nombre}")
        modelo.fit(X_pos_scaled, y_pos)
        evaluar_regresion(modelo, X_pos_scaled, df_train_pos['Au_ppm_clipped'], nombre)

        # Métricas en escala real (ppm) para seleccionar el mejor
        y_pred_ppm = np.expm1(modelo.predict(X_pos_scaled))
        metricas_reg[nombre] = {
            'MAE': mean_absolute_error(df_train_pos['Au_ppm_clipped'], y_pred_ppm),
            'RMSE': np.sqrt(mean_squared_error(df_train_pos['Au_ppm_clipped'], y_pred_ppm)),
            'R2': r2_score(df_train_pos['Au_ppm_clipped'], y_pred_ppm),
        }

    # Gráficos de regresión
    generar_graficos_regresion(modelos_reg, X_pos_scaled, df_train_pos['Au_ppm_clipped'])

    # SELECCIÓN Y GUARDADO DEL MEJOR REGRESOR (menor MAE)
    print(f"\n{'#'*90}")
    print("SELECCIÓN DEL MEJOR MODELO DE REGRESIÓN")
    print(f"{'#'*90}\n")

    for nombre, m in metricas_reg.items():
        print(f"  {nombre:30s} | MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | R²={m['R2']:.4f}")

    mejor_nombre = min(metricas_reg, key=lambda n: metricas_reg[n]['MAE'])
    mejor_modelo = modelos_reg[mejor_nombre]

    print(f"\n  → Mejor regresor: {mejor_nombre} (MAE = {metricas_reg[mejor_nombre]['MAE']:.4f})")
    logger.info(f"Mejor regresor seleccionado: {mejor_nombre}")

    # Guardar mejor regresor + scaler + features para reutilizar en el mapa final
    os.makedirs(MODELS_DIR, exist_ok=True)
    artefacto = {
        'modelo': mejor_modelo,
        'scaler': scaler,
        'features': features,
        'nombre': mejor_nombre,
        'target_transform': 'log1p',   # y = log1p(Au_ppm_clipped) → invertir con expm1
        'upper_limit': UPPER_LIMIT,
        'metricas': metricas_reg[mejor_nombre],
    }
    joblib.dump(artefacto, MODELS_DIR / "mejor_regresor.pkl")
    logger.info(f"Mejor regresor guardado: {MODELS_DIR / 'mejor_regresor.pkl'}")

    print(f"\n{'#'*90}")
    print("BASELINE DE REGRESIÓN COMPLETO")
    print(f"  Mejor regresor guardado en → {MODELS_DIR / 'mejor_regresor.pkl'}")
    print(f"  Resultados guardados en   → results/evaluacion_resultados.txt")
    print(f"{'#'*90}")

    return modelos_reg, mejor_nombre


if __name__ == "__main__":
    ejecutar_baseline()
