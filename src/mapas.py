"""
Módulo para mapas

Genera el mapa final de zona de interés Au combinando:
- Modelo Final de CLASIFICACIÓN (XGBoost Optuna + Mitigación A) → color = probabilidad
- Mejor modelo de REGRESIÓN del baseline (mejor_regresor.pkl)   → tamaño = ley Au predicha (ppm)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from src.config import logger, MODELS_DIR, EXCLUDE_COLS, DATA_INTERIM


def _generar_mapa(df_map, titulo, filename, output_dir="results"):
    """
    Función interna para generar un mapa de zona de interés.

    - Color   = probabilidad de anomalía (siempre).
    - Tamaño  = ley Au predicha (ppm) si existe la columna 'Ley_Predicha_ppm';
                en caso contrario, tamaño según probabilidad.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_pos = df_map[df_map['Clase_Predicha'] == 1].copy()

    if len(df_pos) == 0:
        logger.warning(f"No hay puntos con predicción positiva para {filename}")
        return

    logger.info(f"Generando mapa {filename} con {len(df_pos)} puntos positivos")

    plt.figure(figsize=(15, 11))

    usa_ppm = 'Ley_Predicha_ppm' in df_pos.columns

    if usa_ppm:
        # Tamaños según ley predicha (ppm)
        sizes = df_pos['Ley_Predicha_ppm'].apply(
            lambda v: 30 if v < 0.1 else (90 if v < 0.5 else (150 if v < 1.0 else 210))
        ).values
        leyenda_labels = ['< 0.1 ppm', '0.1 – 0.5 ppm', '0.5 – 1.0 ppm', '> 1.0 ppm']
        leyenda_titulo = "Ley Au predicha"
    else:
        # Tamaños según probabilidad
        sizes = df_pos['Probabilidad_Au'].apply(
            lambda p: 30 if p < 0.4 else (90 if p < 0.6 else (150 if p < 0.8 else 210))
        ).values
        leyenda_labels = ['< 0.4', '0.4 – 0.6', '0.6 – 0.8', '> 0.8']
        leyenda_titulo = "Probabilidad"

    scatter = plt.scatter(
        df_pos['East'],
        df_pos['North'],
        s=sizes,
        c=df_pos['Probabilidad_Au'],
        cmap='YlOrRd',
        alpha=0.9,
        edgecolors='black',
        linewidth=0.6
    )

    plt.colorbar(scatter, label='Probabilidad de Anomalía Au', shrink=0.75)
    plt.title(titulo, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Este (metros)')
    plt.ylabel('Norte (metros)')
    plt.grid(True, alpha=0.3)

    handles = [
        plt.scatter([], [], s=30,  color='darkred', edgecolors='black'),
        plt.scatter([], [], s=90,  color='darkred', edgecolors='black'),
        plt.scatter([], [], s=150, color='darkred', edgecolors='black'),
        plt.scatter([], [], s=210, color='darkred', edgecolors='black'),
    ]
    plt.legend(handles,
               leyenda_labels,
               title=leyenda_titulo,
               loc='upper right',
               fontsize=10,
               title_fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=400, bbox_inches='tight')
    plt.close()
    logger.info(f"Mapa guardado: {output_dir}/{filename}")


def _predecir_ley_regresor(df_test_orig):
    """
    Carga el mejor regresor guardado por src/modelo_baseline.py y predice
    la ley Au (ppm) sobre el test original (datos SIN Feature Engineering).

    Returns:
        np.array con ppm predichos, o None si el regresor no existe.
    """
    regresor_path = MODELS_DIR / "mejor_regresor.pkl"
    if not regresor_path.exists():
        logger.warning(f"No se encontró {regresor_path}. "
                       "Ejecuta primero: python -m src.modelo_baseline")
        return None

    artefacto = joblib.load(regresor_path)
    modelo   = artefacto['modelo']
    scaler   = artefacto['scaler']
    features = artefacto['features']
    nombre   = artefacto.get('nombre', 'regresor')

    faltantes = [f for f in features if f not in df_test_orig.columns]
    if faltantes:
        logger.warning(f"Faltan features para el regresor: {faltantes[:5]}...")
        return None

    X = scaler.transform(df_test_orig[features])
    y_pred_log = modelo.predict(X)
    y_pred_ppm = np.expm1(y_pred_log)  # invertir log1p

    logger.info(f"Ley Au predicha con regresor '{nombre}' "
                f"(rango: {y_pred_ppm.min():.3f} – {y_pred_ppm.max():.3f} ppm)")
    return y_pred_ppm


def generar_mapa_modelo_final():
    """
    Genera el mapa final de zona de interés combinando:
    - XGBoost Optuna + Mitigación A (clasificación) → color = probabilidad
    - Mejor regresor del baseline → tamaño = ley Au predicha (ppm)

    Si el regresor no está disponible, el tamaño se basa en la probabilidad.
    """
    logger.info("=== Generando Mapa Final (XGBoost + Mitigación A + Mejor Regresor) ===")

    # Cargar modelo de clasificación final y scaler
    modelo_path = MODELS_DIR / "modelo_final_xgboost.pkl"
    scaler_path = MODELS_DIR / "modelo_final_scaler.pkl"

    # Si no existe el modelo final, intentar con el de sprint3
    if not modelo_path.exists():
        modelo_path = MODELS_DIR / "sprint3_xgboost_optuna.pkl"
        scaler_path = MODELS_DIR / "sprint3_xgboost_optuna_scaler.pkl"
        logger.info("Usando modelo sprint3_xgboost_optuna (modelo_final no encontrado)")

    if not modelo_path.exists():
        logger.error(f"Modelo no encontrado: {modelo_path}. "
                     "Ejecuta primero: python -m src.modelo_final")
        return

    model = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)

    # Cargar datos
    df_test_fe = pd.read_csv(DATA_INTERIM / "data_prueba_fe.csv")
    df_test_orig = pd.read_csv(DATA_INTERIM / "data_prueba.csv")

    feature_cols = [c for c in df_test_fe.columns if c not in EXCLUDE_COLS]
    X_test = df_test_fe[feature_cols]
    X_test_scaled = scaler.transform(X_test)

    proba = model.predict_proba(X_test_scaled)[:, 1]
    pred_class = (proba >= 0.5).astype(int)

    # Aplicar Mitigación A si hay umbrales guardados
    umbrales_path = os.path.join("results", "umbrales_mitigacion_a.csv")
    if os.path.exists(umbrales_path):
        from src.modelo_final import crear_slices
        df_test_full = df_test_fe.copy()
        df_test_full['East']  = df_test_orig['East'].values[:len(df_test_fe)]
        df_test_full['North'] = df_test_orig['North'].values[:len(df_test_fe)]
        df_test_full['Level'] = df_test_orig['Level'].values[:len(df_test_fe)]

        df_sliced, _, _, _, _ = crear_slices(
            df_test_full, df_test_fe['target_Au'].values, pred_class, proba
        )

        umbrales_df = pd.read_csv(umbrales_path)
        for _, row in umbrales_df.iterrows():
            col = row['slice_col']
            cat = row['slice']
            umb = row['umbral']
            if col in df_sliced.columns:
                mask = df_sliced[col] == cat
                pos = np.where(mask)[0]
                if len(pos) > 0:
                    pred_class[pos] = (proba[pos] >= umb).astype(int)
                    logger.info(f"Mitigación A aplicada: {cat} umbral={umb:.2f}")

    # Predecir ley Au (ppm) con el mejor regresor guardado
    ley_ppm = _predecir_ley_regresor(df_test_orig)

    df_map = pd.DataFrame({
        'East':  df_test_orig['East'].values[:len(df_test_fe)],
        'North': df_test_orig['North'].values[:len(df_test_fe)],
        'Probabilidad_Au': proba,
        'Clase_Predicha': pred_class,
    })

    if ley_ppm is not None:
        df_map['Ley_Predicha_ppm'] = ley_ppm[:len(df_test_fe)]
        subtitulo = 'Color = Probabilidad de anomalía | Tamaño = Ley Au predicha (ppm)'
    else:
        subtitulo = 'Tamaño y Color = Probabilidad de anomalía'
        logger.warning("Mapa generado sin regresor: tamaño según probabilidad")

    _generar_mapa(
        df_map,
        titulo='MAPA FINAL DE ZONA DE INTERÉS Au\n'
               'XGBoost Optuna + Mitigación A (F1 = 0.9808) + Mejor Regresor\n'
               f'{subtitulo}',
        filename='mapa_XGBoost_Optuna_MitigacionA.png'
    )


if __name__ == "__main__":
    generar_mapa_modelo_final()
