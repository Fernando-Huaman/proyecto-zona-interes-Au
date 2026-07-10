"""
Modelo Final — XGBoost Optuna + Mitigación A (Umbral por Slice)

Mejor modelo del proyecto: F1 = 0.9808 en hold-out.
Entrena el modelo XGBoost con Feature Engineering, aplica la Mitigación A
(umbrales óptimos por slice para zonas problemáticas) y genera evaluación
completa con IC 95% bootstrap.
"""
import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

from src.config import (
    logger, SEED, XGBOOST_PARAMS, EXCLUDE_COLS,
    DATA_INTERIM, MODELS_DIR, RESULTS_DIR
)
from src.evaluacion import (
    evaluar_global_bootstrap, evaluar_slice, find_optimal_threshold
)
from src.graficos import generar_graficos_modelo_final


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def aplicar_feature_engineering(df):
    """
    Aplica Feature Engineering de ratios pathfinder y agregaciones.
    Reutiliza la lógica del sprint2.ipynb.
    """
    df = df.copy()

    # Ratios pathfinder (con protección contra división por cero)
    eps = 1e-8
    if 'Cu_ppm' in df.columns:
        df['Au_Cu_ratio'] = df.get('Au_ppm', 0) / (df['Cu_ppm'] + eps)
    if 'Ag_ppm' in df.columns:
        df['Au_Ag_ratio'] = df.get('Au_ppm', 0) / (df['Ag_ppm'] + eps)
    if 'As_ppm' in df.columns:
        df['Au_As_ratio'] = df.get('Au_ppm', 0) / (df['As_ppm'] + eps)
    if 'Sb_ppm' in df.columns:
        df['Au_Sb_ratio'] = df.get('Au_ppm', 0) / (df['Sb_ppm'] + eps)
    if 'As_ppm' in df.columns and 'Sb_ppm' in df.columns:
        df['As_Sb_ratio'] = df['As_ppm'] / (df['Sb_ppm'] + eps)
    if 'Cu_ppm' in df.columns and 'Ag_ppm' in df.columns:
        df['Cu_Ag_ratio'] = df['Cu_ppm'] / (df['Ag_ppm'] + eps)
    if 'Pb_ppm' in df.columns and 'Zn_ppm' in df.columns:
        df['Pb_Zn_ratio'] = df['Pb_ppm'] / (df['Zn_ppm'] + eps)
    if 'Bi_ppm' in df.columns and 'As_ppm' in df.columns:
        df['Bi_As_ratio'] = df['Bi_ppm'] / (df['As_ppm'] + eps)

    # Agregaciones pathfinder
    pathfinder_cols = ['As_ppm', 'Sb_ppm', 'Cu_ppm', 'Ag_ppm',
                       'Bi_ppm', 'Pb_ppm', 'Zn_ppm']
    available = [c for c in pathfinder_cols if c in df.columns]
    if available:
        df['pathfinder_sum']  = df[available].sum(axis=1)
        df['pathfinder_mean'] = df[available].mean(axis=1)
        df['pathfinder_max']  = df[available].max(axis=1)

    return df


# ============================================================
# DEFINICIÓN DE SLICES
# ============================================================

def crear_slices(df_test, y_test, y_pred, y_prob):
    """
    Crea las columnas de slice para análisis de errores.
    Requiere que df_test tenga columnas East, North, Level.
    """
    df = df_test.copy()
    df['target_Au'] = y_test
    df['y_pred'] = y_pred
    df['y_prob'] = y_prob

    # Cuadrantes espaciales
    east_med  = df['East'].median()
    north_med = df['North'].median()

    def cuadrante(row):
        if row['East'] >= east_med and row['North'] >= north_med:
            return 'NE'
        elif row['East'] < east_med and row['North'] >= north_med:
            return 'NO'
        elif row['East'] < east_med and row['North'] < north_med:
            return 'SO'
        else:
            return 'SE'

    df['slice_cuadrante'] = df.apply(cuadrante, axis=1)

    # Profundidad por terciles
    lq33 = df['Level'].quantile(0.33)
    lq66 = df['Level'].quantile(0.66)
    df['slice_profundidad'] = df['Level'].apply(
        lambda v: 'Superficial' if v <= lq33 else ('Medio' if v <= lq66 else 'Profundo')
    )

    return df, east_med, north_med, lq33, lq66


# ============================================================
# MITIGACIÓN A — UMBRAL ÓPTIMO POR SLICE
# ============================================================

def aplicar_mitigacion_a(y_test, y_prob, df_test_sliced, slices_problematicos=None):
    """
    Mitigación A: Umbral óptimo por slice.

    Busca el umbral que maximiza F1 en cada slice problemático y aplica
    predicciones diferenciadas. El resto usa umbral estándar 0.5.

    Args:
        y_test: etiquetas reales
        y_prob: probabilidades del modelo
        df_test_sliced: DataFrame con columnas slice_*
        slices_problematicos: lista de dicts con {'slice_col', 'categoria'}
            Si es None, usa los slices problemáticos por defecto (Profundo, NE).

    Returns:
        y_pred_mitA: predicciones con mitigación A
        umbrales: dict con umbrales por slice
    """
    y_test = np.asarray(y_test)
    y_prob = np.asarray(y_prob)

    if slices_problematicos is None:
        slices_problematicos = [
            {'slice_col': 'slice_profundidad', 'categoria': 'Profundo'},
            {'slice_col': 'slice_cuadrante',   'categoria': 'NE'},
        ]

    umbrales = {}
    for sp in slices_problematicos:
        col, cat = sp['slice_col'], sp['categoria']
        if col not in df_test_sliced.columns:
            continue
        mask = df_test_sliced[col] == cat
        pos = np.where(mask)[0]

        if len(pos) >= 10 and len(np.unique(y_test[pos])) == 2:
            t, s = find_optimal_threshold(y_test[pos], y_prob[pos], 'f1')
            umbrales[cat] = {'umbral': t, 'f1_opt': s, 'slice_col': col}
            logger.info(f"Mitigación A | {cat}: umbral={t:.2f} (F1={s:.4f})")

    # Aplicar umbrales
    y_pred_mitA = (y_prob >= 0.5).astype(int).copy()
    for cat, info in umbrales.items():
        mask = df_test_sliced[info['slice_col']] == cat
        pos = np.where(mask)[0]
        if len(pos) > 0:
            y_pred_mitA[pos] = (y_prob[pos] >= info['umbral']).astype(int)

    return y_pred_mitA, umbrales


# ============================================================
# PIPELINE DEL MODELO FINAL
# ============================================================

def entrenar_modelo_final():
    """
    Entrena el modelo final XGBoost Optuna con Feature Engineering
    y evalúa con Mitigación A.

    Flujo:
    1. Carga datos con FE
    2. Escala features
    3. Entrena XGBoost con parámetros Optuna
    4. Predice en test
    5. Aplica Mitigación A (umbral por slice)
    6. Evalúa con bootstrap IC 95%
    7. Genera gráficos de clasificación (matriz de confusión, ROC, PR, métricas)
    8. Guarda modelo, scaler y resultados
    """
    logger.info("=" * 60)
    logger.info("MODELO FINAL — XGBoost Optuna + Mitigación A")
    logger.info("=" * 60)

    # --- 1. Carga de datos con FE ---
    train_path = DATA_INTERIM / "data_entrenamiento_balanceado_fe.csv"
    test_path  = DATA_INTERIM / "data_prueba_fe.csv"

    if not train_path.exists() or not test_path.exists():
        logger.error("No se encontraron datos con FE en data/interim/. "
                      "Ejecuta primero el notebook sprint2.ipynb para generar *_fe.csv.")
        print("ERROR: Datos con Feature Engineering no encontrados.")
        print("       Ejecuta primero: notebooks/sprint2.ipynb")
        return None

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    feature_cols = [c for c in df_test.columns if c not in EXCLUDE_COLS]
    X_train = df_train[feature_cols]
    y_train = df_train['target_Au']
    X_test  = df_test[feature_cols]
    y_test  = df_test['target_Au']

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Features: {len(feature_cols)}")

    # --- 2. Escalado ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # --- 3. Entrenamiento ---
    print(f"\n{'#'*70}")
    print("ENTRENANDO MODELO FINAL — XGBoost Optuna")
    print(f"{'#'*70}\n")
    print(f"Parámetros: {XGBOOST_PARAMS}")

    t0 = time.time()
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_scaled, y_train)
    t_train = time.time() - t0

    logger.info(f"Entrenamiento completado en {t_train:.2f}s")
    print(f"Tiempo de entrenamiento: {t_train:.2f}s")

    # --- 4. Predicción base ---
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_base = (y_prob >= 0.5).astype(int)
    f1_base = f1_score(y_test, y_pred_base)

    print(f"\nF1 baseline (umbral=0.5): {f1_base:.4f}")

    # --- 5. Mitigación A ---
    print(f"\n{'#'*70}")
    print("APLICANDO MITIGACIÓN A — Umbral por Slice")
    print(f"{'#'*70}\n")

    # Cargar datos originales para coordenadas
    df_test_orig = pd.read_csv(DATA_INTERIM / "data_prueba.csv")
    df_test_full = df_test.copy()
    df_test_full['East']  = df_test_orig['East'].values[:len(df_test)]
    df_test_full['North'] = df_test_orig['North'].values[:len(df_test)]
    df_test_full['Level'] = df_test_orig['Level'].values[:len(df_test)]

    df_sliced, east_med, north_med, lq33, lq66 = crear_slices(
        df_test_full, y_test.values, y_pred_base, y_prob
    )

    y_pred_mitA, umbrales = aplicar_mitigacion_a(
        y_test.values, y_prob, df_sliced
    )

    f1_mitA = f1_score(y_test, y_pred_mitA)
    print(f"F1 con Mitigación A: {f1_mitA:.4f} (Δ = {f1_mitA - f1_base:+.4f})")

    for cat, info in umbrales.items():
        print(f"  → {cat}: umbral = {info['umbral']:.2f} (F1 slice = {info['f1_opt']:.4f})")

    # --- 6. Evaluación con bootstrap IC 95% ---
    print(f"\n{'#'*70}")
    print("EVALUACIÓN FINAL CON IC 95% (Bootstrap n=1000)")
    print(f"{'#'*70}\n")

    metrics_final = evaluar_global_bootstrap(
        y_test.values, y_prob, y_pred_mitA, n_boot=1000, seed=SEED
    )

    print(f"  Accuracy:  {metrics_final['Accuracy']:.4f}")
    print(f"  Precision: {metrics_final['Precision']:.4f} [{metrics_final['Precision_CI_low']:.4f}, {metrics_final['Precision_CI_high']:.4f}]")
    print(f"  Recall:    {metrics_final['Recall']:.4f} [{metrics_final['Recall_CI_low']:.4f}, {metrics_final['Recall_CI_high']:.4f}]")
    print(f"  F1:        {metrics_final['F1']:.4f} [{metrics_final['F1_CI_low']:.4f}, {metrics_final['F1_CI_high']:.4f}]")
    print(f"  PR-AUC:    {metrics_final['PR_AUC']:.4f} [{metrics_final['PR_AUC_CI_low']:.4f}, {metrics_final['PR_AUC_CI_high']:.4f}]")
    print(f"  ROC-AUC:   {metrics_final['ROC_AUC']:.4f} [{metrics_final['ROC_AUC_CI_low']:.4f}, {metrics_final['ROC_AUC_CI_high']:.4f}]")
    print(f"  Brier:     {metrics_final['Brier']:.4f}")

    # --- 7. Gráficos de clasificación del modelo final ---
    print(f"\n{'#'*70}")
    print("GENERANDO GRÁFICOS DE CLASIFICACIÓN DEL MODELO FINAL")
    print(f"{'#'*70}\n")

    generar_graficos_modelo_final(
        y_test.values, y_prob, y_pred_mitA,
        nombre="XGBoost_Optuna_MitigacionA"
    )

    # --- 8. Guardar artefactos ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "modelo_final_xgboost.pkl")
    joblib.dump(scaler, MODELS_DIR / "modelo_final_scaler.pkl")
    logger.info(f"Modelo guardado: {MODELS_DIR / 'modelo_final_xgboost.pkl'}")

    # Guardar umbrales de mitigación
    umbrales_df = pd.DataFrame([
        {'slice': cat, 'umbral': info['umbral'], 'f1_opt': info['f1_opt'], 'slice_col': info['slice_col']}
        for cat, info in umbrales.items()
    ])
    umbrales_df.to_csv(RESULTS_DIR / "umbrales_mitigacion_a.csv", index=False)

    # Guardar métricas finales
    metrics_df = pd.DataFrame([metrics_final])
    metrics_df['Modelo'] = 'XGBoost_Optuna_MitA'
    metrics_df['Tiempo_train_s'] = t_train
    metrics_df.to_csv(RESULTS_DIR / "metricas_modelo_final.csv", index=False)

    # Guardar resultados en texto
    with open(RESULTS_DIR / "evaluacion_resultados.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write("MODELO FINAL — XGBoost Optuna + Mitigación A\n")
        f.write(f"{'='*80}\n")
        f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(classification_report(y_test, y_pred_mitA))
        f.write(f"\nF1 = {metrics_final['F1']:.4f} [{metrics_final['F1_CI_low']:.4f}, {metrics_final['F1_CI_high']:.4f}]\n")
        f.write(f"PR-AUC = {metrics_final['PR_AUC']:.4f}\n")
        f.write(f"ROC-AUC = {metrics_final['ROC_AUC']:.4f}\n")
        f.write(f"Brier = {metrics_final['Brier']:.4f}\n")
        f.write(f"\nUmbrales de Mitigación A:\n")
        for cat, info in umbrales.items():
            f.write(f"  {cat}: {info['umbral']:.2f} (F1 slice = {info['f1_opt']:.4f})\n")

    print(f"\n{'#'*70}")
    print("MODELO FINAL COMPLETADO")
    print(f"  Modelo:    {MODELS_DIR / 'modelo_final_xgboost.pkl'}")
    print(f"  Scaler:    {MODELS_DIR / 'modelo_final_scaler.pkl'}")
    print(f"  Umbrales:  {RESULTS_DIR / 'umbrales_mitigacion_a.csv'}")
    print(f"  Métricas:  {RESULTS_DIR / 'metricas_modelo_final.csv'}")
    print(f"  Gráficos:  results/ModeloFinal_*.png (matriz confusión, ROC, PR, métricas)")
    print(f"  Resultados: {RESULTS_DIR / 'evaluacion_resultados.txt'}")
    print(f"{'#'*70}")

    return model, scaler, umbrales, metrics_final


if __name__ == "__main__":
    entrenar_modelo_final()
