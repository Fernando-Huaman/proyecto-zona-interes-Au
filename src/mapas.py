"""
Módulo para mapas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import logger

def generar_mapas(modelos_clf, modelos_reg, df_test, scaler, features):
    logger.info("=== Generando Mapa Final (Random Forest + Árbol de Decisión) ===")

    os.makedirs("results", exist_ok=True)

    # Modelos específicos
    clf_name = 'Random_Forest'
    reg_name = 'Arbol_Decision_Regressor'

    if clf_name not in modelos_clf or reg_name not in modelos_reg:
        logger.error(f"No se encontraron los modelos solicitados ({clf_name} o {reg_name})")
        return

    clf_model = modelos_clf[clf_name]
    reg_model = modelos_reg[reg_name]

    # PREDICCIONES
    X_scaled = scaler.transform(df_test[features])

    proba = clf_model.predict_proba(X_scaled)[:, 1]
    pred_class = clf_model.predict(X_scaled)

    pred_log = reg_model.predict(X_scaled)
    pred_ppm = np.expm1(pred_log)
    pred_ppm = np.clip(pred_ppm, 0, 2.20)

    # DataFrame
    df_map = df_test[['East', 'North', 'Level', 'Au_ppm', 'target_Au']].copy()
    df_map['Probabilidad_Au'] = proba
    df_map['Au_Predicho_ppm'] = pred_ppm
    df_map['Clase_Predicha'] = pred_class

    df_pos = df_map[df_map['Clase_Predicha'] == 1].copy()

    if len(df_pos) == 0:
        logger.warning("No hay puntos con predicción positiva")
        return

    logger.info(f"Generando mapa con {len(df_pos)} puntos positivos")

    # MAPA
    plt.figure(figsize=(15, 11))

    # ASIGNACIÓN DE TAMAÑOS
    sizes = []
    for ppm in df_pos['Au_Predicho_ppm']:
        if ppm < 0.5:
            sizes.append(30)
        elif ppm < 1.0:
            sizes.append(90)
        elif ppm < 1.5:
            sizes.append(150)
        elif ppm < 2.0:
            sizes.append(210)
        else:
            sizes.append(270)

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

    plt.title('MAPA FINAL DE ZONA DE INTERÉS Au\n'
              'Random Forest (Clasificación) + Árbol de Decisión (Regresión)\n'
              'Tamaño = Ley predicha (ppm) | Color = Probabilidad', 
              fontsize=14, fontweight='bold', pad=20)

    plt.xlabel('Este (metros)')
    plt.ylabel('Norte (metros)')
    plt.grid(True, alpha=0.3)

    # LEYENDA DE TAMAÑOS
    handles = [
        plt.scatter([], [], s=30,  color='darkred', edgecolors='black'),
        plt.scatter([], [], s=90, color='darkred', edgecolors='black'),
        plt.scatter([], [], s=150, color='darkred', edgecolors='black'),
        plt.scatter([], [], s=210, color='darkred', edgecolors='black'),
        plt.scatter([], [], s=270, color='darkred', edgecolors='black')
    ]

    plt.legend(handles, 
               ['< 0.5 ppm', '0.5 - 1.0 ppm', '1.0 - 1.5 ppm', '1.5 - 2.0 ppm', '> 2.0 ppm'],
               title="Ley Predicha (ppm)", 
               loc='upper right', 
               fontsize=10,
               title_fontsize=11)

    plt.tight_layout()
    filename = "mapa_RandomForest_ArbolDecision.png"
    plt.savefig(f"results/{filename}", dpi=400, bbox_inches='tight')
    plt.close()