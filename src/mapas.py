"""
Módulo para mapas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generar_mapas(modelos_clf, modelos_reg, df, scaler, features):
    """
    Genera los 16 mapas combinados (4 clasificadores × 4 regresores)
    """
    print(f"\n{'#'*90}")
    print("GENERANDO 16 MAPAS (Clasificación y Regresión)")
    print(f"{'#'*90}\n")
    
    os.makedirs("results", exist_ok=True)
    X_full_scaled = scaler.transform(df[features])
    
    for clf_name, clf_model in modelos_clf.items():
        proba = clf_model.predict_proba(X_full_scaled)[:, 1]
        pred_target = clf_model.predict(X_full_scaled)
        
        for reg_name, reg_model in modelos_reg.items():
            pred_log = reg_model.predict(X_full_scaled)
            pred_ppm = np.expm1(pred_log)
            
            df_combo = df[['East', 'North', 'Level', 'Au_ppm', 'target_Au']].copy()
            df_combo['Probabilidad'] = proba
            df_combo['Au_Predicho_ppm'] = pred_ppm
            df_combo['Pred_Target'] = pred_target
            
            df_pos = df_combo[df_combo['Pred_Target'] == 1].copy()
            
            if df_pos.empty:
                continue
            
            # MAPA
            plt.figure(figsize=(14, 10))
            
            # Definir rangos de tamaño según concentración
            sizes = []
            labels = []
            for ppm in df_pos['Au_Predicho_ppm']:
                if ppm < 1:
                    sizes.append(75)
                    labels.append('Baja (<1 ppm)')
                elif ppm < 5:
                    sizes.append(150)
                    labels.append('Media (1-5 ppm)')
                elif ppm < 10:
                    sizes.append(300)
                    labels.append('Alta (5-10 ppm)')
                else:
                    sizes.append(600)
                    labels.append('Muy Alta (>10 ppm)')
            
            scatter = plt.scatter(
                df_pos['East'], 
                df_pos['North'],
                s=sizes,
                c=df_pos['Probabilidad'],
                cmap='YlOrRd',           # Amarillo claro a Rojo oscuro
                alpha=0.85,
                edgecolors='black',
                linewidth=0.5
            )
            
            plt.colorbar(scatter, label='Probabilidad de Oro', shrink=0.8)
            plt.title(f'Mapa - {clf_name} + {reg_name}\n'
                      f'Tamaño = Ley predicha | Color = Probabilidad (Amarillo → Rojo)',
                      fontsize=15, fontweight='bold')
            plt.xlabel('Este (metros)')
            plt.ylabel('Norte (metros)')
            plt.grid(True, alpha=0.3)
            
            # Leyenda de tamaños
            handles = [plt.scatter([], [], s=75, color='gray', edgecolors='black'),
                       plt.scatter([], [], s=150, color='gray', edgecolors='black'),
                       plt.scatter([], [], s=300, color='gray', edgecolors='black'),
                       plt.scatter([], [], s=600, color='gray', edgecolors='black')]
            
            plt.legend(handles, ['< 1 ppm', '1-5 ppm', '5-10 ppm', '> 10 ppm'],
                      title="Ley Predicha (ppm)", 
                      loc='upper right', fontsize=10)
            
            plt.tight_layout()
            filename = f"mapa_{clf_name}_{reg_name}.png"
            plt.savefig(f"results/{filename}", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Mapa guardado: results/{filename}")
    
    print(f"\n{'#'*90}")
    print("Se generaron los 16 mapas")
    print(f"{'#'*90}")