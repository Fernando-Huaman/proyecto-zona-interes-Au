"""
Módulo para rankings, priorización y visualizaciones
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generar_rankings_y_visualizaciones(modelos_clf, modelos_reg, df, scaler, features):
    """
    Genera todos los rankings y gráficos de priorización
    """
    print(f"\n{'#'*90}")
    print("GENERANDO RANKINGS Y VISUALIZACIONES DE PRIORIZACIÓN")
    print(f"{'#'*90}\n")
    
    os.makedirs("results", exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    X_full_scaled = scaler.transform(df[features])
    ranking_base = df[['East', 'North', 'Level', 'Au_ppm', 'target_Au']].copy()
    
    # Clasificación - Probabilidad
    for nombre, modelo in modelos_clf.items():
        proba = modelo.predict_proba(X_full_scaled)[:, 1]
        pred = modelo.predict(X_full_scaled)
        
        ranking_base[f'{nombre}_pred'] = pred
        ranking_base[f'{nombre}_proba'] = proba
        
        positivos = ranking_base[ranking_base[f'{nombre}_pred'] == 1].copy()
        if not positivos.empty:
            positivos = positivos.sort_values(by=f'{nombre}_proba', ascending=False).reset_index(drop=True)
            positivos.insert(0, 'Rank', positivos.index + 1)
            positivos.to_csv(f"results/ranking_{nombre}_probabilidad.csv", index=False)
            
            generar_top10_barras(positivos.head(10), nombre, 'proba')
            generar_mapa_espacial(positivos, nombre, 'proba')
    
    # Regresión - Concentración
    for nombre, modelo in modelos_reg.items():
        pred_log = modelo.predict(X_full_scaled)
        ranking_base[f'{nombre}_pred_ppm'] = np.expm1(pred_log)
        
        ranking_reg = ranking_base.sort_values(by=f'{nombre}_pred_ppm', ascending=False).reset_index(drop=True)
        ranking_reg.insert(0, 'Rank', ranking_reg.index + 1)
        ranking_reg.to_csv(f"results/ranking_{nombre}_concentracion.csv", index=False)
        
        generar_top10_barras(ranking_reg.head(20), nombre, 'ppm')
        generar_mapa_espacial(ranking_reg.head(30), nombre, 'ppm')
    
    # Ranking combinado
    if 'Random_Forest' in modelos_clf and 'Random_Forest_Regressor' in modelos_reg:
        ranking_combinado = ranking_base[ranking_base['Random_Forest_pred'] == 1].copy()
        if not ranking_combinado.empty:
            ranking_combinado = ranking_combinado.sort_values(
                by=['Random_Forest_proba', 'Random_Forest_Regressor_pred_ppm'],
                ascending=[False, False]
            ).reset_index(drop=True)
            ranking_combinado.insert(0, 'Rank', ranking_combinado.index + 1)
            ranking_combinado[['Rank', 'East', 'North', 'Level', 'Au_ppm',
                             'Random_Forest_proba', 'Random_Forest_Regressor_pred_ppm']].to_csv(
                "results/ranking_final_recomendado.csv", index=False)
            print(f"Ranking combinado guardado ({len(ranking_combinado)} zonas)")
    
    ranking_base.to_csv("results/ranking_todos_modelos.csv", index=False)
    print("Todos los rankings generados")


def generar_top10_barras(df_rank, nombre_modelo, tipo):
    plt.figure(figsize=(14, 7))
    top10 = df_rank.head(10)
    
    if tipo == 'proba':
        y_col = [c for c in df_rank.columns if '_proba' in c][0]
        titulo = f'Top 10 - {nombre_modelo} (Probabilidad)'
        ylabel = 'Probabilidad de Oro'
    else:
        y_col = [c for c in df_rank.columns if '_pred_ppm' in c][0]
        titulo = f'Top 20 - {nombre_modelo} (Concentración ppm)'
        ylabel = 'Au predicho (ppm)'
    
    sns.barplot(x='Rank', y=y_col, data=top10, palette='viridis')
    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.ylabel(ylabel)
    plt.ylim(0, top10[y_col].max() * 1.1)
    
    for i, v in enumerate(top10[y_col]):
        plt.text(i, v * 1.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"results/top10_{nombre_modelo}_{tipo}.png", dpi=300, bbox_inches='tight')
    plt.close()


def generar_mapa_espacial(df_rank, nombre_modelo, tipo):
    plt.figure(figsize=(12, 9))
    if tipo == 'proba':
        color_col = [c for c in df_rank.columns if '_proba' in c][0]
        cbar_label = 'Probabilidad'
    else:
        color_col = [c for c in df_rank.columns if '_pred_ppm' in c][0]
        cbar_label = 'Au predicho (ppm)'
    
    scatter = plt.scatter(df_rank['East'], df_rank['North'],
                         c=df_rank[color_col], s=80, cmap='plasma', alpha=0.9, edgecolors='black')
    
    plt.colorbar(scatter, label=cbar_label)
    plt.title(f'Mapa - {nombre_modelo}', fontsize=16, fontweight='bold')
    plt.xlabel('Este (metros)')
    plt.ylabel('Norte (metros)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/mapa_{nombre_modelo}_{tipo}.png", dpi=300, bbox_inches='tight')
    plt.close()