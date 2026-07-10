"""
Gráficos de resultados
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                           accuracy_score, f1_score, roc_curve, auc, 
                           precision_recall_curve, average_precision_score,
                           mean_absolute_error, r2_score, mean_squared_error)
from src.config import logger

def generar_graficos_desempeno(modelos, X_test_scaled, y_test, feature_names=None):
    """Genera todos los gráficos de desempeño"""
    os.makedirs("results", exist_ok=True)
    logger.info("Generando gráficos de desempeño del baseline...")
    
    nombres = list(modelos.keys())
    accuracies = []
    f1_scores = []
    
    # 1. Comparación general
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        accuracies.append(acc)
        f1_scores.append(f1)
    
    # Gráfico de barras
    x = np.arange(len(nombres))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#1f77b4')
    ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#ff7f0e')
    
    ax.set_ylabel('Métrica')
    ax.set_title('Comparación de Desempeño - Modelos de Clasificación')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
        ax.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center')
        ax.text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/Comparacion_clasificacion.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Matrices de confusión
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    for i, (nombre, modelo) in enumerate(modelos.items()):
        y_pred = modelo.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Target=0', 'Target=1'])
        disp.plot(ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{nombre}')
    plt.suptitle('Matrices de Confusión - Comparacion de Modelos de Clasificación', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/Comparacion_clasif_matrices_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Curvas ROC
    plt.figure(figsize=(10, 8))
    for nombre, modelo in modelos.items():
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curvas ROC - Comparación de Modelos de Clasificación')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/Comparacion_clasif_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Curvas Precision-Recall
    plt.figure(figsize=(10, 8))
    for nombre, modelo in modelos.items():
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            plt.plot(recall, precision, label=f'{nombre} (AP = {ap:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas Precision-Recall - Comparación de Modelos Clasificación')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/Comparacion_clasif_precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Gráficos de clasificación generados")

def generar_graficos_regresion(modelos_reg, X_test_scaled, y_test_real_ppm):
    """Genera gráficos de Regresión"""
    os.makedirs("results", exist_ok=True)
    logger.info("Generando gráficos de regresión...")
    
    nombres = list(modelos_reg.keys())
    
    # Calcular predicciones
    predicciones = {}
    for nombre, modelo in modelos_reg.items():
        y_pred_log = modelo.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)
        predicciones[nombre] = y_pred
    
    # 1. Comparación General
    maes = [mean_absolute_error(y_test_real_ppm, predicciones[n]) for n in nombres]
    rmses = [np.sqrt(mean_squared_error(y_test_real_ppm, predicciones[n])) for n in nombres]
    r2s = [r2_score(y_test_real_ppm, predicciones[n]) for n in nombres]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].bar(nombres, maes, color='coral')
    axes[0].set_title('MAE (menor es mejor)')
    axes[0].set_ylabel('MAE (ppm)')
    axes[0].tick_params(axis='x', rotation=15)
    
    axes[1].bar(nombres, rmses, color='orange')
    axes[1].set_title('RMSE (menor es mejor)')
    axes[1].set_ylabel('RMSE (ppm)')
    axes[1].tick_params(axis='x', rotation=15)
    
    axes[2].bar(nombres, r2s, color='teal')
    axes[2].set_title('R² (mayor es mejor)')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.suptitle('Comparación de Desempeño - Modelos de Regresión', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/Comparacion_regresion.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Predicted vs Actual
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, nombre in enumerate(nombres):
        y_pred = predicciones[nombre]
        ax = axes[i]
        ax.scatter(y_test_real_ppm, y_pred, alpha=0.6, edgecolors='k', s=40)
        ax.plot([y_test_real_ppm.min(), y_test_real_ppm.max()], 
                [y_test_real_ppm.min(), y_test_real_ppm.max()], 'r--', lw=2)
        ax.set_xlabel('Au Real (ppm)')
        ax.set_ylabel('Au Predicho (ppm)')
        ax.set_title(f'{nombre}\nPredicted vs Actual')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Predicted vs Actual - Modelos de Regresión', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/Comparacion_regre_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Residuals Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, nombre in enumerate(nombres):
        y_pred = predicciones[nombre]
        residuals = y_test_real_ppm - y_pred
        ax = axes[i]
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=40)
        ax.axhline(0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Au Predicho (ppm)')
        ax.set_ylabel('Residuales (Real - Predicho)')
        ax.set_title(f'{nombre}\nResidual Plot')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis de Residuales - Modelos de Regresión', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/Comparacion_regre_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Distribución de Residuales
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, nombre in enumerate(nombres):
        y_pred = predicciones[nombre]
        residuals = y_test_real_ppm - y_pred
        ax = axes[i]
        sns.histplot(residuals, kde=True, ax=ax, color='purple')
        ax.set_title(f'{nombre}\nDistribución de Residuales')
        ax.set_xlabel('Residuales')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribución de Residuales - Modelos de Regresión', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/Comparacion_regre_hist_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Gráficos de regresión generados")


def generar_graficos_modelo_final(y_test, y_prob, y_pred,
                                  nombre="XGBoost_Optuna_MitigacionA",
                                  output_dir="results"):
    """
    Genera los gráficos de clasificación del MODELO FINAL
    (matriz de confusión, curva ROC, curva Precision-Recall y barras de métricas).

    Recibe directamente las predicciones (y_pred) y probabilidades (y_prob)
    para respetar los umbrales de la Mitigación A.
    """
    from sklearn.metrics import precision_score, recall_score
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generando gráficos de clasificación del modelo final ({nombre})...")

    y_test = np.asarray(y_test)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    # --- 1. Matriz de confusión ---
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Target=0', 'Target=1'])
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'Matriz de Confusión — Modelo Final\n{nombre}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ModeloFinal_matriz_confusion.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. Curva ROC ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color='#d62728', lw=2, label=f'{nombre} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC — Modelo Final')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ModeloFinal_roc_curve.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # --- 3. Curva Precision-Recall ---
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(9, 7))
    plt.plot(recall, precision, color='#2ca02c', lw=2, label=f'{nombre} (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall — Modelo Final')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ModeloFinal_precision_recall_curve.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # --- 4. Barras de métricas ---
    metricas = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_test, y_pred),
        'ROC-AUC':   roc_auc,
        'PR-AUC':    ap,
    }
    plt.figure(figsize=(10, 6))
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = plt.bar(list(metricas.keys()), list(metricas.values()), color=colores)
    for bar, valor in zip(bars, metricas.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, valor + 0.005,
                 f'{valor:.4f}', ha='center', fontweight='bold')
    plt.ylim(0, 1.05)
    plt.ylabel('Métrica')
    plt.title(f'Métricas de Clasificación — Modelo Final\n{nombre}')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ModeloFinal_metricas.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Gráficos de clasificación del modelo final generados en results/")
