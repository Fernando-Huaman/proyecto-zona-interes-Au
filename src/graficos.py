"""
Gráficos de resultados
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                           accuracy_score, f1_score, roc_curve, auc, 
                           precision_recall_curve, average_precision_score)
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
    ax.set_title('Comparación de Desempeño - Modelos Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
        ax.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center')
        ax.text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/comparacion_modelos_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Matrices de confusión
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    for i, (nombre, modelo) in enumerate(modelos.items()):
        y_pred = modelo.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Au', 'Au > 0.10'])
        disp.plot(ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{nombre}')
    plt.suptitle('Matrices de Confusión - Todos los Modelos', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/matrices_confusion_todas.png', dpi=300, bbox_inches='tight')
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
    plt.title('Curvas ROC - Comparación de Modelos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
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
    plt.title('Curvas Precision-Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Gráficos adicionales generados en results/:")