"""
Gráficos de resultados
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from src.config import logger

def generar_graficos_desempeno(modelos, X_test_scaled, y_test):
    """Genera y guarda los gráficos de desempeño del baseline en results/"""
    os.makedirs("results", exist_ok=True)
    logger.info("Generando gráficos de desempeño del baseline...")
    
    nombres = list(modelos.keys())
    accuracies = []
    f1_scores = []
    
    # 1. Gráfico de comparación general
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        accuracies.append(acc)
        f1_scores.append(f1)
    
    x = np.arange(len(nombres))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width/2, accuracies, width, label='Accuracy', color='blue')
    ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='red')
    
    ax.set_ylabel('Métrica')
    ax.set_title('Comparación de Desempeño - Modelos Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
        ax.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center')
        ax.text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/comparacion_modelos_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Matriz de confusión para cada modelo
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Au', 'Au > 0.10'])
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f'Matriz de Confusión - {nombre}')
        plt.savefig(f'results/confusion_matrix_{nombre.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Todos los gráficos guardados en carpeta results/")
    print("Gráficos guardados en results")