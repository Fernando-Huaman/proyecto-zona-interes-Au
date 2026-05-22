"""
Experimentos A/B
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import traceback

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_recall_curve, auc, 
                           roc_curve, roc_auc_score, confusion_matrix,
                           average_precision_score)

from src.config import logger
from src.feature_engineering import aplicar_feature_engineering, get_fe_pipeline

RESULTS_CSV = "results/metrics_experimentos.csv"


def log_experiment(nombre: str, config: dict, metricas: dict):
    # Registra experimento sin columnas extras
    try:
        os.makedirs("results", exist_ok=True)
        clean_config = {k: v for k, v in config.items() if k not in metricas.keys()}
        row = {**clean_config, **metricas, 'experimento': nombre}
        df = pd.DataFrame([row])
        
        if os.path.exists(RESULTS_CSV):
            df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
        else:
            df.to_csv(RESULTS_CSV, index=False)
        
        logger.info(f"{nombre} → F1={metricas.get('f1',0):.4f} | PR-AUC={metricas.get('pr_auc',0):.4f}")
    except Exception as e:
        logger.error(f"Error guardando {nombre}: {e}")

def safe_pr_auc(y_true, y_prob):
    #Calcula PR-AUC de forma segura
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        # Ordenar para evitar error de auc
        sorted_idx = np.argsort(recall)
        return auc(recall[sorted_idx], precision[sorted_idx])
    except:
        # Alternativa más robusta
        return average_precision_score(y_true, y_prob)

def plot_combined_feature_importance(models_dict, feature_names_dict, save_path="results/FE_Importancia.png"):
    # Gráfico combinado de importancia de las 3 variantes
    try:
        n = len(models_dict)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 8))
        if n == 1:
            axes = [axes]

        for i, (name, model) in enumerate(models_dict.items()):
            rf_model = model[-1] if hasattr(model, 'steps') else model
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            feat_names = feature_names_dict[name]
            top_features = [feat_names[j] for j in indices]

            sns.barplot(x=importances[indices], y=top_features, ax=axes[i], palette='viridis')
            axes[i].set_title(f'Importancia - {name}', fontsize=12)
            axes[i].set_xlabel('Importancia')
            axes[i].tick_params(axis='y', labelsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"FE_Importancia.png guardado")
    except Exception as e:
        print(f"Error en gráfico de importancia: {e}")


def plot_comparative_bar(results_df, save_path="results/FE_comparacion.png"):
    #Gráfico de barras comparativo
    try:
        plt.figure(figsize=(12, 7))
        x = np.arange(len(results_df))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, results_df['f1'], width, 
                       label='F1 Score', color='skyblue', alpha=0.9)
        bars2 = plt.bar(x + width/2, results_df['pr_auc'], width, 
                       label='PR-AUC', color='orange', alpha=0.9)
        
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(x, results_df['experimento'], rotation=15, ha='right', fontsize=11)
        plt.ylabel('Score', fontsize=12)
        plt.title('Comparación de Métricas - Experimentos A/B', fontsize=14, pad=20)
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        
        plt.ylim(0, max(results_df[['f1', 'pr_auc']].max()) * 1.12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"FE_comparacion.png guardado (con valores en barras)")
        
    except Exception as e:
        print(f"Error en gráfico de barras: {e}")

def plot_pr_curves(models_dict, X_test_dict, y_test, save_path="results/FE_curves_precision_recall.png"):
    # Curvas Precision-Recall comparativas
    try:
        plt.figure(figsize=(10, 7))
        for name, model in models_dict.items():
            X_test = X_test_dict[name]
            y_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{name} (PR-AUC={pr_auc:.4f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curvas Precision-Recall Comparativas')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"FE_curves_precision_recall.png guardado")
    except Exception as e:
        print(f"Error en curvas PR: {e}")

def plot_roc_curves(models_dict, X_test_dict, y_test, save_path="results/FE_curves_roc.png"):
    # Curvas ROC comparativas
    try:
        plt.figure(figsize=(10, 7))
        for name, model in models_dict.items():
            X_test = X_test_dict[name]
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curvas ROC Comparativas')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"FE_curves_roc.png guardado")
    except Exception as e:
        print(f"Error en curvas ROC: {e}")

def plot_confusion_matrices(models_dict, X_test_dict, y_test, save_path="results/FE_matrices_confusion.png"):
    # Comparación de matrices de confusión
    try:
        n = len(models_dict)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
        if n == 1:
            axes = [axes]

        for i, (name, model) in enumerate(models_dict.items()):
            X_test = X_test_dict[name]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Matriz de Confusión - {name}')
            axes[i].set_xlabel('Predicho')
            axes[i].set_ylabel('Real')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"FE_matrices_confusion.png guardado")
    except Exception as e:
        print(f"Error en matrices de confusión: {e}")


def ejecutar_experimentos_ab():
    logger.info("="*90)
    logger.info("INICIANDO EXPERIMENTOS A/B + FEATURE ENGINEERING")
    logger.info("="*90)

    try:
        # Cargar datos
        df_train = pd.read_csv("data/interim/data_entrenamiento_balanceado.csv")
        df_test = pd.read_csv("data/interim/data_prueba.csv")

        exclude = ['Au_ppm', 'target_Au', 'East', 'North', 'Level', 'is_synthetic', 'Codigo']
        features = [col for col in df_train.columns if col not in exclude]

        X_train = df_train[features]
        y_train = df_train['target_Au']
        X_test = df_test[features]
        y_test = df_test['target_Au']

        #BASELINE
        start = time.time()
        baseline = make_pipeline(StandardScaler(), 
                               RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        baseline.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = baseline.predict(X_test)
        y_prob = baseline.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred)
        pr_auc = safe_pr_auc(y_test, y_prob)

        log_experiment("Baseline_RF", 
                      {'model': 'RandomForest', 'features': 'originales', 'fe': False},
                      {'f1': f1, 'pr_auc': pr_auc, 'train_time_sec': round(train_time, 2)})

        # VAR1 y VAR2 (CON FEATURE ENGINEERING)
        df_train_fe, df_test_fe = aplicar_feature_engineering(df_train, df_test)
        df_train_fe.to_csv("data/interim/data_entrenamiento_balanceado_fe.csv", index=False)
        df_test_fe.to_csv("data/interim/data_prueba_fe.csv", index=False)

        features_fe = [col for col in df_train_fe.columns if col not in exclude]
        X_train_fe = df_train_fe[features_fe]
        X_test_fe = df_test_fe[features_fe]

        # VAR1: Solo Feature Engineering
        start = time.time()
        var1 = make_pipeline(get_fe_pipeline(), 
                           RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        var1.fit(X_train_fe, y_train)
        train_time_v1 = time.time() - start

        y_pred_v1 = var1.predict(X_test_fe)
        y_prob_v1 = var1.predict_proba(X_test_fe)[:, 1]
        f1_v1 = f1_score(y_test, y_pred_v1)
        pr_auc_v1 = safe_pr_auc(y_test, y_prob_v1)

        log_experiment("Var1_FE_RF", 
                      {'model': 'RandomForest', 'features': 'con_FE', 'fe': True},
                      {'f1': f1_v1, 'pr_auc': pr_auc_v1, 'train_time_sec': round(train_time_v1, 2)})

        # VAR2: FE + Tuning
        start = time.time()
        var2 = make_pipeline(get_fe_pipeline(), 
                           RandomForestClassifier(n_estimators=200, max_depth=15, 
                                                min_samples_leaf=2, random_state=42, class_weight='balanced'))
        var2.fit(X_train_fe, y_train)
        train_time_v2 = time.time() - start

        y_pred_v2 = var2.predict(X_test_fe)
        y_prob_v2 = var2.predict_proba(X_test_fe)[:, 1]
        f1_v2 = f1_score(y_test, y_pred_v2)
        pr_auc_v2 = safe_pr_auc(y_test, y_prob_v2)

        log_experiment("Var2_FE_RF_tuned", 
                      {'model': 'RandomForest_n200_depth15', 'features': 'con_FE', 'fe': True},
                      {'f1': f1_v2, 'pr_auc': pr_auc_v2, 'train_time_sec': round(train_time_v2, 2)})
        # GRÁFICOS
        print("\nGenerando gráficos comparativos...")
        models_dict = {
            "Baseline": baseline,
            "Var1_FE": var1,
            "Var2_Tuned": var2
        }

        feature_names_dict = {
            "Baseline": features,
            "Var1_FE": features_fe,
            "Var2_Tuned": features_fe
        }

        X_test_dict = {
            "Baseline": X_test,
            "Var1_FE": X_test_fe,
            "Var2_Tuned": X_test_fe
        }

        # Generar todos los gráficos
        plot_combined_feature_importance(models_dict, feature_names_dict)
        plot_comparative_bar(pd.read_csv(RESULTS_CSV))
        plot_pr_curves(models_dict, X_test_dict, y_test)
        plot_roc_curves(models_dict, X_test_dict, y_test)
        plot_confusion_matrices(models_dict, X_test_dict, y_test)

        # Tabla final
        print("\n" + "="*100)
        print("TABLA COMPARATIVA FINAL")
        print("="*100)
        df_results = pd.read_csv(RESULTS_CSV)
        print(df_results[['experimento', 'model', 'features', 'fe', 'f1', 'pr_auc', 'train_time_sec']])
        print("\nExperimentos completados correctamente.")

    except Exception as e:
        logger.error(f"Error crítico: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    ejecutar_experimentos_ab()