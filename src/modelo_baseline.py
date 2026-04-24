"""
Baseline Models - Con guardado automático de resultados
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import logger
from src.evaluacion import evaluar_holdout, evaluar_cross_validation

def ejecutar_baseline():
    logger.info("Iniciando Baseline con guardado de resultados...")
    
    # Cargar datos
    df = pd.read_csv("data/processed/data_procesada.csv")
    
    features = [col for col in df.columns if col not in ['Au_ppm', 'target_Au', 'East', 'North', 'Level']]
    X = df[features]
    y = df['target_Au']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    modelos = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Regresion_Logistica': LogisticRegression(max_iter=1000, random_state=42),
        'Arbol_Decision': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }
    
    for nombre, modelo in modelos.items():
        logger.info(f"Entrenando {nombre}...")
        modelo.fit(X_train_scaled, y_train)
        
        evaluar_holdout(modelo, X_test_scaled, y_test, nombre)
        evaluar_cross_validation(modelo, X_train_scaled, y_train, nombre)
    
    logger.info("Pipeline completado. Resultados guardados en carpeta 'results/'")
    print("\n¡Todo finalizado! Revisa la carpeta 'results/evaluacion_resultados.txt'")

if __name__ == "__main__":
    ejecutar_baseline()