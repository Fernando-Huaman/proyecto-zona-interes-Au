# Proyecto Zona de Interés de Oro

**Detección de zonas de interés de oro (> 0.10 ppm Au) usando datos geoquímicos y Machine Learning.**

---

👥 **Autores**
- Fernando Huaman Sanchez
- Gabriel Vilcahuaman Canchanya

---

📊 **Dataset**
- **Fuente**: DataSet_final.csv (muestreos geoquímicos)
- **Registros**: 2,132 muestras
- **Variables**: Coordenadas (East, North, Level) + 47 elementos químicos (Au_ppm, Ag_ppm, Cu_ppm, etc.)
- **Objetivo**: Clasificar zonas de interés de oro (`target_Au = 1` si Au > 0.10 ppm) según probabilidad y concentración predicha (Au_ppm)
- **Versión usada**: Abril 2026

---

🗂️ **Estructura del repositorio**

```bash
data/
├── raw/                    # DataSet.csv (original)
├── processed/              # data_procesada.csv (limpio y target)
├── interim/                # data train y prueba (sin/con FE)
notebooks/ 
├── EDA.ipynb               # Análisis exploratorio completo
├── sprint2.ipynb           # Feature Engineering + Experimentos A/B
src/
├── config.py               # Configuración y logging
├── ingesta.py              # Carga del dataset original
├── preprocesamiento.py     # Limpieza, creación de target y balanceo
├── evaluacion.py           # Métricas, Hold-out y Validación Cruzada
├── graficos.py             # Generación de gráficos de desempeño
├── mapas.py                # 1 mapa de probabilidad y predicción Au
├── modelo_baseline.py      # Modelos, Entrenamiento y Evaluación
└── run_all.py              # Ejecuta todo el pipeline
results/                    # Metricas, Gráficos y Mapas
notebooks/output/           # Gráficos y tablas del EDA
notebooks/outputs_sprint2/  # Gráficos y tablas del FE + Exp A/B + Shap + Learning + Calibración
logs/                       # pipeline.log
```

---

⚙️ **Requisitos**

```bash
pip install -r requirements.txt
```

---

🚀 **Cómo ejecutar el pipeline**
Opción recomendada (todo de una vez):
```bash
python src/run_all.py
```
Ejecución paso a paso:
1. Ingesta de datos
```bash
python -m src.ingesta
```
2. Preprocesamiento
```bash
python -m src.preprocesamiento
```
Genera archivos en data/interim y data/processed

3. Análisis Exploratorio (EDA)
```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/EDA.ipynb
```
Los gráficos y tablas se guardan automáticamente en notebooks/output/

4. Modelos y Evaluación
```bash
python -m src.modelo_baseline
```
Incluye Métricas, Hold-out y Validación Cruzada (5-Fold). 
Resultados en results/evaluacion_resultados.txt, Gráficos en results/Comparacion*.png y Mapas en results/mapa*.png.

5. Feature Engineeting y Experimentos A/B
```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/sprint2.ipynb
```
Aplica Feature Engineering (ratios y transformaciones log).
Ejecuta 3 experimentos: Baseline, Var1 (con FE) y Var2 (con FE + Tuning a Random Forest). Resultados en notebooks/outputs_sprint2/metrics_experimentos.csv y Graficos en notebooks/outputs_sprint2/feature_importance*.png      
SHAP, Learning y Calibración. Graficos en notebooks/outputs_sprint2

---

📈 **Resultados esperados**

EDA completo con matriz de correlación, distribuciones de todos los elementos y distribución de la variable objetivo.

Modelo Baseline: 
* Clasificación (KNN, Regresión Logística, Árbol de Decisión y Random Forest)
* Regresión (KNN_Regresor, Ridge, Árbol de Decisión Regressor y Random Forest Regressor)
Validación: Hold-out + Validación Cruzada (5-Fold Stratified).
Archivos generados:
- results/evaluacion_resultados.txt
- results/Comparacion*.png (Comparacion de los modelos)
- results/mapa*.png (Mapa de probabilidad y predicción Au)
- notebooks/output/ (Imágenes y tablas)
- logs/pipeline.log

---

🧪 **Feature Engineering + Experimentos A/B**

*Feature Engineering*   
Se aplicaron las siguientes técnicas:

- Transformaciones logarítmica (log1p) para manejar distribuciones sesgadas
- Ratios Pathfinder: Au_Sb_ratio, Au_Cu_ratio, Au_Ag_ratio, Au_As_ratio, etc.
- Agregaciones: pathfinder_sum, pathfinder_mean, pathfinder_max

Pathfinder son elementos quimicos asocioado al oro (As_ppm, Sb_ppm, Cu_ppm, Ag_ppm, Bi_ppm, Pb_ppm, Zn_ppm)

*Experimentos A/B*  
Se ejecutaron 4 variantes, realizando una modificación por experimento:

| Experimento                  | Features                        | F1 Score   | PR-AUC     | Tiempo (s) |
|------------------------------|---------------------------------|------------|------------|------------|
| **Baseline**              | Datos Balanceados                | 0.6789     | 0.7182     | 1.71       |
| **Var1_FE**               | + Feature Engineering           | 0.8723 | 0.9792 | 1.10       |
| **Var2_FE_tuned RF**         | FE + Tuning                    | 0.9072 | 0.9757     | 0.38       |
| **Var3_FE_optuna XGBoost**         | FE + Optuna                    | **0.9608** | **0.9909**     | 0.38       |

Conclusión principal:   
El **Feature Engineering** generó una mejora muy importante en F1 Score (+0.1934) y PR-AUC (+0.2610).     
El **Tuning** (n=40, depth=11) a Random Forest permitío una mejora adicional en F1 Score (+0.0349) y una disminución pequeña en PR-AUC (-0.0035)    
El **Optuna** a XGBoost (Gradient Boosting) permitío una mejora adicional en F1 Score (+0.0608) y una aumento en PR-AUC (+0.0152)

La **mejor** configuración actual es **Var3**.

Para el Var2 se realizo las siguientes modificaciones:

| Aspecto                  | Baseline y Var1_FE                        | Var2_FE_Tuned   |
|------------------------------|---------------------------------|------------|
| n_estimators             | 100 árboles                | 40 árboles     |
| max_depth               | Ninguno (por defecto ilimitado)          | 11|
| min_samples_leaf         | 1 (por defecto)                    | 2 |

Archivos generados:
- notebooks/outputs_sprint2/metrics_experimentos.csv
- notebooks/FE_*.png
- notebooks/outputs_sprint2/tuning.csv
- notebooks/tuning_*.png
- - notebooks/optuna_*.png
- notebooks/outputs_sprint2/feature_importance.csv

---

📌 **Roadmap**
```bash
[X] Sprint 0 → Pipeline mínimo reproducible + Baseline.
[X] Sprint 1 → EDA + Outliers y Balanceo con BorderlineSMOTE.
[X] Sprint 2 → Feature Engineering + Experimentos A/B.
[ ] Sprint 3 → Resultados finales y defensa.
```

---

📜 **Licencia**
Uso académico – Universidad Nacional de Ingeniería (UNI).
Proyecto desarrollado como parte de la Maestría en Inteligencia Artificial.