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
proyecto-zona-interes-Au/
├── configs/
│   └── config.yaml                 # Configuración centralizada (rutas, params, mitigación)
├── data/
│   ├── raw/DataSet.csv             # Dataset original
│   ├── processed/data_procesada.csv
│   └── interim/                    # Train/test con y sin FE
├── src/
│   ├── config.py                   # Configuración, rutas y constantes globales
│   ├── ingesta.py                  # Carga del dataset original
│   ├── preprocesamiento.py         # Limpieza, target, balanceo (BorderlineSMOTE)
│   ├── evaluacion.py               # Métricas, Hold-out, CV, Bootstrap IC 95%, Slicing
│   ├── modelo_baseline.py          # Baseline de regresión (guarda mejor_regresor.pkl)
│   ├── modelo_final.py             # XGBoost Optuna + Mitigación A
│   ├── graficos.py                 # Gráficos de desempeño
│   ├── mapas.py                    # Mapa final (XGBoost: probabilidad + regresor: ley ppm)
│   └── run_all.py                  # Orquestador del pipeline completo
├── notebooks/
│   ├── EDA.ipynb                   # Sprint 1: Análisis exploratorio
│   ├── sprint2.ipynb               # Sprint 2: Feature Engineering + Experimentos A/B
│   ├── sprint3.ipynb               # Sprint 3: MLOps (MLflow) + Control de Overfitting
│   ├── sprint4.ipynb               # Sprint 4: Análisis de Errores + Mitigaciones
│   ├── sprint4_comparativo.ipynb   # Sprint 4: Comparativo baseline vs actual + IC 95%
│   ├── sprint4_latencia.ipynb      # Sprint 4: Informe de latencia + ONNX
│   ├── DEMO.ipynb                  # Notebook de defensa/presentación
│   ├── output/                     # Gráficos EDA
│   ├── outputs_sprint2/            # Métricas A/B, tuning, SHAP, calibración
│   ├── outputs_sprint3/            # MLflow, modelos .pkl, tablero de corridas
│   │   └── models/                 # Modelos serializados (XGBoost, RF, scalers)
│   ├── outputs_sprint4/            # Slicing, mitigaciones, dashboards
│   ├── outputs_sprint4_comparativo/ # Tablas y gráficos del comparativo
│   └── outputs_sprint4_latencia/   # Benchmark de latencia, ONNX
├── results/                        # Métricas finales, gráficos, mapas
├── logs/                           # pipeline.log, benchmark_latencia.log
├── requirements.txt
└── README.md
```

---

⚙️ **Requisitos**

```bash
pip install -r requirements.txt
```

Dependencias principales: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `shap`, `mlflow`, `onnxruntime`, `skl2onnx`, `imbalanced-learn`, `matplotlib`, `seaborn`.

---

🚀 **Cómo ejecutar**

#### Opción 1: Pipeline completo (scripts)
```bash
python src/run_all.py
```
Ejecuta: ingesta → preprocesamiento → baseline de regresión (guarda el mejor
regresor) → modelo final (métricas + gráficos de clasificación) → mapa final.

> **Nota:** Los notebooks (EDA, sprint2, sprint3, sprint4) se ejecutan de forma
> independiente y NO forman parte de `run_all.py`. El modelo final requiere los
> archivos `*_fe.csv` que genera `notebooks/sprint2.ipynb` (Feature Engineering).

#### Opción 2: Paso a paso

```bash
# 1. Ingesta de datos
python -m src.ingesta

# 2. Preprocesamiento y balanceo
python -m src.preprocesamiento

# 3. Baseline de REGRESIÓN (guarda mejor_regresor.pkl para el mapa)
python -m src.modelo_baseline

# 4. Modelo Final de CLASIFICACIÓN (XGBoost Optuna + Mitigación A)
#    Genera métricas con IC 95% y gráficos: matriz de confusión, ROC, PR
#    (requiere *_fe.csv de notebooks/sprint2.ipynb)
python -m src.modelo_final

# 5. Mapa final (XGBoost → probabilidad/color + Mejor Regresor → ley ppm/tamaño)
python -m src.mapas
```

#### Notebooks (ejecución independiente)
```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/EDA.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/sprint2.ipynb   # genera *_fe.csv
jupyter nbconvert --execute --to notebook --inplace notebooks/sprint3.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/sprint4.ipynb
```

---

📈 **Resultados**

### Evolución del modelo

| Sprint | Modelo | F1 (Hold-Out) | PR-AUC | Mejora |
|--------|--------|---------------|--------|--------|
| Sprint 0-1 | Random Forest Baseline | 0.6789 | 0.7182 | — |
| Sprint 2 | + Feature Engineering | 0.8723 | 0.9792 | +0.1934 |
| Sprint 2 | + Tuning RF | 0.9072 | 0.9757 | +0.0349 |
| Sprint 3 | + Optuna XGBoost | 0.9608 | 0.9909 | +0.0536 |
| Sprint 4 | + Mitigación A (Umbral por Slice) | **0.9808** | **0.9932** | **+0.0293** |

### Modelo Final — XGBoost Optuna + Mitigación A

| Métrica | Valor |
|---------|-------|
| **F1-Score** | **0.9808** |
| Accuracy | 0.9953 |
| Precision | 0.9808 |
| Recall | 0.9808 |
| PR-AUC | 0.9932 |
| ROC-AUC | 0.9988 |
| Brier Score | 0.0087 |

**Validación Cruzada (5-Fold):** F1 = 0.9896 ± 0.0052

### Feature Engineering
Técnicas aplicadas:
- **Ratios pathfinder**: Au/Cu, Au/Ag, Au/As, Au/Sb, As/Sb, Cu/Ag, Pb/Zn, Bi/As
- **Agregaciones**: pathfinder_sum, pathfinder_mean, pathfinder_max
- **Impacto**: +0.1934 en F1-Score (mejora más significativa del proyecto)

### Mitigación A — Umbral por Slice
Optimización de umbrales de decisión por subpoblación:
- **Profundo** (profundidad > Q66): F1 mejoró de 0.857 → 0.952 (+0.095)
- **NE** (cuadrante noreste): F1 mejoró de 0.889 → 1.000 (+0.111)

### MLOps
- Tracking con **MLflow** (SQLite): parámetros, métricas por fold, artefactos
- 6 corridas registradas con comparación automática
- Modelos serializados (.pkl) + scalers versionados

---

📌 **Roadmap**
```
[✓] Sprint 0 → Pipeline mínimo reproducible + Baseline
[✓] Sprint 1 → EDA + Outliers y Balanceo con BorderlineSMOTE
[✓] Sprint 2 → Feature Engineering + Experimentos A/B + Optuna
[✓] Sprint 3 → MLOps (MLflow) + Control de Overfitting
[✓] Sprint 4 → Análisis de Errores + Mitigaciones → F1 = 0.9808
[✓] Sprint 4 → Comparativo baseline vs actual (IC 95%)
[✓] Sprint 4 → Informe de latencia + optimización ONNX
```

---

📜 **Licencia**
Uso académico — Universidad Nacional de Ingeniería (UNI).
Proyecto desarrollado como parte de la Maestría en Inteligencia Artificial.