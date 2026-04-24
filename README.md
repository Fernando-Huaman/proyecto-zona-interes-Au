# Proyecto Zona de Interés de Oro

**Detección de zonas de interés de oro (> 0.10 ppm Au) usando datos geoquímicos y Machine Learning.**

---

👥 **Autores**
- Fernando Huaman Sanchez
- Gabriel Vilcahuaman Canchanya

---

📊 **Dataset**
- **Fuente**: DataSet_final.csv (muestreos geoquímicos)
- **Registros**: 2,137 muestras
- **Variables**: Coordenadas (East, North, Level) + 47 elementos químicos (Au_ppm, Ag_ppm, Cu_ppm, etc.)
- **Objetivo**: Clasificar zonas de interés (`target_Au = 1` si Au > 0.10 ppm)
- **Versión usada**: Abril 2026

---

🗂️ **Estructura del repositorio**

```bash
data/
├── raw/                    # DataSet_final.csv (original)
├── processed/              # data_procesada.csv (limpio y target)
notebooks/
├── EDA.ipynb               # Análisis exploratorio completo
src/
├── config.py               # Configuración y logging
├── ingesta.py              # Carga del dataset original
├── preprocesamiento.py     # Limpieza de censurados y creación de target
├── evaluacion.py           # Hold-out y Validación Cruzada
├── modelo_baseline.py      # Modelos, Entrenamiento y Evaluación
└── run_all.py              # Ejecuta todo el pipeline
results/                    # evaluacion_resultados.txt
notebooks/output/           # Gráficos y tablas del EDA
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
3. Análisis Exploratorio (EDA)
```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/EDA.ipynb
```
Los gráficos y tablas se guardan automáticamente en notebooks/output/

4. Modelos y Evaluación
```bash
python -m src.modelo_baseline
```
Incluye Hold-out y Validación Cruzada (5-Fold). 
Resultados en results/evaluacion_resultados.txt

---

📈 **Resultados esperados**

EDA completo con matriz de correlación, distribuciones normal y lognormal de todos los elementos.
Modelos Baseline: KNN, Regresión Logística, Árbol de Decisión y Random Forest.
Validación Cruzada (5-Fold Stratified).
Archivos generados:
- results/evaluacion_resultados.txt
- notebooks/output/ (imágenes y tablas)
- logs/pipeline.log

---

📌 **Roadmap**

[x] Sprint 1 → Pipeline mínimo reproducible + EDA + Baseline.
[ ] Sprint 2 → Mejora de features, balanceo y modelos avanzados.
[ ] Sprint 3 → Optimización e interpretabilidad.
[ ] Sprint 4 → Resultados finales y defensa.

---

📜 **Licencia**
Uso académico – Universidad Nacional de Ingeniería (UNI).
Proyecto desarrollado como parte de la Maestría en Inteligencia Artificial.