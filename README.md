# Predicción del Tramo de Carrera Docente (Evaluación Docente Chile 2024) — Metodología CRISP-DM

Proyecto final de la asignatura **Minería de Datos (IEI097)** de la carrera **Ingeniería de Ejecución Informática** (Instituto Profesional Santo Tomás). Desarrolla un flujo completo de aprendizaje automático supervisado multiclase bajo la metodología **CRISP-DM**, instrumentado con **MLflow** para el seguimiento experimental y contrastado con el informe técnico de planificación (`IEI097_Informe_G00.pdf`).

* **Autor:** Gonzalo Matías Pino Araya
* **Grupo / Experimento MLflow:** `IEI097-G00`
* **Entorno de ejecución:** Python 3, Scikit-Learn, Pandas, MLflow 3.18 (Google Colab / GitHub)

---

## 1. Estructura del Repositorio

```text
├── Evaluacion_Final_MD_Pino_Gonzalo.ipynb          # Notebook principal (Secciones 1 a 8 CRISP-DM)
├── IEI097_Informe_G00.pdf                          # Informe técnico de planificación (Actividad 2.13)
├── 20251118_EVALUACION_DOCENTE_2024_20251118_PUBL.csv  # Dataset público MINEDUC (25.197 × 22)
├── evidencias_mlflow/                              # Respaldo visual de trazabilidad experimental
│   ├── 01_lista_runs.png                           # Tabla de ejecuciones y métricas en MLflow
│   ├── 02_comparacion_runs.png                     # Comparación de coordenadas paralelas (4 modelos)
│   └── 03_detalle_runs.png                         # Ficha detallada del modelo ganador
├── .gitignore                                      # Exclusión de artefactos locales temporales (mlruns/, *.db)
└── README.md                                       # Documentación general del proyecto
2. Resumen Metodológico (CRISP-DM)Comprensión del Negocio y los Datos: Modelado de clasificación multiclase sobre la variable objetivo PF_CAT_CARRERA (tramos A, B, C, D y E) a partir de 25.197 registros docentes del proceso evaluativo 2024. Se identificó un fuerte desbalance de clases concentrado en el tramo B (45,42%) frente a los extremos A (6,40%) y E (6,27%).Preparación de los Datos:Prevención de fuga de datos (Data Leakage): Exclusión explícita de PF_PJE (puntaje de portafolio), ya que los tramos de carrera derivan determinísticamente de sus cortes.Limpieza e ingeniería de atributos: Eliminación de PF_ESC (100% valores nulos), identificadores únicos (MRUN, RBD) y etiquetas textuales redundantes. Creación de EDAD a partir de DOC_FEC_NAC con tratamiento de valores centinela (190001).Preprocesamiento sin fuga estadística: Partición estratificada 80/20 (20.157 entrenamiento / 5.040 prueba, random_state=42) e implementación de ColumnTransformer (SimpleImputer + StandardScaler en variables numéricas; OneHotEncoder(handle_unknown="ignore") en variables categóricas, generando 476 atributos).Modelado y Evaluación: Entrenamiento con validación cruzada estratificada y ponderación de clases (class_weight="balanced") para optimizar la métrica guía F1-Macro, contrastando tres familias algorítmicas frente a un modelo base no informativo (DummyClassifier).3. Resultados ExperimentalesModelo (Run Name en MLflow)F1-Macro (CV)F1-Macro (Test)Precision-Macro (Test)Recall-Macro (Test)Accuracy (Test)01_Baseline_DummyClassifier (most_frequent)0.12500.12490.09080.20000.4542*02_Regresion_Logistica_Multinomial (Ganador)0.21000.21360.25400.29800.225803_Arbol_Decision (class_weight="balanced")0.17380.19380.23890.28360.205203_Random_Forest (class_weight="balanced")0.18790.18170.24120.29980.1883*Nota: El DummyClassifier alcanza 45,42% de exactitud global al predecir exclusivamente la clase mayoritaria B, obteniendo Recall = 0.00 en los tramos A, C, D y E. La Regresión Logística Multinomial lidera en F1-Macro (+71,0% relativo sobre el baseline), Precision-Macro y Accuracy efectivo.4. Trazabilidad Experimental en MLflow (IEI097-G00)Lista de Ejecuciones RegistradasComparación Multivariable de Hiperparámetros y MétricasDetalle del Modelo Seleccionado (02_Regresion_Logistica_Multinomial)5. Instrucciones de EjecuciónAbrir Evaluacion_Final_MD_Pino_Gonzalo.ipynb directamente en Google Colab mediante el botón Open in Colab o clonando este repositorio:Bashgit clone [https://github.com/gonmpa/Notebook_Jupyter_Mineria_Datos_Supervisado_Clasificacion.git](https://github.com/gonmpa/Notebook_Jupyter_Mineria_Datos_Supervisado_Clasificacion.git)
Ejecutar las celdas secuencialmente desde la Sección 1 hasta la Sección 8; el notebook descarga automáticamente el dataset 20251118_EVALUACION_DOCENTE_2024_20251118_PUBL.csv desde el repositorio en caso de no encontrarse en el entorno local.Para inspeccionar localmente la interfaz de MLflow tras ejecutar el notebook:Bashmlflow ui --backend-store-uri sqlite:///mlflow.db

Para cargarlo en GitHub, entra a `README.md` en tu rama `main
