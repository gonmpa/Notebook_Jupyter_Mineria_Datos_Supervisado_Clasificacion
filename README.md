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
