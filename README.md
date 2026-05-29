# Aplicación de Redes Neuronales para la clasificación de ingresos anual (> $50K)

Este repositorio contiene un proyecto enfocado en el ciclo completo de diseño, optimización, interpretación y despliegue de un modelo de Machine Learning basado en una red neuronal profunda tipo **Perceptrón Multicapa (MLP)** para datos tabulares estructurados.

## 👥 Integrantes del Equipo
* **David Barrientos**
* **Marcello Anchante**
* **Sebastian Rios**
* **Selene Ramos**

---
 ## 🌍 Contexto
La predicción de ingresos a partir de datos sociodemográficos tiene aplicaciones directas en sectores como finanzas, políticas públicas y sistemas de crédito. El dataset Adult/Census Income, recopilado del Censo de EE.UU. de 1994, es un benchmark clásico en Machine Learning que permite estudiar el comportamiento de modelos en datos tabulares reales con desbalance de clases, valores faltantes y variables mixtas.
Este proyecto aplica el ciclo completo de ML con énfasis en redes neuronales profundas (MLP), priorizando la interpretabilidad y reproducibilidad del proceso.

## 🎯 Objetivo del Proyecto
Construir, ajustar, interpretar y desplegar un modelo predictivo basado en redes neuronales para clasificar si los ingresos anuales de un individuo superan los \$50,000 dólares (`>50K` o `<=50K`), utilizando el conjunto de datos histórico **Adult / Census Income Dataset** de la Oficina de Censos de los EE. UU. (1994).

El proyecto demuestra el dominio de:
`Preprocesamiento` ➡️ `Ingeniería de Variables` ➡️ `Modelado & Tuning` ➡️ `Ablation Study` ➡️ `Evaluación & Interpretabilidad` ➡️ `Despliegue Interactivo`

---
## 📊 Descripción del Dataset
| Atributo         | Detalle                              |
|------------------|--------------------------------------|
| Fuente           | UCI Machine Learning Repository      |
| Año              | 1994 (Censo EE.UU.)                  |
| Instancias       | 48,842 registros                     |
| Variables        | 14 features + 1 variable objetivo    |
| Tipos            | 6 numéricas / 8 categóricas          |
| Variable Target  | Ingreso `>50K` o `<=50K` (binaria)   |
| Desbalance       | ~76% clase ≤50K / ~24% clase >50K   |

🔗 [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

---
## 📋 Variables del Dataset
| Variable       | Tipo        | Descripción                        |
|----------------|-------------|------------------------------------|
| age            | Numérica    | Edad del individuo                 |
| education-num  | Numérica    | Años de educación                  |
| capital-gain   | Numérica    | Ganancia de capital registrada     |
| occupation     | Categórica  | Tipo de ocupación laboral          |
| hours-per-week | Numérica    | Horas trabajadas por semana        |
| income         | **Target**  | >50K o <=50K (variable objetivo)   |

---
## 📂 Estructura del Repositorio

El proyecto está organizado de manera secuencial siguiendo los requerimientos estrictos de la rúbrica del examen:

```text
├── 01_eda.ipynb                           # Análisis Exploratorio de Datos y Calidad de Datos
├── 02_preprocesamiento_feature_engineering.ipynb # Limpieza, codificación y creación de variables (No Leakage)
├── 03_modelo_keras.py                     # Implementación y tuning de la red neuronal en Keras
├── 04_modelo_pytorch_o_sklearn.py         # Implementación alternativa en PyTorch / Scikit-Learn
├── 05_ablation_study.ipynb                # Estudio de ablación comparativo incremental
├── 06_evaluacion_interpretabilidad.ipynb  # Métricas avanzadas, análisis de errores y SHAP/Permutations
├── app_streamlit/                         # Carpeta de la aplicación interactiva para predicción en vivo
│   └── app.py                             # Código principal de la interfaz en Streamlit
├── modelo_final/                          # Modelos entrenados y serializados (pesos y pipelines)
├── requirements.txt                       # Dependencias del proyecto (pip)
└── README.md                              # Documentación general del proyecto (este archivo)
```

---

<img width="1408" height="768" alt="Gemini_Generated_Image_x3e1xhx3e1xhx3e1" src="https://github.com/user-attachments/assets/06025091-b591-430f-8ac4-8c52cc61f805" />

---

## 🛠️ Requisitos e Instalación

Para asegurar la **reproducibilidad** total del proyecto (punto crítico de penalización), siga los siguientes pasos para configurar su entorno local:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Marcelloprime/rnn_adult_income_prediction.git
   cd rnn_adult_income_prediction
   ```

2. **Crear y activar un entorno virtual (Recomendado):**
   * En Linux/macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * En Windows (Git Bash / PowerShell):
     ```bash
     python -m venv venv
     source venv/Scripts/activate
     ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---
## 🤖 Modelos Implementados
Este proyecto entrena y compara tres modelos de Machine Learning sobre el mismo
conjunto de datos y condiciones controladas de experimentación.

## Modelo 1 — MLP con Keras / TensorFlow
Red neuronal profunda tipo Perceptrón Multicapa construida con la API funcional
de Keras sobre TensorFlow 2.x. Es el modelo principal del proyecto.
Su arquitectura final consiste en dos capas ocultas de 128 y 64 neuronas con
activación ReLU, BatchNormalization y Dropout entre capas para regularización,
y una capa de salida con activación Sigmoid para la clasificación binaria.
Los hiperparámetros fueron optimizados automáticamente con Optuna en 30 trials.

## Modelo 2 — MLP con PyTorch
Implementación equivalente de la misma arquitectura MLP usando PyTorch puro.
Su propósito es verificar que los resultados del modelo Keras son consistentes
e independientes del framework utilizado. Comparte la misma lógica de
entrenamiento: optimizador Adam, EarlyStopping, class weights y tuning con Optuna.

## Modelo 3 — XGBoost
Modelo de gradient boosting incluido como punto de comparación externo.
Representa el estado del arte en clasificación con datos tabulares estructurados
y permite evaluar si la complejidad de las redes neuronales está justificada
frente a un modelo más clásico. También fue optimizado con Optuna en 30 trials.

