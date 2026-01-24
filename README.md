# FlightOnTime – MVP de Predicción de Retrasos en Vuelos

**Hackathon de Data Science | Equipo de Ciencia de Datos**

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Origen de Datos](#origen-de-datos)
- [Objetivo del MVP](#objetivo-del-mvp)
- [Resultados Clave](#resultados-clave)
- [Dataset Utilizado](#dataset-utilizado)
- [Features Utilizadas](#features-utilizadas)
- [Instalación y Configuración](#instalación-y-configuración)
- [Cómo Usar el Modelo](#cómo-usar-el-modelo)
- [Modelo Escogido](#modelo-escogido)
- [Mejoras Futuras](#mejoras-futuras)
- [Referencias](#referencias)
- [Contacto](#contacto)

---

## Descripción del Proyecto

FlightOnTime es un modelo predictivo que estima la probabilidad de que un vuelo doméstico en Estados Unidos se retrase más de 15 minutos en su llegada, según el estándar oficial del Bureau of Transportation Statistics (BTS) del Departamento de Transporte de EE.UU.

El modelo está diseñado para ser integrado en una API REST que permita a aerolíneas, aeropuertos y pasajeros anticipar retrasos operativos y tomar decisiones proactivas.

---

## Origen de Datos

Se construye un dataframe a partir de una base de datos existente en Kaggle, que consta de todos los vuelos en el transcurso de un año en Estados Unidos. A esta se agregan las coordenadas de origen y destino. Luego, a partir de esta información, se agregan las variables meteorológicas consultando una API de Open-Meteo.com.

---

## Objetivo del MVP

Crear un modelo de clasificación binaria que, dado un conjunto de características de un vuelo antes de su salida, prediga si:
- **Puntual**: llega con menos de 15 minutos de retraso.
- **Retrasado**: llega con 15 minutos o más de retraso.

El modelo se entrega como un archivo serializado (*.ONNX) junto con metadatos para su uso en producción.

---

## Resultados Clave

| Métrica    | Valor |
|------------|-------|
| Precision  | 0.89  |
| F1-Score   | 0.87  |
| Recall     | 0.85  |
| Accuracy   | 0.87  |

**Interpretación**: El modelo captura correctamente 85% de los vuelos que realmente sufren retraso (Recall = 85%). De todas las alertas de "retraso" que genera el sistema, el 89% son correctas (Precision = 89%). Esto demuestra un modelo robusto y equilibrado (F1-Score de 0.87).

**Uso sugerido**: El modelo es apto para automatización de decisiones operativas, ya que tiene muy pocas falsas alarmas y, al mismo tiempo, se le escapan muy pocos incidentes reales.

---

## Dataset Utilizado

- **Nombre**: US Domestic Flights 2024
- **Fuente**: Kaggle - Flight Data 2024 (https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024)
- **Período**: Vuelos domésticos en EE.UU. durante el año 2024
- **Procesamiento**:
  - Se cruzaron datos con Open-Meteo.com mediante API, previo a agregar las coordenadas de cada aeropuerto.
  - Limpieza aplicada:
    - Eliminación de filas con NaN (71% de 1 millón de filas, debido a falta de datos climáticos).
    - Eliminación de vuelos cancelados o desviados.
    - Eliminación de variables de año, mes, día.
    - Creación de un diccionario numérico para aeropuertos y aerolíneas.
    - Eliminación de variables repetitivas (ciudad, longitud y latitud).
    - Ajuste de tipos de datos.
    - Variables categóricas: `op_unique_carrier`, `origin`, `dest`.
    - Revisión de correlaciones y eliminación de variables altamente correlacionadas.
    - Definición de retraso: `arr_delay >= 15` minutos.
    - Transformación de `arr_delay` a variable binaria.
    - Eliminación de variables de fuga (ej.: `dep_delay`, `carrier_delay`).
    - Reducción de uso de RAM mediante tipos optimizados (`category`, `float32`).
    - Aplicación de One-Hot Encoding.
    - Escalado Min-Max en variables numéricas.
    - Aplicación de SMOTE para balancear clases (50/50).

---

## Features Utilizadas (14)

El modelo utiliza exclusivamente información disponible antes del despegue del vuelo:

| #  | Column              | Non-Null Count | Dtype   |
|----|---------------------|----------------|---------|
| 0  | `day_of_week`      | 322,069        | float64 |
| 1  | `op_unique_carrier`| 322,069        | object  |
| 2  | `op_carrier_fl_num`| 322,069        | float64 |
| 3  | `origin`           | 322,069        | object  |
| 4  | `dest`             | 322,069        | object  |
| 5  | `crs_dep_time`     | 322,069        | float64 |
| 6  | `crs_arr_time`     | 322,069        | float64 |
| 7  | `distance`         | 322,069        | float64 |
| 8  | `temp_max`         | 322,069        | float64 |
| 9  | `rain_sum`         | 322,069        | float64 |
| 10 | `snow_sum`         | 322,069        | float64 |
| 11 | `wind_speed`       | 322,069        | float64 |
| 12 | `weather_code`     | 322,069        | float64 |
| 13 | `arr_delay_binary` | 322,069        | int64   |

---

## Instalación y Configuración

### Requisitos
- Python 3.8+
- Entorno virtual (venv recomendado)

### Pasos de Instalación
1. Clona o descarga el repositorio.
2. Activa el entorno virtual:
   ```bash
   .\.venv\Scripts\Activate.ps1  # En Windows
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Archivos del Proyecto
- `Flights_On_Time_Data_Science.ipynb`: Notebook principal para análisis y modelado.
- `df_vuelos_con_clima_ok.ipynb`: Notebook para preparación de datos con clima.
- `modelo_retrasos_light.onnx`: Modelo entrenado en formato ONNX.
- `model_metadata`: Metadatos del modelo (umbral óptimo, etc.).
- Archivos JSON: Datos procesados y gráficos.

---

## Cómo Usar el Modelo

1. **Cargar el modelo y metadatos**:
   ```python
   import json
   import pandas as pd
   from onnxruntime import InferenceSession

   # Cargar metadatos
   with open('model_metadata', 'r') as f:
       metadata = json.load(f)

   # Cargar modelo ONNX
   session = InferenceSession('modelo_retrasos_light.onnx')
   ```

2. **Preparar los datos de entrada**:
   El input debe ser un array o DataFrame con las 14 features en el orden exacto listado arriba.
   Las variables categóricas deben codificarse con el mismo esquema usado en entrenamiento (ver notebooks).

3. **Obtener predicción y probabilidad**:
   ```python
   # Ejemplo de input (reemplaza con datos reales)
   input_data = [[1, 'AA', 123, 'JFK', 'LAX', 800, 1100, 2475, 25.0, 0.0, 0.0, 10.0, 0, 0]]  # Ajusta según features

   # Obtener probabilidades
   probas = session.run(None, {'input': input_data})[0]
   prob_retraso = probas[0][1]  # Probabilidad de retraso (clase 1)

   # Umbral óptimo para F1-score (guardado en metadatos)
   umbral = metadata.get("umbral_optimo_f1", 0.5)  # Ej.: 0.327

   # Predicción final
   prediccion = "Retrasado" if prob_retraso >= umbral else "Puntual"
   ```

---

## Modelo Escogido

- **Nombre del Modelo**: Random Forest Classifier
- **Técnica de Balanceo**: SMOTE (50/50)
- **Parámetros**:
  - `n_estimators`: 35
  - `random_state`: 108

---

## Resumen de Métricas Finales

|                    | **Predicción: No Retraso** | **Predicción: Retraso** |
|--------------------|---------------------------|-------------------------|
| **Realidad: No Retraso** | 45,700 (TN)              | 5,364 (FP)             |
| **Realidad: Retraso**    | 7,544 (FN)               | 43,184 (TP)            |

**Métricas Derivadas**:
- **Precision**: 89.0% (Calidad de la alerta)
- **Recall**: 85.1% (Capacidad de detección)
- **F1-Score**: 0.87 (Balance general)
- **Accuracy**: 87.3% (Acierto global)

El modelo identificó que los factores climáticos son los determinantes más fuertes para predecir retrasos.

| Variable       | Peso (0-1) | Impacto Visual       |
|----------------|------------|----------------------|
| **temp_max**   | 0.1230    | ████████████▎       |
| **weather_code**| 0.1200    | ████████████        |
| **day_of_week**| 0.1170    | ███████████▋        |
| **wind_speed** | 0.1110    | ███████████         |
| **crs_dep_time**| 0.0965    | █████████▋          |
| **crs_arr_time**| 0.0940    | █████████▍          |
| **op_carrier_fl_num** | 0.0590 | █████▉              |
| **distance**   | 0.0429    | ████▎               |

---

## Mejoras Futuras

1. Pasar los modelos a un servidor.
2. Complementar con datos de vuelos internacionales.
3. Agregar aeropuertos fuera de USA.

---

## Referencias

- U.S. Department of Transportation (DOT). Air Travel Consumer Report.  
  https://www.transportation.gov/airconsumer
- Bureau of Transportation Statistics (BTS). On-Time Performance Data.  
  https://www.transtats.bts.gov/ONTIME/
- Kaggle Dataset: Flight Data 2024 by Hrishit Patil.  
  https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024
- Open-Meteo API: https://api.open-meteo.com/v1/forecast?...

---

## Contacto

Equipo de Ciencia de Datos – Hackathon FlightOnTime  
Modelo entrenado el: 22-01-2026

- Nombre: US Domestic Flights 2024
- Fuente: Kaggle - Flight Data 2024 (https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024)
- Periodo: Vuelos domésticos en EE.UU. durante el año 2024
- Se cruzo datos con Meteo.com, mediante una api, previo a agregar las coordenadas de cada aeropuerto. 
- Limpieza aplicada:
  - Se eliminan filas nan, que por tiempo no se alcanza a compilar con el clima (71%, de 1 millon de columnas)
  - Eliminación de vuelos cancelados o desviados.
  - Se eliminan variables de año, mes, dia.
  - se crea un diccionario numerico, para aeropuertos y aerolineas
  - Se eliminan variables repetitivas (ciudad, longitud y latitud)
  - Se ajustan variables, segun tipo de dato.
  - se deja las variables op_unique_carrier, origin y dest como variables categoricas
  - se revisa correlaciones y se borran las variables con alta correlacion. 
  - Definición de retraso: arr_delay >= 15 minutos.
  - se transforma la variable arr delay a binaria. 
  - Eliminación de variables de fuga (ej.: dep_delay, carrier_delay).
  - Reducción de uso de RAM mediante tipos optimizados (category, float32).
  - Se aplica un one-hot encoding
  - se aplica una escala de min-max en variables numericas. a exepcion de las variables 
  - Se aplica smoote, para tener la misma cantidad de vualos a tiempo, como vuelos retrasados.
 

---

 Features Utilizadas (9)

El modelo utiliza exclusivamente información disponible antes del despegue del vuelo:

| # | Column | Non-Null Count | Dtype |
| :--- | :--- | :--- | :--- |
| 0 | `day_of_week` | 322,069 non-null | float64 |
| 1 | `op_unique_carrier` | 322,069 non-null | object |
| 2 | `op_carrier_fl_num` | 322,069 non-null | float64 |
| 3 | `origin` | 322,069 non-null | object |
| 4 | `dest` | 322,069 non-null | object |
| 5 | `crs_dep_time` | 322,069 non-null | float64 |
| 6 | `crs_arr_time` | 322,069 non-null | float64 |
| 7 | `distance` | 322,069 non-null | float64 |
| 8 | `temp_max` | 322,069 non-null | float64 |
| 9 | `rain_sum` | 322,069 non-null | float64 |
| 10 | `snow_sum` | 322,069 non-null | float64 |
| 11 | `wind_speed` | 322,069 non-null | float64 |
| 12 | `weather_code` | 322,069 non-null | float64 |
| 13 | `arr_delay_binary` | 322,069 non-null | int64 |

Cómo Usar el Modelo

1. Cargar el modelo y metadatos
```python
# 1. Montar Drive (si no está montado)
drive.mount('/content/drive')

ruta_drive = "/content/drive/MyDrive/Dataset_Vuelos/ultima_prueba.json"

# Cargar el JSON
with open(ruta_drive, 'r') as f:
    data_json = json.load(f)

# 2. Usar json_normalize
df = pd.json_normalize(data_json)



print("Dimensiones del DataFrame:", df.shape)
df.head()
```

2. Preparar los datos de entrada
El input debe ser un array o DataFrame con las 9 features en el orden exacto listado arriba.  
Las variables categóricas deben codificarse con el mismo esquema usado en entrenamiento (ver notebook).

3. Obtener predicción y probabilidad

```python
# Probabilidad de retraso (clase 1)
prob_retraso = model.predict_proba(input_features)[0][1]

# Umbral óptimo para F1-score (guardado en metadatos)
umbral = metadata["umbral_optimo_f1"]  # Ej.: 0.327

# Predicción final
prevision = "Retrasado" if prob_retraso >= umbral else "Puntual"
```


---
Modelo escogido 

"nombre_modelo": "Random Forest Classifier",
    "tecnica_balanceo": "SMOTE (50/50)",
    "parametros": {
        "n_estimators": 35,
        "random_state": 108

  ---
**Resumen de Métricas Finales:**

|                    | **Predicción: No Retraso** | **Predicción: Retraso** |
| :---               | :---: | :---: |
| **Realidad: No Retraso** | **45,700** (TN) | 5,364 (FP) |
| **Realidad: Retraso** | 7,544 (FN) | **43,184** (TP) |

**Métricas Derivadas:**
* **Precision:** 89.0% (Calidad de la alerta)
* **Recall:** 85.1% (Capacidad de detección)
* **F1-Score:** 0.87 (Balance general)
* **Accuracy:** 87.3% (Acierto global)
  

El modelo identificó que los factores climáticos son los determinantes más fuertes para predecir retrasos.

| Variable       | Peso (0-1) | Impacto Visual       |
|----------------|------------|----------------------|
| **temp_max**   | 0.1230    | ████████████▎       |
| **weather_code**| 0.1200    | ████████████        |
| **day_of_week**| 0.1170    | ███████████▋        |
| **wind_speed** | 0.1110    | ███████████         |
| **crs_dep_time**| 0.0965    | █████████▋          |
| **crs_arr_time**| 0.0940    | █████████▍          |
| **op_carrier_fl_num** | 0.0590 | █████▉              |
| **distance**   | 0.0429    | ████▎               |

---

## Mejoras Futuras

1. Pasar los modelos a un servidor.
2. Complementar con datos de vuelos internacionales.
3. Agregar aeropuertos fuera de USA.

---

## Referencias

- U.S. Department of Transportation (DOT). Air Travel Consumer Report.  
  https://www.transportation.gov/airconsumer
- Bureau of Transportation Statistics (BTS). On-Time Performance Data.  
  https://www.transtats.bts.gov/ONTIME/
- Kaggle Dataset: Flight Data 2024 by Hrishit Patil.  
  https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024
- Open-Meteo API: https://api.open-meteo.com/v1/forecast?...

---

## Contacto

Equipo de Ciencia de Datos – Hackathon FlightOnTime  
Modelo entrenado el: 22-01-2026
