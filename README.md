FlightOnTime – MVP de Predicción de Retrasos en Vuelos

Hackathon de Data Science | Equipo de Ciencia de Datos

---

Descripción del Proyecto

FlightOnTime es un modelo predictivo que estima la probabilidad de que un vuelo doméstico en Estados Unidos se retrase  más de 15 minutos en su llegada, según el estándar oficial del Bureau of Transportation Statistics (BTS) del Departamento de Transporte de EE.UU.

El modelo está diseñado para ser integrado en una API REST que permita a aerolíneas, aeropuertos y pasajeros anticipar retrasos operativos y tomar decisiones proactivas.

---
Origen de Datos. 

Se construye un dataframe, a partir de una base de datos existentes en kaggle, que consta de todos los vuelos en el transcurso de un año, en estados unidos y a esta se agregan las coodenadas de origen y destino. Luego a partir de esta informacion, se agrega las variables meteoroligas, consultando a una Api, de Meteo.com

---

Objetivo del MVP

Crear un modelo de clasificación binaria que, dado un conjunto de características de un vuelo antes de su salida, prediga si:
- Puntual: llega con mas de 15 minutos de retraso.
- Retrasado: llega con 15 minutos o más de retraso.

El modelo se entrega como un archivo serializado (*.ONX) junto con metadatos para su uso en producción.

---

Resultados Clave

Métrica          | Valor
-----------------|------
Precision        | 0.85
F1-Score         | 0.86
Recall           | 0.86
Precisión        | 0.85

Basado en los valores de la imagen que subiste (0.85 y 0.86) y asumiendo el contexto de tu proyecto de predicción de retrasos en vuelos, aquí tienes la interpretación con el formato que solicitaste:

Interpretación: El modelo captura correctamente 86 de cada 100 vuelos que realmente sufren retraso (Recall = 86%). A su vez, de todas las alertas de "retraso" que genera el sistema, el 85% son correctas y realmente ocurren (Precision = 85%).

Esto demuestra un modelo muy robusto y equilibrado (F1-Score de 0.86). A diferencia de escenarios desbalanceados donde se sacrifica precisión para ganar cobertura, aquí el modelo es confiable en ambos frentes.

Uso sugerido: El modelo es apto para automatización de decisiones operativas, ya que tiene muy pocas falsas alarmas y, al mismo tiempo, se le escapan muy pocos incidentes reales. No requiere ajustes agresivos del umbral.   

---

 Dataset Utilizado

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

#   Column             Non-Null Count   Dtype  
---  ------             --------------   -----  
 0   day_of_week        322069 non-null  float64
 1   op_unique_carrier  322069 non-null  object 
 2   op_carrier_fl_num  322069 non-null  float64
 3   origin             322069 non-null  object 
 4   dest               322069 non-null  object 
 5   crs_dep_time       322069 non-null  float64
 6   crs_arr_time       322069 non-null  float64
 7   distance           322069 non-null  float64
 8   temp_max           322069 non-null  float64
 9   rain_sum           322069 non-null  float64
 10  snow_sum           322069 non-null  float64
 11  wind_speed         322069 non-null  float64
 12  weather_code       322069 non-null  float64
 13  arr_delay_binary   322069 non-null  int64
---

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
  
 "top_variables_importantes": 
        "temp_max": 0.123,
        "weather_code": 0.120,
        "day_of_week": 0.117,
        "wind_speed": 0.111,
        "crs_dep_time": 0.0965,
        "crs_arr_time": 0.094,
        "op_carrier_fl_num": 0.059,
        "distance": 0.0429,
 
        
----


Mejoras Futuras

3. Ajustar el umbral dinámicamente según el usuario:
   - Pasajeros: umbral alto → mayor precisión.
   - Aerolíneas: umbral bajo → mayor recall.
4. Probar modelos más avanzados (LightGBM, CatBoost) o ensamblados.

---

Referencias

- U.S. Department of Transportation (DOT). Air Travel Consumer Report.  
  https://www.transportation.gov/airconsumer
- Bureau of Transportation Statistics (BTS). On-Time Performance Data.  
  https://www.transtats.bts.gov/ONTIME/
- Kaggle Dataset: Flight Data 2024 by Hrishit Patil.  
  https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024
- https://api.open-meteo.com/v1/forecast?latitude=-33.393&longitude=-70.785&hourly=temperature_2m,dew_point_2m,precipitation,weather_code,visibility,wind_speed_10m,wind_direction_10m,wind_gusts_10m,freezing_level_height,cloud_cover_low,snow_depth,cape&wind_speed_unit=kn&timezone=auto&start_date=2026-01-08&end_date=2026-01-10
---

## Contacto

Equipo de Ciencia de Datos – Hackathon FlightOnTime  
Modelo entrenado el: 22-01-2026
