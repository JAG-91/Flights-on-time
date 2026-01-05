FlightOnTime – MVP de Predicción de Retrasos en Vuelos

Hackathon de Data Science | Equipo de Ciencia de Datos

---

Descripción del Proyecto

FlightOnTime es un modelo predictivo que estima la probabilidad de que un vuelo doméstico en Estados Unidos se retrase 15 minutos o más en su llegada, según el estándar oficial del Bureau of Transportation Statistics (BTS) del Departamento de Transporte de EE.UU.

El modelo está diseñado para ser integrado en una API REST que permita a aerolíneas, aeropuertos y pasajeros anticipar retrasos operativos y tomar decisiones proactivas.

---

Objetivo del MVP

Crear un modelo de clasificación binaria que, dado un conjunto de características de un vuelo antes de su salida, prediga si:
- Puntual: llega con menos de 15 minutos de retraso.
- Retrasado: llega con 15 minutos o más de retraso.

El modelo se entrega como un archivo serializado (*.pkl) junto con metadatos para su uso en producción.

---

Resultados Clave

Métrica          | Valor
-----------------|------
AUC-ROC          | 0.694
AUC-PR           | 0.366
F1-Score         | 0.415
Recall           | 61.7%
Precisión        | 31.8%

Interpretación: El modelo identifica correctamente 6 de cada 10 vuelos retrasados reales (Recall = 61.7%). Sin embargo, de los vuelos que marca como "retrasados", solo 3 de cada 10 realmente lo están (Precisión = 31.8%). Esto es común en escenarios altamente desbalanceados (~20% de retrasos).  
El modelo es útil para detección temprana (no perder vuelos retrasados).  
Para reducir falsas alarmas, se puede elevar el umbral de decisión (ver sección de uso).        

---

 Dataset Utilizado

- Nombre: US Domestic Flights 2024
- Fuente: Kaggle - Flight Data 2024 (https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024)
- Periodo: Vuelos domésticos en EE.UU. durante el año 2024
- Limpieza aplicada:
  - Eliminación de vuelos cancelados o desviados.
  - Definición de retraso: arr_delay >= 15 minutos.
  - Eliminación de variables de fuga (ej.: dep_delay, carrier_delay).
  - Reducción de uso de RAM mediante tipos optimizados (category, float32).

---

 Features Utilizadas (9)

El modelo utiliza exclusivamente información disponible antes del despegue del vuelo:

Feature   | Descripción     | Tipo
------------------------|-------------------------------------------------|-----------
op_unique_carrier       | Código de la aerolínea (ej.: "AA", "DL")        | Categórica
origin                  | Aeropuerto de origen (ej.: "ATL", "JFK")        | Categórica
dest                    | Aeropuerto de destino                           | Categórica
origin_state_nm         | Estado de origen (ej.: "California", "Texas")   | Categórica
dest_state_nm           | Estado de destino                               | Categórica
distance                | Distancia del vuelo (km)                        | Numérica
hora_salida_programada  | Hora programada de salida (0–23)                | Numérica
dia_semana              | Día de la semana (0 = lunes, 6 = domingo)       | Numérica
mes                     | Mes del año (1–12)                              | Numérica

No se usan variables posteriores al vuelo (ej.: retrasos reales, causas), garantizando validez en producción.

---

Cómo Usar el Modelo

1. Cargar el modelo y metadatos
```python
import joblib
import json

# Cargar modelo
model = joblib.load("flight_delay_model_xgb.pkl")

# Cargar metadatos
with open("model_metadata.json", "r") as f:
    metadata = json.load(f)
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

4. Resultado esperado (para API)
{
  "prevision": "Retrasado",
  "probabilidad": 0.78
}

---

Mejoras Futuras

1. Agregar features de congestión histórica por aeropuerto (ej.: % de retrasos promedio en origen/destino).
2. Incorporar datos meteorológicos en tiempo real (vía API externa).
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

---

## Contacto

Equipo de Ciencia de Datos – Hackathon FlightOnTime  
Modelo entrenado el: 2025-06-15
