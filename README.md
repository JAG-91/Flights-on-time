Flights-on-time

Prediccion de Retrasos en Vuelos – Proyecto de Analisis de Datos

Este proyecto implementa un flujo de trabajo optimizado para bajo consumo de RAM (< 3 GB) para predecir si un vuelo llegara con retraso, utilizando datos reales del sistema aereo de EE.UU. (2024). Esta disenado para ejecutarse en entornos limitados como Google Colab.

Caracteristicas principales:
- Carga eficiente: lectura selectiva de columnas y tipos de datos ligeros (float32, int8, category).
- Limpieza inteligente: filtrado de vuelos cancelados/desviados, creacion de variable objetivo binaria (retraso) y extraccion de hora de salida.
- Encoding selectivo: One-Hot Encoding solo en variables de baja cardinalidad (aerolineas, estados) para evitar explosion dimensional.
- Alta cardinalidad gestionada: variables como origin y dest se mantienen como tipo category (listas para target encoding avanzado si se requiere).
- Ahorro extremo de RAM: uso de dtype='int8' en dummies, eliminacion inmediata de archivos temporales y liberacion explicita de memoria.

Flujo del analisis:
1. Autenticacion en Kaggle (subida flexible de kaggle.json desde Colab).
2. Descarga y lectura directa desde ZIP sin descomprimir.
3. Filtrado y transformacion con operaciones in-place.
4. One-Hot Encoding selectivo (umbral configurable de cardinalidad).
5. Reporte de uso de RAM y recomendacion de muestreo si es necesario.

Variables clave utilizadas:
- Categoricas (One-Hot): op_unique_carrier, origin_state_nm, dest_state_nm.
- Alta cardinalidad (preservadas): origin, dest.
- Numericas: distance, dep_delay, nas_delay, late_aircraft_delay, etc.
- Temporal: hora_salida_programada (0–23).
- Objetivo: retraso (1 si arr_delay > 0, 0 en caso contrario).

Proximos pasos recomendados:
- Aplicar target encoding a origin y dest usando la tasa de retrasos por aeropuerto.
- Dividir en train/test con estratificacion por retraso.
- Entrenar modelos (Random Forest, XGBoost) con metricas como AUC, recall y F1.

Requisitos:
- Entorno: Google Colab (o cualquier sistema con Python, pandas, kaggle).
- Archivo kaggle.json de tu cuenta de Kaggle (disponible en kaggle.com/settings/account).

Ideal para cursos de machine learning, analisis de datos o proyectos con recursos limitados.
