# ANÁLISIS DE DATOS DE VEHÍCULOS ELÉCTRICOS
===========================================

## Descripción General

Este proyecto realiza un análisis completo de datos de vehículos eléctricos, desde la limpieza y procesamiento de los datos hasta la exploración visual y la creación de un modelo predictivo utilizando Random Forest.

## Requisitos

- Python 3.6 o superior
- Bibliotecas requeridas (instaladas automáticamente mediante requirements.txt):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - joblib

## Estructura de Archivos

- `main.py`: Script principal que coordina la ejecución de todos los demás scripts.
- `1_limpieza_datos.py`: Realiza la limpieza y procesamiento de los datos originales.
- `2_exploracion_datos.py`: Explora y visualiza los datos limpios.
- `3_random_forest.py`: Crea y evalúa un modelo de clasificación Random Forest.
- `requirements.txt`: Lista de dependencias necesarias.
- `Electric Vehicle Population Data.csv`: Archivo de datos original (debe ser descargado).
- `Electric_Vehicle_Population_Data_Cleaned.csv`: Archivo de datos limpio (generado por el script 1).
- `encoding_mappings.json`: Mapeo de variables categóricas (generado por el script 1).
- `random_forest_classifier_model.pkl`: Modelo entrenado (generado por el script 3).
- `output/`: Directorio donde se guardan todos los gráficos generados.

## Instrucciones de Ejecución

1. **Preparación**:
   - Descarga el dataset "Electric Vehicle Population Data.csv".
   - Coloca el archivo en la misma carpeta que los scripts.

2. **Ejecución**:
   - Ejecuta el script principal: `python main.py`
   - El script verificará e instalará las dependencias necesarias, y ejecutará los tres scripts de análisis en secuencia.
   - Alternativamente, puedes ejecutar cada script individualmente en este orden:
     1. `python 1_limpieza_datos.py`
     2. `python 2_exploracion_datos.py`
     3. `python 3_random_forest.py`

3. **Resultados**:
   - Los resultados se mostrarán en la consola durante la ejecución.
   - Los gráficos se guardarán en el directorio `output/`.
   - El modelo entrenado se guardará como `random_forest_classifier_model.pkl`.

## Descripción de los Scripts

### 1. Limpieza de Datos (`1_limpieza_datos.py`)

Este script realiza las siguientes tareas:
- Carga el dataset original
- Selecciona columnas relevantes
- Maneja valores faltantes
- Elimina duplicados
- Convierte tipos de datos
- Codifica variables categóricas
- Elimina outliers
- Crea variables derivadas (edad del vehículo, categoría de autonomía)
- Guarda el dataset limpio y el mapeo de codificación

### 2. Exploración de Datos (`2_exploracion_datos.py`)

Este script realiza las siguientes tareas:
- Carga el dataset limpio
- Muestra estadísticas descriptivas
- Visualiza la distribución de variables numéricas
- Analiza correlaciones entre variables
- Muestra la distribución geográfica de vehículos
- Analiza preferencias de fabricantes y modelos
- Examina tipos de vehículos eléctricos

### 3. Modelado con Random Forest (`3_random_forest.py`)

Este script realiza las siguientes tareas:
- Carga el dataset limpio
- Prepara los datos para el modelado
- Entrena un modelo Random Forest para clasificar vehículos según su autonomía
- Evalúa el modelo mediante:
  - Matriz de confusión
  - Reporte de clasificación
  - Curvas ROC
  - Validación cruzada
- Analiza la importancia de las características
- Guarda el modelo entrenado

## Resultados Esperados

1. **Limpieza de Datos**:
   - Dataset limpio sin valores faltantes ni duplicados
   - Variables categóricas codificadas
   - Nuevas variables derivadas

2. **Exploración de Datos**:
   - Visualizaciones de distribuciones
   - Mapa de calor de correlaciones
   - Estadísticas sobre fabricantes, modelos y tipos de vehículos

3. **Modelo Predictivo**:
   - Modelo de clasificación Random Forest
   - Métricas de rendimiento (precisión, recall, F1-score)
   - Visualización de la importancia de características

## Interpretación de Resultados

- **Matriz de Confusión**: Muestra cuántos vehículos fueron clasificados correctamente en cada categoría de autonomía.
- **Curvas ROC**: Evalúa la capacidad del modelo para distinguir entre las diferentes categorías.
- **Importancia de Características**: Identifica qué características tienen mayor influencia en la autonomía eléctrica de un vehículo.

## Posibles Problemas y Soluciones

1. **Error al cargar el dataset original**:
   - Asegúrate de haber descargado el archivo y colocarlo en la carpeta correcta.
   - Verifica que el nombre del archivo sea exactamente "Electric Vehicle Population Data.csv".

2. **Error al instalar dependencias**:
   - Instala manualmente las bibliotecas requeridas: `pip install pandas numpy matplotlib seaborn scikit-learn joblib`.

3. **Gráficos no se muestran**:
   - Los gráficos no se muestran interactivamente, sino que se guardan en la carpeta 'output/'.
   - Verifica que la carpeta 'output/' exista y tenga permisos de escritura.

4. **Errores en la ejecución de scripts individuales**:
   - Asegúrate de ejecutar los scripts en el orden correcto.
   - El script 2 y 3 dependen de que el script 1 se haya ejecutado correctamente.

## Notas Adicionales

- El modelo está configurado para clasificar vehículos en tres categorías de autonomía, pero estos umbrales pueden ajustarse en el script `1_limpieza_datos.py`.
- Para usar el modelo entrenado en nuevos datos, puedes cargarlo con `joblib.load('random_forest_classifier_model.pkl')`.
- Los mapeos de codificación se guardan en 'encoding_mappings.json' para referencia futura y para decodificar las predicciones si es necesario.
