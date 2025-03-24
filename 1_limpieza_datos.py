# Importación de bibliotecas necesarias
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np  # Para operaciones numéricas
import seaborn as sns  # Para visualización de datos estadísticos
import matplotlib.pyplot as plt  # Para creación de gráficos
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder  # Para transformación de datos
from sklearn.model_selection import train_test_split  # Para dividir datos en conjuntos de entrenamiento y prueba
import warnings  # Para gestionar advertencias
import json  # Para manejar archivos JSON
import os  # Para operaciones del sistema de archivos
import datetime  # Para operaciones con fechas y horas

# Ignorar advertencias para mantener el output limpio
warnings.filterwarnings('ignore')

# Crear directorio para guardar gráficos
os.makedirs('output', exist_ok=True)  # Crea el directorio 'output' si no existe

# ------------------------------------------------------------------------------------
# Carga de Datos
# ------------------------------------------------------------------------------------

# Verificar si el archivo existe, si no, mostrar instrucciones
file_path = 'Electric Vehicle Population Data.csv'
if not os.path.exists(file_path):
    print(f"ERROR: El archivo '{file_path}' no existe.")
    print(
        "Por favor, descarga el dataset 'Electric Vehicle Population Data.csv' y colócalo en la misma carpeta que este script.")
    print(
        "Puedes encontrar datasets similares en: https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2")
    exit(1)  # Termina la ejecución con código de error

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(file_path)  # Lee el archivo CSV y lo carga en un DataFrame de pandas

# Mostrar las primeras filas del DataFrame para inspeccionar los datos
print("Primeras filas del DataFrame:")
print(df.head())  # Muestra las primeras 5 filas del DataFrame

# Verificar los tipos de datos de cada columna
print("\nTipos de datos de cada columna:")
print(df.dtypes)  # Muestra el tipo de datos de cada columna

# Verificar la forma del DataFrame (número de filas y columnas)
print("\nForma del DataFrame (filas, columnas):")
print(df.shape)  # Muestra el número de filas y columnas

# Verificar si hay filas duplicadas
print("\nNúmero de filas duplicadas:")
print(df.duplicated().sum())  # Cuenta el número de filas duplicadas

# Verificar valores faltantes en cada columna
print("\nValores faltantes por columna:")
print(df.isna().sum())  # Cuenta los valores faltantes en cada columna

# Obtener información general del DataFrame
print("\nInformación general del DataFrame:")
print(df.info())  # Muestra información general sobre el DataFrame

# Estadísticas descriptivas de las columnas numéricas y categóricas
print("\nEstadísticas descriptivas:")
print(df.describe(include="all"))  # Muestra estadísticas descriptivas para todas las columnas

# ------------------------------------------------------------------------------------
# Limpieza y Procesamiento de Datos
# ------------------------------------------------------------------------------------

# Paso 1: Identificación de Columnas Relevantes
# Seleccionar las columnas que son relevantes para el análisis
try:
    # Lista de columnas que consideramos relevantes para nuestro análisis
    columnas_relevantes = [
        'County', 'City', 'State', 'Postal Code', 'Model Year', 'Make', 'Model',
        'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility',
        'Electric Range', 'Base MSRP', 'Legislative District', 'Vehicle Location',
        'Electric Utility', '2020 Census Tract'
    ]
    # Verificar que todas las columnas existen en el DataFrame
    missing_columns = [col for col in columnas_relevantes if col not in df.columns]
    if missing_columns:
        print(f"\nADVERTENCIA: Las siguientes columnas no existen en el DataFrame: {missing_columns}")
        columnas_relevantes = [col for col in columnas_relevantes if col in df.columns]
        print(f"Continuando con las columnas disponibles: {columnas_relevantes}")

    # Seleccionar solo las columnas relevantes
    df = df[columnas_relevantes]
except Exception as e:
    print(f"\nError al seleccionar columnas: {e}")
    print("Continuando con todas las columnas disponibles.")

# Paso 2: Manejo de Valores Faltantes
print("\nValores faltantes por columna después de seleccionar columnas relevantes:")
print(df.isna().sum())  # Verificar valores faltantes después de seleccionar columnas

# Eliminar filas con valores faltantes en columnas clave
# Estas columnas son esenciales para nuestro análisis
required_columns = ['Electric Range', 'Model Year', 'Make', 'Model']
available_required = [col for col in required_columns if col in df.columns]
if available_required:
    df = df.dropna(subset=available_required)  # Elimina filas con valores faltantes en columnas clave
    print(f"\nSe eliminaron filas con valores faltantes en: {available_required}")

# Imputar valores faltantes en otras columnas numéricas con la mediana
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
print(f"\nSe imputaron valores faltantes en columnas numéricas con la mediana.")

# Paso 3: Eliminación de Duplicados
print("\nNúmero de filas duplicadas antes de la eliminación:", df.duplicated().sum())
df = df.drop_duplicates()  # Elimina filas duplicadas
print("Número de filas duplicadas después de la eliminación:", df.duplicated().sum())

# Paso 4: Conversión de Tipos de Datos
# Asegurarse de que las columnas tengan el tipo de dato correcto
if 'Model Year' in df.columns:
    try:
        # Convertir 'Model Year' a formato de año
        df['Model Year'] = pd.to_datetime(df['Model Year'], format='%Y').dt.year
        print("\nSe convirtió 'Model Year' a tipo numérico (año)")
    except Exception as e:
        print(f"\nError al convertir 'Model Year': {e}")
        print("Intentando convertir directamente a entero...")
        try:
            df['Model Year'] = df['Model Year'].astype(int)
            print("Conversión exitosa.")
        except:
            print("No se pudo convertir 'Model Year'. Manteniendo el tipo original.")

if 'Postal Code' in df.columns:
    df['Postal Code'] = df['Postal Code'].astype(str)  # Convertir códigos postales a string
    print("Se convirtió 'Postal Code' a tipo string")

# Paso 5: Normalización y Codificación de Variables Categóricas
# Codificar variables categóricas para que puedan ser utilizadas en modelos de machine learning
label_encoder = LabelEncoder()  # Inicializar el codificador de etiquetas

# Definir las columnas categóricas que se van a codificar
categorical_columns = [col for col in ['County', 'City', 'State', 'Make', 'Model',
                                       'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility',
                                       'Vehicle Location', 'Electric Utility']
                       if col in df.columns]

# Crear un diccionario para almacenar los mapeos de cada columna categórica
encoding_mappings = {}

# Codificar cada columna categórica y guardar el mapeo
for col in categorical_columns:
    # Asegurarse de que la columna sea de tipo string antes de codificar
    df[col] = df[col].astype(str)
    # Aplicar la codificación
    df[col] = label_encoder.fit_transform(df[col])
    # Guardar el mapeo de cada columna en el diccionario
    encoding_mappings[col] = {k: int(v) for k, v in
                              zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

# Guardar el mapeo en un archivo JSON para referencia futura
with open('encoding_mappings.json', 'w') as f:
    json.dump(encoding_mappings, f, indent=4)
print("\nMapeo de codificación guardado en 'encoding_mappings.json'.")

# Paso 6: Análisis de Outliers
if 'Electric Range' in df.columns:
    # Crear un boxplot para visualizar la distribución y detectar outliers
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['Electric Range'])
    plt.title('Boxplot de Electric Range')
    # Guardar en la carpeta output
    plt.savefig('output/boxplot_electric_range.png')
    plt.close()
    print("\nSe guardó el boxplot de Electric Range en 'output/boxplot_electric_range.png'")

    # Eliminar outliers (valores por encima del percentil 99)
    q99 = df['Electric Range'].quantile(0.99)
    df = df[df['Electric Range'] <= q99]  # Filtrar valores extremos
    print(f"\nSe eliminaron outliers de 'Electric Range' por encima del percentil 99 ({q99}).")

# Paso 7: Creación de Variables Derivadas
# Crear nuevas variables que puedan ser útiles para el análisis
if 'Model Year' in df.columns:
    # Calcular la edad del vehículo (año actual - año del modelo)
    df['Vehicle Age'] = datetime.datetime.now().year - df['Model Year']
    print("\nSe creó la variable 'Vehicle Age' (edad del vehículo).")

# Discretizar la variable objetivo (Electric Range) en categorías
if 'Electric Range' in df.columns:
    # Definir los límites de las categorías
    bins = [0, 100, 200, float('inf')]  # Rangos: 0-100, 100-200, >200
    labels = ['Bajo', 'Medio', 'Alto']  # Etiquetas para las categorías
    # Crear la nueva columna categórica
    df['Electric Range Category'] = pd.cut(df['Electric Range'], bins=bins, labels=labels)
    print("Se creó la variable 'Electric Range Category' con categorías: Bajo, Medio, Alto.")

    # Eliminar filas con valores faltantes en 'Electric Range Category'
    df = df.dropna(subset=['Electric Range Category'])
    print("Se eliminaron filas con valores faltantes en 'Electric Range Category'.")

# Paso 8: Redondear flotantes a 3 decimales
# Redondear todas las columnas numéricas a 3 decimales para mejorar la legibilidad
df = df.round(3)
print("\nSe redondearon todas las columnas numéricas a 3 decimales.")

# Paso 9: Análisis de Correlación
# Realizar un análisis de correlación entre las variables numéricas
numeric_df = df.select_dtypes(include=[np.number])  # Seleccionar solo las columnas numéricas

if not numeric_df.empty:
    # Crear un mapa de calor de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Mapa de Calor de Correlación')
    # Guardar en la carpeta output
    plt.savefig('output/heatmap_correlacion.png')
    plt.close()
    print("\nSe guardó el mapa de calor de correlación en 'output/heatmap_correlacion.png'")
else:
    print("\nNo hay suficientes columnas numéricas para crear un mapa de correlación.")

# ------------------------------------------------------------------------------------
# Guardado de la Base de Datos Resultante
# ------------------------------------------------------------------------------------

# Guardar el DataFrame Limpio en un archivo CSV
output_file_path = 'Electric_Vehicle_Population_Data_Cleaned.csv'
df.to_csv(output_file_path, index=False)  # Guardar sin índices
print(f"\nBase de datos limpia guardada en: {output_file_path}")
print(f"Número de filas en el dataset limpio: {df.shape[0]}")
print(f"Número de columnas en el dataset limpio: {df.shape[1]}")