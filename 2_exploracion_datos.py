# Importación de bibliotecas necesarias
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np  # Para operaciones numéricas
import seaborn as sns  # Para visualización de datos estadísticos
import matplotlib.pyplot as plt  # Para creación de gráficos
import warnings  # Para gestionar advertencias
import os  # Para operaciones del sistema de archivos

# Ignorar advertencias para mantener el output limpio
warnings.filterwarnings('ignore')

# Crear directorio para guardar gráficos
os.makedirs('output', exist_ok=True)  # Crea el directorio 'output' si no existe

# ------------------------------------------------------------------------------------
# Carga de Datos
# ------------------------------------------------------------------------------------

# Verificar si el archivo existe
file_path = 'Electric_Vehicle_Population_Data_Cleaned.csv'
if not os.path.exists(file_path):
    print(f"ERROR: El archivo '{file_path}' no existe.")
    print("Por favor, ejecuta primero el script '1_limpieza_datos.py' para generar este archivo.")
    exit(1)  # Termina la ejecución con código de error

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(file_path)  # Lee el archivo CSV limpio y lo carga en un DataFrame

# ------------------------------------------------------------------------------------
# Exploración de Datos
# ------------------------------------------------------------------------------------

# Verificar valores faltantes después de la limpieza
print("\nValores faltantes por columna después de la limpieza:")
print(df.isna().sum())  # Muestra el número de valores faltantes por columna

# 1: Primeras Filas del DataFrame
print("\nPrimeras filas del DataFrame:")
print(df.head())  # Muestra las primeras 5 filas del DataFrame

# 2: Tipos de Datos de Cada Columna
print("\nTipos de datos de cada columna:")
print(df.dtypes)  # Muestra el tipo de datos de cada columna

# 3: Forma del DataFrame (Número de Filas y Columnas)
print("\nForma del DataFrame (filas, columnas):")
print(df.shape)  # Muestra el número de filas y columnas

# 4: Valores Faltantes
print("\nValores faltantes por columna:")
print(df.isna().sum())  # Muestra el número de valores faltantes por columna

# 5: Estadísticas Descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe(include="all"))  # Muestra estadísticas descriptivas para todas las columnas

# 6: Distribución de Variables Numéricas
# Visualizar la distribución de cada columna numérica con histogramas y KDE
# Limitamos a 5 columnas para no generar demasiados gráficos
for col in df.select_dtypes(include=[np.number]).columns[:5]:
    plt.figure(figsize=(8, 4))
    sns.histplot(x=col, data=df, kde=True)  # Crea un histograma con estimación de densidad
    plt.title(f'Distribución de {col}')
    # Guardar en la carpeta output
    plt.savefig(f'output/distribucion_{col}.png')
    plt.close()
    print(f"\nSe guardó el histograma de {col} en 'output/distribucion_{col}.png'")

# 7: Análisis de Correlación
# Crear un mapa de calor de correlación entre variables numéricas
numeric_df = df.select_dtypes(include=[np.number])  # Seleccionar solo las columnas numéricas
if not numeric_df.empty and numeric_df.shape[1] > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')  # Crea un mapa de calor de correlación
    plt.title('Mapa de Calor de Correlación')
    # Guardar en la carpeta output
    plt.savefig('output/correlation_heatmap_exploracion.png')
    plt.close()
    print("\nSe guardó el mapa de calor de correlación en 'output/correlation_heatmap_exploracion.png'")
else:
    print("\nNo hay suficientes columnas numéricas para crear un mapa de correlación.")

# 8: Distribución Geográfica
# Contar el número de vehículos por estado
if 'State' in df.columns:
    print("\nNúmero de vehículos por estado:")
    print(df['State'].value_counts())  # Muestra la frecuencia de cada estado
else:
    print("\nLa columna 'State' no está disponible en el dataset.")

# 9: Preferencias de Fabricantes y Modelos
# Contar el número de vehículos por fabricante
if 'Make' in df.columns:
    print("\nNúmero de vehículos por fabricante:")
    print(df['Make'].value_counts())  # Muestra la frecuencia de cada fabricante
else:
    print("\nLa columna 'Make' no está disponible en el dataset.")

# Contar el número de vehículos por modelo
if 'Model' in df.columns:
    print("\nNúmero de vehículos por modelo:")
    print(df['Model'].value_counts())  # Muestra la frecuencia de cada modelo
else:
    print("\nLa columna 'Model' no está disponible en el dataset.")

# 10: Tipos de Vehículos Eléctricos (BEV vs. PHEV)
# Contar el número de vehículos por tipo (BEV o PHEV)
if 'Electric Vehicle Type' in df.columns:
    print("\nNúmero de vehículos por tipo eléctrico:")
    print(df['Electric Vehicle Type'].value_counts())  # Muestra la frecuencia de cada tipo de vehículo eléctrico
else:
    print("\nLa columna 'Electric Vehicle Type' no está disponible en el dataset.")

print("\nExploración de datos completada con éxito.")