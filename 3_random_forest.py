# Importación de bibliotecas necesarias
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np  # Para operaciones numéricas
import seaborn as sns  # Para visualización de datos estadísticos
import matplotlib.pyplot as plt  # Para creación de gráficos
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score  # Para división y validación de datos
from sklearn.preprocessing import StandardScaler, label_binarize  # Para preprocesamiento de datos
from sklearn.ensemble import RandomForestClassifier  # Algoritmo de clasificación Random Forest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # Para evaluación del modelo
import joblib  # Para guardar y cargar modelos
import warnings  # Para gestionar advertencias
import os  # Para operaciones del sistema de archivos

# Ignorar advertencias para mantener el output limpio
warnings.filterwarnings('ignore')

# Crear directorio para guardar gráficos
os.makedirs('output', exist_ok=True)  # Crea el directorio 'output' si no existe

# ------------------------------------------------------------------------------------
# Carga de la Base de Datos Limpia
# ------------------------------------------------------------------------------------

# Verificar si el archivo existe
cleaned_file_path = 'Electric_Vehicle_Population_Data_Cleaned.csv'
if not os.path.exists(cleaned_file_path):
    print(f"ERROR: El archivo '{cleaned_file_path}' no existe.")
    print("Por favor, ejecuta primero el script '1_limpieza_datos.py' para generar este archivo.")
    exit(1)  # Termina la ejecución con código de error

# Cargar el archivo CSV limpio en un DataFrame
df_cleaned = pd.read_csv(cleaned_file_path)  # Lee el archivo CSV limpio y lo carga en un DataFrame

# Mostrar las primeras filas del DataFrame para inspeccionar los datos
print("Primeras filas del DataFrame limpio:")
print(df_cleaned.head())  # Muestra las primeras 5 filas del DataFrame

# Verificar la forma del DataFrame (número de filas y columnas)
print("\nForma del DataFrame limpio (filas, columnas):")
print(df_cleaned.shape)  # Muestra el número de filas y columnas

# ------------------------------------------------------------------------------------
# Preparación de Datos para Modelado
# ------------------------------------------------------------------------------------

# Verificar si la columna objetivo existe
if 'Electric Range Category' not in df_cleaned.columns:
    print("ERROR: La columna 'Electric Range Category' no existe en el dataset.")
    print("Por favor, asegúrate de que el script de limpieza haya creado esta columna.")
    exit(1)  # Termina la ejecución con código de error

# Definir las variables independientes (X) y la variable dependiente (y)
# X contiene todas las columnas excepto 'Electric Range Category' y 'Electric Range'
X = df_cleaned.drop(columns=['Electric Range Category'])
if 'Electric Range' in df_cleaned.columns:
    X = X.drop(columns=['Electric Range'])  # Eliminamos 'Electric Range' para evitar fuga de datos

y = df_cleaned['Electric Range Category']  # Variable objetivo (categorías de autonomía eléctrica)

# Verificar que tenemos datos suficientes
if len(y.unique()) < 2:
    print(f"ERROR: La variable objetivo solo tiene {len(y.unique())} categorías. Se necesitan al menos 2.")
    print(f"Categorías disponibles: {y.unique()}")
    exit(1)  # Termina la ejecución con código de error

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binarizar las etiquetas para la curva ROC multiclase (solo para y_test)
try:
    classes = sorted(y.unique())  # Obtener las clases únicas ordenadas
    y_test_binarized = label_binarize(y_test, classes=classes)  # Binarizar las etiquetas
    print(f"\nClases para clasificación: {classes}")
except Exception as e:
    print(f"\nError al binarizar etiquetas: {e}")
    print("Continuando sin binarización para ROC...")
    y_test_binarized = None

# Escalar las variables numéricas para mejorar el rendimiento del modelo
scaler = StandardScaler()  # Inicializar el escalador
X_train = scaler.fit_transform(X_train)  # Ajustar y transformar los datos de entrenamiento
X_test = scaler.transform(X_test)  # Transformar los datos de prueba con el mismo escalador

# ------------------------------------------------------------------------------------
# Modelado con Random Forest Classifier
# ------------------------------------------------------------------------------------

# Crear el modelo Random Forest Classifier
# n_estimators=100: número de árboles en el bosque
# random_state=42: semilla para reproducibilidad
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
print("\nEntrenando el modelo Random Forest...")
rf_classifier.fit(X_train, y_train)  # Ajustar el modelo a los datos de entrenamiento
print("Entrenamiento completado.")

# Predecir las categorías para los conjuntos de entrenamiento y prueba
y_train_pred = rf_classifier.predict(X_train)  # Predicciones en datos de entrenamiento
y_test_pred = rf_classifier.predict(X_test)  # Predicciones en datos de prueba

# ------------------------------------------------------------------------------------
# Evaluación del Modelo de Clasificación
# ------------------------------------------------------------------------------------

# Classification Report: muestra precision, recall, f1-score para cada clase
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Matriz de Confusión: muestra verdaderos positivos, falsos positivos, etc.
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
# Guardar en la carpeta output
plt.savefig('output/confusion_matrix.png')
plt.close()
print("\nSe guardó la matriz de confusión en 'output/confusion_matrix.png'")

# Curva ROC y AUC (One-vs-Rest para multiclase)
# Evalúa la capacidad del modelo para distinguir entre clases
if y_test_binarized is not None and len(classes) > 1:
    plt.figure(figsize=(8, 6))
    for i in range(len(classes)):
        if i < y_test_binarized.shape[1]:  # Verificar que el índice es válido
            try:
                # Calcular la curva ROC para cada clase
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], rf_classifier.predict_proba(X_test)[:, i])
                roc_auc = auc(fpr, tpr)  # Calcular el área bajo la curva
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {classes[i]}')
            except Exception as e:
                print(f"Error al calcular curva ROC para clase {classes[i]}: {e}")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal de referencia
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC (One-vs-Rest)')
    plt.legend(loc='lower right')
    # Guardar en la carpeta output
    plt.savefig('output/roc_curve.png')
    plt.close()
    print("\nSe guardó la curva ROC en 'output/roc_curve.png'")
else:
    print("\nNo se pudo generar la curva ROC debido a problemas con la binarización o número insuficiente de clases.")

# Validación Cruzada (k-fold cross-validation)
# Evalúa el rendimiento del modelo en diferentes subconjuntos de datos
try:
    cv_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
    print("\nResultados de Validación Cruzada (k=5):")
    print(f"Puntuaciones de cada fold: {cv_scores}")
    print(f"Precisión promedio: {cv_scores.mean():.2f}")
except Exception as e:
    print(f"\nError en la validación cruzada: {e}")
    print("Omitiendo validación cruzada...")

# Importancia de Características
# Muestra qué variables son más importantes para el modelo
feature_importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)  # Ordenar por importancia descendente

print("\nImportancia de las Características:")
print(feature_importance_df)

# Visualizar la importancia de las características (mostrar solo las 15 más importantes)
plt.figure(figsize=(10, 6))
top_features = feature_importance_df.head(15)  # Seleccionar las 15 características más importantes
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Importancia de las Características en el Modelo Random Forest (Top 15)')
plt.xlabel('Importancia')
plt.ylabel('Característica')
# Guardar en la carpeta output
plt.savefig('output/feature_importance.png')
plt.close()
print("\nSe guardó el gráfico de importancia de características en 'output/feature_importance.png'")

# ------------------------------------------------------------------------------------
# Guardado del Modelo
# ------------------------------------------------------------------------------------

# Guardar el modelo entrenado para uso futuro
model_file_path = 'random_forest_classifier_model.pkl'
joblib.dump(rf_classifier, model_file_path)  # Guardar el modelo en un archivo
print(f"\nModelo guardado en: {model_file_path}")

print("\nAnálisis con Random Forest completado con éxito.")