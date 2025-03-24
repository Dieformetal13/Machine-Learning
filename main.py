import os  # Para operaciones del sistema de archivos
import subprocess  # Para ejecutar procesos externos
import sys  # Para acceder a variables y funciones específicas del sistema
import matplotlib  # Para configuración de gráficos

# Configurar matplotlib para no mostrar gráficos interactivos
# Esto evita que se abran ventanas de gráficos durante la ejecución
matplotlib.use('Agg')


def check_requirements():
    """
    Verifica que todas las dependencias estén instaladas.
    Intenta instalar las dependencias desde requirements.txt si es necesario.

    Returns:
        bool: True si todas las dependencias están instaladas, False en caso contrario.
    """
    print("Verificando dependencias...")
    try:
        # Intenta instalar las dependencias si no están instaladas
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Todas las dependencias están instaladas correctamente.")
        return True
    except subprocess.CalledProcessError:
        print("ERROR: No se pudieron instalar las dependencias.")
        return False
    except FileNotFoundError:
        print("ADVERTENCIA: No se encontró el archivo 'requirements.txt'.")
        print("Continuando sin verificar dependencias...")
        return True


def run_script(script_name):
    """
    Ejecuta un script de Python y verifica si se ejecutó correctamente.

    Args:
        script_name (str): Nombre del script a ejecutar.

    Returns:
        bool: True si el script se ejecutó correctamente, False en caso contrario.
    """
    print(f"\n{'=' * 50}")
    print(f"Ejecutando {script_name}...")
    print(f"{'=' * 50}\n")

    # Configurar variables de entorno para matplotlib
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'  # Forzar el backend no interactivo

    # Ejecutar el script con las variables de entorno modificadas
    try:
        result = subprocess.run([sys.executable, script_name], env=env, capture_output=False)

        if result.returncode == 0:
            print(f"\n✅ {script_name} se ejecutó correctamente.")
            return True
        else:
            print(f"\n❌ Error al ejecutar {script_name}.")
            return False
    except Exception as e:
        print(f"\n❌ Error al ejecutar {script_name}: {e}")
        return False


def main():
    """
    Función principal que coordina la ejecución de todos los scripts.
    Gestiona el flujo de trabajo completo del análisis de datos.
    """
    print("🚗 ANÁLISIS DE DATOS DE VEHÍCULOS ELÉCTRICOS 🔋")
    print("=" * 50)

    # Crear directorio para guardar gráficos
    os.makedirs('output', exist_ok=True)
    print("Se ha creado el directorio 'output' para guardar los gráficos.")

    # Verificar dependencias
    if not check_requirements():
        return

    # Verificar si existe el archivo de datos original
    if not os.path.exists('Electric Vehicle Population Data.csv'):
        print("\n⚠️ ADVERTENCIA: No se encontró el archivo 'Electric Vehicle Population Data.csv'")
        print("Por favor, descarga el dataset y colócalo en la misma carpeta que este script.")
        print(
            "Puedes encontrar datasets similares en: https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2")

        continuar = input("\n¿Deseas continuar de todos modos? (s/n): ").lower()
        if continuar != 's':
            print("Análisis cancelado.")
            return

    # Ejecutar scripts en orden
    scripts = [
        "1_limpieza_datos.py",
        "2_exploracion_datos.py",
        "3_random_forest.py"
    ]

    for script in scripts:
        if not os.path.exists(script):
            print(f"\n❌ Error: El archivo '{script}' no existe.")
            print(f"Asegúrate de que el archivo '{script}' esté en la misma carpeta que este script.")
            return

        if not run_script(script):
            print(f"Se detuvo la ejecución debido a un error en {script}.")
            return

    print("\n🎉 ¡Análisis completado con éxito! 🎉")
    print("\nArchivos generados:")
    print("  - Electric_Vehicle_Population_Data_Cleaned.csv (Dataset limpio)")
    print("  - encoding_mappings.json (Mapeo de codificación)")
    print("  - random_forest_classifier_model.pkl (Modelo entrenado)")
    print("  - Gráficos guardados en el directorio 'output/'")


if __name__ == "__main__":
    main()