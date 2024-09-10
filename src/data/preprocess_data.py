import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path):
    """
    Carga los datos desde un archivo CSV.

    Args:
        file_path (str): La ruta al archivo CSV que se va a cargar.

    Returns:
        pd.DataFrame: Un DataFrame de pandas que contiene los datos cargados.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocesa los datos para el análisis y modelado.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos a preprocesar.

    Returns:
        pd.DataFrame: El DataFrame preprocesado con las transformaciones aplicadas.
    """

    # Crear una copia del DataFrame para evitar advertencias de SettingWithCopyWarning
    df = df.copy()

    # Eliminar columnas innecesarias que podrían no ser útiles para el análisis
    df = df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore') 

    # Manejar valores faltantes en la columna 'Arrival Delay in Minutes'
    if 'Arrival Delay in Minutes' in df.columns:
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean())

    # Codificar variables categóricas utilizando LabelEncoder
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    if 'satisfaction' in df.columns:  # Incluir 'satisfaction' si existe (posiblemente la variable objetivo)
        categorical_cols.append('satisfaction')

    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str)) 

    # Normalizar variables numéricas para que tengan media 0 y desviación estándar 1
    numerical_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df

# Bloque principal que se ejecuta cuando el script se ejecuta directamente
if __name__ == "__main__":
    # Definir rutas de archivos utilizando la estructura del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    input_file = os.path.join(project_dir, 'data', 'raw', 'airline_passenger_satisfaction.csv')
    output_file = os.path.join(project_dir, 'data', 'processed', 'cleaned_airlines_data.csv')

    # Cargar datos desde el archivo CSV de entrada
    print(f"Cargando datos desde {input_file}")
    df_raw = load_data(input_file)

    # Preprocesar los datos cargados
    print("Preprocesando datos...")
    df_processed = preprocess_data(df_raw)

    # Crear el directorio de salida si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Guardar los datos preprocesados en un nuevo archivo CSV
    df_processed.to_csv(output_file, index=False)
    print(f"Datos preprocesados guardados en {output_file}")