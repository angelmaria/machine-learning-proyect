import pandas as pd
import numpy as np
import os

def create_total_rating(df):
    """
    Crea una nueva característica que es la suma de todas las calificaciones de servicio.
    """
    rating_columns = ['Inflight wifi service', 'Departure/Arrival time convenient',
                      'Ease of Online booking', 'Gate location', 'Food and drink',
                      'Online boarding', 'Seat comfort', 'Inflight entertainment',
                      'On-board service', 'Leg room service', 'Baggage handling',
                      'Checkin service', 'Inflight service', 'Cleanliness']
    
    # Usar solo las columnas que existen en el DataFrame
    existing_columns = [col for col in rating_columns if col in df.columns]
    
    if existing_columns:
        df['total_rating'] = df[existing_columns].sum(axis=1)
    else:
        df['total_rating'] = 0  # O cualquier otro valor predeterminado
    return df

def create_total_delay(df):
    """
    Crea una característica que suma el retraso de salida y llegada.
    """
    if 'Departure Delay in Minutes' in df.columns and 'Arrival Delay in Minutes' in df.columns:
        df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    else:
        df['total_delay'] = 0  # O cualquier otro valor predeterminado
    return df

def bin_age(df):
    """
    Crea categorías de edad.
    """
    if 'Age' in df.columns:
        df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 35, 55, 100], labels=['0-18', '19-35', '36-55', '55+'])
    else:
        df['age_group'] = 'Unknown'  # O cualquier otro valor predeterminado
    return df

def engineer_features(df):
    """
    Aplica todas las transformaciones de ingeniería de características.
    """
    df = create_total_rating(df)
    df = create_total_delay(df)
    df = bin_age(df)
    return df

if __name__ == "__main__":
    # Definir rutas de archivos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    input_file = os.path.join(project_dir, 'data', 'processed', 'cleaned_airlines_data.csv')
    output_file = os.path.join(project_dir, 'data', 'processed', 'featured_airlines_data.csv')
    
    # Verificar si el archivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Error: El archivo de entrada {input_file} no existe.")
        print("Asegúrate de que el archivo CSV con los datos preprocesados esté en la carpeta 'data/processed/'.")
        exit(1)
    
    # Cargar datos
    print(f"Cargando datos preprocesados desde {input_file}")
    df = pd.read_csv(input_file)
    
    # Aplicar ingeniería de características
    print("Aplicando ingeniería de características...")
    df_featured = engineer_features(df)
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Guardar el nuevo DataFrame con las características ingenieradas
    df_featured.to_csv(output_file, index=False)
    print(f"Datos con nuevas características guardados en {output_file}")