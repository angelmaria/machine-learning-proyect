import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

def get_db_connection():
    """Establece una conexión con la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

def create_table_if_not_exists():
    """Crea la tabla para almacenar predicciones si no existe."""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS customer_predictions (
                    id SERIAL PRIMARY KEY,
                    gender VARCHAR(10),
                    customer_type VARCHAR(20),
                    age INT,
                    type_of_travel VARCHAR(20),
                    customer_class VARCHAR(10),
                    flight_distance INT,
                    inflight_wifi_service INT,
                    departure_arrival_time_convenient INT,
                    ease_of_online_booking INT,
                    gate_location INT,
                    food_and_drink INT,
                    online_boarding INT,
                    seat_comfort INT,
                    inflight_entertainment INT,
                    on_board_service INT,
                    leg_room_service INT,
                    baggage_handling INT,
                    checkin_service INT,
                    inflight_service INT,
                    cleanliness INT,
                    departure_delay INT,
                    arrival_delay INT,
                    prediction BOOLEAN,
                    probability FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error al crear la tabla: {e}")
    finally:
        conn.close()

def insert_prediction(input_data, prediction, probability):
    """Inserta una nueva predicción en la base de datos."""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO customer_predictions 
                (gender, customer_type, age, type_of_travel, customer_class, flight_distance,
                inflight_wifi_service, departure_arrival_time_convenient, ease_of_online_booking,
                gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment,
                on_board_service, leg_room_service, baggage_handling, checkin_service, inflight_service,
                cleanliness, departure_delay, arrival_delay, prediction, probability)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(input_data['Gender'].iloc[0]),
                str(input_data['Customer Type'].iloc[0]),
                int(input_data['Age'].iloc[0]),
                str(input_data['Type of Travel'].iloc[0]),
                str(input_data['Class'].iloc[0]),
                int(input_data['Flight Distance'].iloc[0]),
                int(input_data['Inflight wifi service'].iloc[0]),
                int(input_data['Departure/Arrival time convenient'].iloc[0]),
                int(input_data['Ease of Online booking'].iloc[0]),
                int(input_data['Gate location'].iloc[0]),
                int(input_data['Food and drink'].iloc[0]),
                int(input_data['Online boarding'].iloc[0]),
                int(input_data['Seat comfort'].iloc[0]),
                int(input_data['Inflight entertainment'].iloc[0]),
                int(input_data['On-board service'].iloc[0]),
                int(input_data['Leg room service'].iloc[0]),
                int(input_data['Baggage handling'].iloc[0]),
                int(input_data['Checkin service'].iloc[0]),
                int(input_data['Inflight service'].iloc[0]),
                int(input_data['Cleanliness'].iloc[0]),
                int(input_data['Departure Delay in Minutes'].iloc[0]),
                int(input_data['Arrival Delay in Minutes'].iloc[0]),
                bool(prediction),
                float(probability)
            ))
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error al insertar la predicción: {e}")
    finally:
        conn.close()

# Crear la tabla al importar este módulo
create_table_if_not_exists()