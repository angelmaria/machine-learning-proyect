import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Añadir el directorio raíz del proyecto al PATH para poder importar los módulos personalizados
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.preprocess_data import preprocess_data
from src.features.feature_engineering import engineer_features
from app.database import insert_prediction
from app.utils import load_model, make_prediction

def main():
    st.title('F5 Airlines - Predicción de Satisfacción del Cliente')

    # Cargar el modelo
    model = load_model()

    if model is None:
        st.error("No se pudo cargar el modelo. Por favor, verifica que el archivo del modelo existe y es accesible.")
        return

    # Crear formulario para la entrada de datos
    with st.form("customer_data_form"):
        st.write("Por favor, ingrese los datos del cliente:")
        
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox('Género', ['Male', 'Female'])
            customer_type = st.selectbox('Tipo de Cliente', ['Loyal Customer', 'disloyal Customer'])
            age = st.number_input('Edad', min_value=0, max_value=120, value=30)
            type_of_travel = st.selectbox('Tipo de Viaje', ['Personal Travel', 'Business travel'])
            customer_class = st.selectbox('Clase', ['Eco', 'Eco Plus', 'Business'])
            flight_distance = st.number_input('Distancia de Vuelo', min_value=0, value=1000)
            wifi_service = st.slider('Servicio de WiFi', 0, 5, 3)
            departure_arrival_convenience = st.slider('Conveniencia de Salida/Llegada', 0, 5, 3)
            online_booking_ease = st.slider('Facilidad de Reserva Online', 0, 5, 3)
            gate_location = st.slider('Ubicación de la Puerta', 0, 5, 3)

        with col2:
            food_and_drink = st.slider('Comida y Bebida', 0, 5, 3)
            online_boarding = st.slider('Embarque Online', 0, 5, 3)
            seat_comfort = st.slider('Comodidad del Asiento', 0, 5, 3)
            inflight_entertainment = st.slider('Entretenimiento a Bordo', 0, 5, 3)
            onboard_service = st.slider('Servicio a Bordo', 0, 5, 3)
            leg_room_service = st.slider('Espacio para Piernas', 0, 5, 3)
            baggage_handling = st.slider('Manejo de Equipaje', 0, 5, 3)
            checkin_service = st.slider('Servicio de Check-in', 0, 5, 3)
            inflight_service = st.slider('Servicio en Vuelo', 0, 5, 3)
            cleanliness = st.slider('Limpieza', 0, 5, 3)
            departure_delay = st.number_input('Retraso en la Salida (minutos)', min_value=0, value=0)
            arrival_delay = st.number_input('Retraso en la Llegada (minutos)', min_value=0, value=0)

        submitted = st.form_submit_button("Predecir Satisfacción")

    if submitted:
        # Crear un DataFrame con los datos ingresados
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Customer Type': [customer_type],
            'Age': [age],
            'Type of Travel': [type_of_travel],
            'Class': [customer_class],
            'Flight Distance': [flight_distance],
            'Inflight wifi service': [wifi_service],
            'Departure/Arrival time convenient': [departure_arrival_convenience],
            'Ease of Online booking': [online_booking_ease],
            'Gate location': [gate_location],
            'Food and drink': [food_and_drink],
            'Online boarding': [online_boarding],
            'Seat comfort': [seat_comfort],
            'Inflight entertainment': [inflight_entertainment],
            'On-board service': [onboard_service],
            'Leg room service': [leg_room_service],
            'Baggage handling': [baggage_handling],
            'Checkin service': [checkin_service],
            'Inflight service': [inflight_service],
            'Cleanliness': [cleanliness],
            'Departure Delay in Minutes': [departure_delay],
            'Arrival Delay in Minutes': [arrival_delay]
        })

        # Preprocesar los datos
        try:
            preprocessed_data = preprocess_data(input_data)
            
            # Aplicar ingeniería de características
            featured_data = engineer_features(preprocessed_data)

            # Hacer la predicción
            prediction, probability = make_prediction(model, featured_data)

            if prediction is not None and probability is not None:
                # Mostrar el resultado
                st.subheader('Resultado de la Predicción:')
                if prediction[0] == 1:
                    st.success(f'El cliente está satisfecho con una probabilidad del {probability[0]*100:.2f}%')
                else:
                    st.error(f'El cliente está insatisfecho con una probabilidad del {(1-probability[0])*100:.2f}%')

                # Guardar la predicción en la base de datos
                try:
                    insert_prediction(input_data, prediction[0], probability[0])
                    st.info('Los datos y la predicción han sido guardados en la base de datos.')
                except Exception as e:
                    st.warning(f'No se pudo guardar la predicción en la base de datos: {str(e)}')
            else:
                st.error('No se pudo realizar la predicción. Por favor, verifica los datos de entrada.')
        except Exception as e:
            st.error(f'Ocurrió un error al procesar los datos: {str(e)}')

if __name__ == "__main__":
    main()