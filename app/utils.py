import joblib
import os

def load_model():
    """Carga el modelo entrenado desde el archivo."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model.pkl')
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en {model_path}")
        return None
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def make_prediction(model, data):
    """Realiza una predicción utilizando el modelo cargado."""
    if model is None:
        return None, None

    try:
        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1]  # Probabilidad de la clase positiva
        return prediction, probability
    except Exception as e:
        print(f"Error al hacer la predicción: {e}")
        return None, None