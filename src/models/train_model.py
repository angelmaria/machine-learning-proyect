import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint
import joblib
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocesa los datos, codificando variables categóricas."""
    le = LabelEncoder()
    
    # Codificar 'age_group' si existe
    if 'age_group' in df.columns:
        df['age_group'] = le.fit_transform(df['age_group'])
    
    # Codificar otras variables categóricas si es necesario
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    return df

def split_data(X, y, test_size=0.2, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def perform_cross_validation(model, X, y, cv=5):
    """Realiza validación cruzada y devuelve las puntuaciones."""
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
    return accuracy_scores, f1_scores, precision_scores, recall_scores

def optimize_hyperparameters(X, y):
    """Optimiza los hiperparámetros utilizando búsqueda aleatoria."""
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11)
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    rand_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=100, 
        cv=5, random_state=42, n_jobs=-1,
        scoring=['accuracy', 'f1', 'precision', 'recall'],
        refit='accuracy'
    )
    
    rand_search.fit(X, y)
    return rand_search

def train_model(X, y):
    """Entrena el modelo con los mejores hiperparámetros."""
    print("Realizando validación cruzada inicial...")
    base_model = RandomForestClassifier(random_state=42)
    accuracy_scores, f1_scores, precision_scores, recall_scores = perform_cross_validation(base_model, X, y)
    
    print("\nOptimizando hiperparámetros...")
    rand_search = optimize_hyperparameters(X, y)
    best_model = rand_search.best_estimator_
    print("Mejores hiperparámetros encontrados:")
    print(best_model.get_params())

    print("\nEntrenando modelo final...")
    best_model.fit(X, y)
    
    # Crear visualización
    create_performance_visualization(rand_search, accuracy_scores, f1_scores, precision_scores, recall_scores)
    
    return best_model

def evaluate_overfitting(model, X_train, y_train, X_test, y_test):
    """Evalúa el overfitting comparando el rendimiento en entrenamiento y prueba."""
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    overfitting = train_accuracy - test_accuracy
    
    print(f"Accuracy en entrenamiento: {train_accuracy:.4f}")
    print(f"Accuracy en prueba: {test_accuracy:.4f}")
    print(f"Diferencia (overfitting): {overfitting:.4f}")
    
    if overfitting > 0.05:
        print("ADVERTENCIA: Posible overfitting detectado (diferencia > 5%)")
    else:
        print("No se detecta overfitting significativo")
    
    return overfitting

def save_model(model, file_path):
    """Guarda el modelo entrenado en un archivo."""
    joblib.dump(model, file_path)
    print(f"Modelo guardado en: {file_path}")

def create_performance_visualization(rand_search, initial_accuracy, initial_f1, initial_precision, initial_recall):
    """Crea una visualización interactiva de las métricas de rendimiento."""
    iterations = list(range(1, len(rand_search.cv_results_['mean_test_accuracy']) + 1))
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Accuracy", "F1-Score", "Precision", "Recall"))
    
    # Accuracy
    fig.add_trace(go.Scatter(x=iterations, y=rand_search.cv_results_['mean_test_accuracy'],
                             mode='lines+markers', name='Accuracy (Optimización)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1], y=[initial_accuracy.mean()],
                             mode='markers', name='Accuracy Inicial', marker=dict(size=10)), row=1, col=1)
    
    # F1-Score
    fig.add_trace(go.Scatter(x=iterations, y=rand_search.cv_results_['mean_test_f1'],
                             mode='lines+markers', name='F1-Score (Optimización)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[1], y=[initial_f1.mean()],
                             mode='markers', name='F1-Score Inicial', marker=dict(size=10)), row=1, col=2)
    
    # Precision
    fig.add_trace(go.Scatter(x=iterations, y=rand_search.cv_results_['mean_test_precision'],
                             mode='lines+markers', name='Precision (Optimización)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[1], y=[initial_precision.mean()],
                             mode='markers', name='Precision Inicial', marker=dict(size=10)), row=2, col=1)
    
    # Recall
    fig.add_trace(go.Scatter(x=iterations, y=rand_search.cv_results_['mean_test_recall'],
                             mode='lines+markers', name='Recall (Optimización)'), row=2, col=2)
    fig.add_trace(go.Scatter(x=[1], y=[initial_recall.mean()],
                             mode='markers', name='Recall Inicial', marker=dict(size=10)), row=2, col=2)
    
    fig.update_layout(height=800, width=1000, title_text="Evolución de Métricas de Rendimiento")
    fig.write_html("reports/performance_metrics_evolution.html")
    print("Visualización de métricas guardada en 'reports/performance_metrics_evolution.html'")

if __name__ == "__main__":
    # Definir rutas de archivos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, "data", "processed", "featured_airlines_data.csv")
    model_path = os.path.join(project_root, "models", "trained_model.pkl")
    
    # Cargar datos
    print(f"Cargando datos desde {data_path}")
    df = load_data(data_path)
    
    # Preprocesar datos
    print("Preprocesando datos...")
    df = preprocess_data(df)
    
    # Dividir datos en características y etiqueta
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Entrenar el modelo
    model = train_model(X_train, y_train)
    
    # Evaluar overfitting
    print("\nEvaluando overfitting...")
    overfitting = evaluate_overfitting(model, X_train, y_train, X_test, y_test)
    
    # Guardar el modelo entrenado
    save_model(model, model_path)