import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def load_model(file_path):
    """Carga el modelo entrenado desde un archivo."""
    return joblib.load(file_path)

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

def evaluate_model(model, X, y):
    """Evalúa el modelo y devuelve métricas de rendimiento."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return accuracy, report, cm, fpr, tpr, roc_auc

def plot_confusion_matrix(cm, output_path):
    """Crea y guarda un gráfico de la matriz de confusión."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(model, feature_names, output_path):
    """Crea y guarda un gráfico de importancia de características."""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Características Más Importantes')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_roc_curve_plot(fpr, tpr, roc_auc):
    """Crea un gráfico interactivo de la curva ROC."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    
    return fig

if __name__ == "__main__":
    # Definir rutas de archivos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, "data", "processed", "featured_airlines_data.csv")
    model_path = os.path.join(project_root, "models", "trained_model.pkl")
    reports_dir = os.path.join(project_root, "reports")
    
    # Crear directorio de reportes si no existe
    os.makedirs(reports_dir, exist_ok=True)
    
    # Cargar datos y modelo
    print(f"Cargando datos desde {data_path}")
    df = load_data(data_path)
    
    # Preprocesar datos
    print("Preprocesando datos...")
    df = preprocess_data(df)
    
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    
    print(f"Cargando modelo desde {model_path}")
    model = load_model(model_path)
    
    # Evaluar modelo
    print("Evaluando modelo...")
    accuracy, report, cm, fpr, tpr, roc_auc = evaluate_model(model, X, y)
    print(f"Precisión del modelo: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(report)
    
    # Generar visualizaciones
    print("Generando visualizaciones...")
    plot_confusion_matrix(cm, os.path.join(reports_dir, "confusion_matrix.png"))
    plot_feature_importance(model, X.columns, os.path.join(reports_dir, "feature_importance.png"))
    
    # Crear y guardar gráfico de curva ROC
    roc_fig = create_roc_curve_plot(fpr, tpr, roc_auc)
    roc_fig.write_html(os.path.join(reports_dir, "roc_curve.html"))
    
    print("Evaluación completada. Visualizaciones guardadas en el directorio 'reports'.")