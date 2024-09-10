# Usa una imagen base oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo de requerimientos al directorio de trabajo
COPY requirements.txt .
# Instala las dependencias del sistema necesarias para psycopg2
RUN apt-get update && apt-get install -y postgresql libpq-dev

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación al directorio de trabajo
COPY . .

# Expone el puerto en el que la aplicación correrá
EXPOSE 8501

# Comando para ejecutar la aplicación
# CMD ["python", "src/models/train_model.py"]
CMD ["streamlit", "run", "app/main.py"]