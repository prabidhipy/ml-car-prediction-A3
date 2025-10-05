FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app folder
COPY ./app /app

EXPOSE 8050

ARG MLFLOW_TRACKING_URI
ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI

# Run app.py inside code folder
CMD ["python", "code/app.py"]
