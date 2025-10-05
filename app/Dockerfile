# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

COPY ./app /app/

# Copy model folder into container
# COPY model /app/model

# Expose Dash default port
EXPOSE 8050

ARG MLFLOW_TRACKING_URI

ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI

# Run the app
CMD ["python", "code/app.py"]