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

# Expose Dash default port
EXPOSE 8050

# Run the app
CMD ["python", "code/app.py"]
