FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . /app

# Install system dependencies and AWS CLI
RUN apt-get update -y && \
    apt-get install -y awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the app port
EXPOSE 8080

# Run your FastAPI app (uses app_run in app.py)
CMD ["python", "app.py"]
