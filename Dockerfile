FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies (updated repo URLs & AWS CLI)
RUN apt-get update -y && \
    apt-get install -y awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python3", "app.py"]
