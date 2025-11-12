# Use official Python 3.13.3 image (supports both AMD64 and ARM64)
FROM python:3.13.3-slim

# Set working directory
WORKDIR /app

# Configure runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADEBOT_DATA_DIR=/app/data

    
# Install system dependencies (including bash for the startup script)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements file
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Make startup scripts executable
RUN chmod +x start.sh docker-entrypoint.py

# Expose Streamlit port
EXPOSE 8501

# Set the entrypoint (use Python-based entrypoint for better reliability)
ENTRYPOINT ["python3", "/workspace/docker-entrypoint.py"]

