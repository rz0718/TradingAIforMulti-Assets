# Use official Python 3.13.3 image (supports both AMD64 and ARM64)
FROM python:3.13.3-slim

# Set working directory
WORKDIR /app

# Configure runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADEBOT_DATA_DIR=/app/data

# Prepare data directory for volume mounting
RUN mkdir -p /app/data
VOLUME ["/app/data"]

# Install system dependencies (including bash for the startup script)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Expose Streamlit port
EXPOSE 8501

# Run both trading bot and dashboard
CMD ["./start.sh"]
