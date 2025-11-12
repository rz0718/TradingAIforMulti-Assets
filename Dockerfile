ARG PRIVATE_AWS_ECR_URL
FROM ${PRIVATE_AWS_ECR_URL}/pluangpython:3.9-v2

WORKDIR /workspace

# Configure runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADEBOT_DATA_DIR=/workspace/data

    
# Install system dependencies (including bash for the startup script)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    bash \
    && rm -rf /var/lib/apt/lists/*

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

ENTRYPOINT ["python"]

CMD ["docker-entrypoint.py"]
